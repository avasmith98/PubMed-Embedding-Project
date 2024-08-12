import hashlib
import ftplib
import gzip
from io import BytesIO
import os
import xml.etree.ElementTree as ET
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# FTP server details
ftp_server = "ftp.ncbi.nlm.nih.gov"
ftp_directory = "/pubmed/baseline/"
file_pattern = "pubmed24n{:04d}.xml.gz"
md5_file_pattern = "pubmed24n{:04d}.xml.gz.md5"

# OpenAI client setup
open_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_2"))
MODEL = "text-embedding-3-small"

# Qdrant client setup
qdrant_client = QdrantClient(host='localhost', port=6333)

def generate_text_embedding(text, model=MODEL):
    response = open_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def ensure_collection_exists(client, collection_name, vector_size):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def parse_pubmed_articles(data, max_articles): #get rid of max_articles when ready to run on full dataset
    root = ET.fromstring(data)
    articles_data = []
    article_count = 0

    for medline_citation in root.findall(".//MedlineCitation"):
        if article_count >= max_articles:
            break
        
        article_data = {}

        # Extract CommentsCorrections (to handle updates and retractions)
        comments_corrections = medline_citation.findall(".//CommentsCorrections")
        is_retracted = any(comment.attrib.get('RefType', '') in ["Retraction of", "Retraction in"] for comment in comments_corrections)
        if is_retracted:
            continue  # Skip retracted articles
        
        # Extract Abstract
        abstract_elements = medline_citation.findall(".//Abstract/AbstractText")
        if abstract_elements:
            abstract_texts = [abstract.text for abstract in abstract_elements if abstract.text]
            article_data['Abstract'] = ' '.join(abstract_texts)
        else:
            continue  # Skip articles without an abstract

        # Extract PMID
        article_data['PMID'] = medline_citation.find("PMID").text if medline_citation.find("PMID") is not None else ''
        article_data['PMID_Version'] = medline_citation.find("PMID").attrib.get('Version', '') if medline_citation.find("PMID") is not None else ''

        # Extract Journal Information
        journal = medline_citation.find(".//Journal")
        if journal is not None:
            article_data['Journal'] = {
                'Title': journal.find("Title").text if journal.find("Title") is not None else '',
                'Volume': journal.find(".//JournalIssue/Volume").text if journal.find(".//JournalIssue/Volume") is not None else '',
                'PubDate': {
                    'Year': journal.find(".//JournalIssue/PubDate/Year").text if journal.find(".//JournalIssue/PubDate/Year") is not None else '',
                    'Month': journal.find(".//JournalIssue/PubDate/Month").text if journal.find(".//JournalIssue/PubDate/Month") is not None else '',
                    'Day': journal.find(".//JournalIssue/PubDate/Day").text if journal.find(".//JournalIssue/PubDate/Day") is not None else ''
                }
            }
        else:
            article_data['Journal'] = {
                'Title': '',
                'Volume': '',
                'PubDate': {
                    'Year': '',
                    'Month': '',
                    'Day': ''
                }
            }

        # Extract Article Title
        article_data['Title'] = medline_citation.find(".//ArticleTitle").text if medline_citation.find(".//ArticleTitle") is not None else ''

        # Extract Authors
        authors = medline_citation.findall(".//AuthorList/Author")
        complete_yn = medline_citation.find(".//AuthorList").attrib.get('CompleteYN', 'Y')
        author_list = []
        for author in authors:
            author_data = {
                'LastName': author.find("LastName").text if author.find("LastName") is not None else '',
                'ForeName': author.find("ForeName").text if author.find("ForeName") is not None else '',
            }
            author_list.append(author_data)
        
        article_data['Authors'] = author_list
        article_data['AuthorsComplete'] = complete_yn == 'Y'

        # Extract Keywords
        keywords = medline_citation.findall(".//Keyword")
        article_data['Keywords'] = [keyword.text for keyword in keywords if keyword.text]

        # Extract Publication Identifiers
        article_data['PublicationIdentifiers'] = {
            'DOI': medline_citation.find(".//ELocationID[@EIdType='doi']").text if medline_citation.find(".//ELocationID[@EIdType='doi']") is not None else '',
        }

        articles_data.append(article_data)
        article_count += 1

    return articles_data

def generate_payload(article_data):
    # Generate text embedding for the abstract
    embedding = generate_text_embedding(article_data['Abstract'])
    
    payload = {
        "pmid": article_data['PMID'],
        "pmid_version": article_data['PMID_Version'],
        "title": article_data['Title'],
        "abstract": article_data['Abstract'],
        "authors": article_data['Authors'],
        "authors_complete": article_data['AuthorsComplete'],
        "journal": {
            "title": article_data['Journal']['Title'],
            "volume": article_data['Journal']['Volume'],
            "pub_date": {
                "year": article_data['Journal']['PubDate']['Year'],
                "month": article_data['Journal']['PubDate']['Month'],
                "day": article_data['Journal']['PubDate']['Day']
            }
        },
        "keywords": article_data['Keywords'],
        "publication_identifiers": article_data['PublicationIdentifiers'],
        "embedding": embedding  # Add the generated embedding to the payload
    }
    return payload

def insert_payload(client, payload, collection_name):
    point = PointStruct(
        id=int(payload['pmid']),  # Use the PMID as the ID
        vector=payload['embedding'],  # Use the generated embedding
        payload=payload
    )
    response = client.upsert(collection_name=collection_name, points=[point])
    return response

def process_and_upload(file_name, compressed_data, collection_name):
    with gzip.GzipFile(fileobj=compressed_data, mode='rb') as f_in:
        extracted_data = f_in.read()
    
    articles_data = parse_pubmed_articles(extracted_data, max_articles=10)  # get rid of max_articles when ready to run on full dataset
    if articles_data:
        for article_data in articles_data:
            payload = generate_payload(article_data)
            response = insert_payload(qdrant_client, payload, collection_name=collection_name)
            print(response)
    return True

def main():
    ftp = ftplib.FTP(ftp_server)
    ftp.login()
    ftp.cwd(ftp_directory)

    collection_name = "PubMed"
    vector_size = 1536
    ensure_collection_exists(qdrant_client, collection_name, vector_size)

    for i in range(1, 2):  # Adjust the range up to 1220 for the full dataset. Goes from pumed24n0001.xml.gz to pubmed24n1220.xml.gz.
        file_name = file_pattern.format(i)
        md5_file_name = md5_file_pattern.format(i)
        
        # Retrieve and check MD5
        md5_data = BytesIO()
        ftp.retrbinary(f"RETR {md5_file_name}", md5_data.write)
        md5_contents = md5_data.getvalue().decode().strip()
        expected_md5 = md5_contents.split('=')[1].strip() if '=' in md5_contents else md5_contents.split()[0].strip()

        # Retrieve the compressed file
        compressed_data = BytesIO()
        ftp.retrbinary(f"RETR {file_name}", compressed_data.write)
        
        # Calculate MD5 for the compressed data
        calculated_md5 = hashlib.md5(compressed_data.getvalue()).hexdigest()

        if calculated_md5 == expected_md5:
            compressed_data.seek(0)  # Reset buffer position
            print(f"Checksums matched. Processing file {file_name}...")
            process_and_upload(file_name, compressed_data, collection_name)
        else:
            print(f"MD5 mismatch for file {file_name}. Expected: {expected_md5}, Calculated: {calculated_md5}")

    ftp.quit()

if __name__ == "__main__":
    main()
