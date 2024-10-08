import hashlib
import ftplib
import gzip
from io import BytesIO
import os
import ollama
import xml.etree.ElementTree as ET
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import logging

# FTP server details
ftp_server = "ftp.ncbi.nlm.nih.gov"
ftp_directory = "/pubmed/baseline/"
file_pattern = "pubmed24n{:04d}.xml.gz"
md5_file_pattern = "pubmed24n{:04d}.xml.gz.md5"

# Qdrant client setup
qdrant_client = QdrantClient(host='localhost', port=6333)

# Setup logging
logging.basicConfig(filename='pubmed_processing.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# File to store the last processed PMID
PMID_FILE = 'last_pmid.txt'


def write_last_pmid(pmid):
    """Writes the last processed PMID to a file."""
    with open(PMID_FILE, 'w') as f:
        f.write(str(pmid))


def read_last_pmid():
    """Reads the last processed PMID from a file."""
    if os.path.exists(PMID_FILE):
        with open(PMID_FILE, 'r') as f:
            return int(f.read().strip())
    return None


def generate_bgem3_embedding(text, model='bge-m3'):
    response = ollama.embeddings(model=model, prompt=text)
    logging.info(f"Generated BGEM3 embedding for text of length {len(text)}")
    return response['embedding']


def generate_bge_large_embedding(text, model='bge-large'):
    response = ollama.embeddings(model=model, prompt=text)
    logging.info(f"Generated BGE Large embedding for text of length {len(text)}")
    return response['embedding']


def ensure_collection_exists(client, collection_name):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "bgem3_embedding": VectorParams(size=1024, distance=Distance.COSINE),
                "bge_large_embedding": VectorParams(size=1024, distance=Distance.COSINE)
            }
        )
        logging.info(f"Created new collection {collection_name}")
    else:
        logging.info(f"Collection {collection_name} already exists")


def parse_pubmed_articles(data, max_articles): 
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

    logging.info(f"Parsed {article_count} articles from PubMed data")
    return articles_data


def generate_payload(article_data):
    # Generate text embedding for the abstract
    bgem3_embedding = generate_bgem3_embedding(article_data['Abstract'])  
    bge_large_embedding = generate_bge_large_embedding(article_data['Abstract'])

    # Create the payload (metadata)
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
        "bgem3_embedding": bgem3_embedding,
        "bge_large_embedding": bge_large_embedding    
    }
    logging.info(f"Generated payload for PMID {article_data['PMID']}")
    return payload


def insert_payload(client, payload, collection_name):
    # Vectors (embeddings) go into the 'vector' field
    vectors = { 
        "bgem3_embedding": payload['bgem3_embedding'], 
        "bge_large_embedding": payload["bge_large_embedding"]
    }

    # Metadata (payload) excluding the vectors
    metadata_payload = {
        "pmid": payload['pmid'],
        "pmid_version": payload['pmid_version'],
        "title": payload['title'],
        "abstract": payload['abstract'],
        "authors": payload['authors'],
        "authors_complete": payload['authors_complete'],
        "journal": payload['journal'],
        "keywords": payload['keywords'],
        "publication_identifiers": payload['publication_identifiers']
    }

    # PointStruct requires the id, the vectors, and the payload (metadata)
    point = PointStruct(
        id=int(payload['pmid']),  # Unique ID for the article
        vector=vectors,           # Insert vectors here for similarity search
        payload=metadata_payload  # Insert the metadata (payload) separately
    )

    # Upsert the point (vector + metadata) into the Qdrant collection
    response = client.upsert(collection_name=collection_name, points=[point])
    logging.info(f"Inserted payload for PMID {payload['pmid']} into collection {collection_name}")
    return response


def process_and_upload(file_name, compressed_data, collection_name, last_processed_pmid):
    with gzip.GzipFile(fileobj=compressed_data, mode='rb') as f_in:
        extracted_data = f_in.read()

    articles_data = parse_pubmed_articles(extracted_data, max_articles=2)
    if articles_data:
        for article_data in articles_data:
            pmid = int(article_data['PMID'])
            if last_processed_pmid and pmid <= last_processed_pmid:
                logging.info(f"Skipping PMID {pmid} as it has already been processed")
                continue

            payload = generate_payload(article_data)
            response = insert_payload(qdrant_client, payload, collection_name=collection_name)
            print(response)

            # Update the last processed PMID
            write_last_pmid(pmid)
            logging.info(f"Processed and uploaded article with PMID {pmid}")
    logging.info(f"Processed and uploaded articles from {file_name}")
    return True


def main():
    ftp = ftplib.FTP(ftp_server)
    ftp.login()
    ftp.cwd(ftp_directory)

    collection_name = "PubMed"
    ensure_collection_exists(qdrant_client, collection_name)

    # Read the last processed PMID, if available
    last_processed_pmid = read_last_pmid()

    for i in range(1, 2):  # Adjust the range up to 1220 for the full dataset.
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
            logging.info(f"Checksums matched. Processing file {file_name}...")
            process_and_upload(file_name, compressed_data, collection_name, last_processed_pmid)
        else:
            logging.error(f"MD5 mismatch for file {file_name}. Expected: {expected_md5}, Calculated: {calculated_md5}")

    ftp.quit()


if __name__ == "__main__":
    main()
