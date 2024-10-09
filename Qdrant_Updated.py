import logging
import hashlib
import ftplib
import gzip
import time  
from io import BytesIO
import ollama
import xml.etree.ElementTree as ET
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import json
import os

# Set up logging to file and console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("my_pubmed_processing.log"),
                        logging.StreamHandler()
                    ])

# FTP server details
ftp_server = "ftp.ncbi.nlm.nih.gov"
ftp_directory = "/pubmed/baseline/"
file_pattern = "pubmed24n{:04d}.xml.gz"
md5_file_pattern = "pubmed24n{:04d}.xml.gz.md5"

# Qdrant client setup
qdrant_client = QdrantClient(host='localhost', port=6333)

# JSON file to track processed PMIDs
PROCESSED_PMIDS_FILE = "my_processed_pmids.json"

def load_processed_pmids():
    """Load processed PMIDs from a JSON file."""
    if os.path.exists(PROCESSED_PMIDS_FILE):
        with open(PROCESSED_PMIDS_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_pmid(pmid):
    """Save a processed PMID to the JSON file."""
    processed_pmids = load_processed_pmids()
    processed_pmids.add(pmid)
    with open(PROCESSED_PMIDS_FILE, 'w') as f:
        json.dump(list(processed_pmids), f)

def generate_bgem3_embedding(text, model='bge-m3'):
    response = ollama.embeddings(model=model, prompt=text)
    return response['embedding']

def generate_bge_large_embedding(text, model='bge-large'):
    response = ollama.embeddings(model=model, prompt=text)
    return response['embedding']

def ensure_collection_exists(client, collection_name):
    if not client.collection_exists(collection_name):
        logging.info(f"Collection {collection_name} does not exist. Creating collection...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "bgem3_embedding": VectorParams(size=1024, distance=Distance.COSINE),
                "bge_large_embedding": VectorParams(size=1024, distance=Distance.COSINE)
            }
        )
    else:
        logging.info(f"Collection {collection_name} already exists.")

def parse_pubmed_articles(data, collection_name):
    root = ET.fromstring(data)
    processed_pmids = load_processed_pmids()

    for medline_citation in root.findall(".//MedlineCitation"):
        article_data = {}
        
        # Extract PMID and its version
        pmid_element = medline_citation.find("PMID")
        if pmid_element is None:
            continue  # Skip articles without a PMID
        pmid = pmid_element.text
        
        # Skip if this PMID has already been processed
        if pmid in processed_pmids:
            logging.info(f"Skipped article with PMID: {pmid}, Reason: Already processed")
            continue

        pmid_version = pmid_element.attrib.get('Version', '')

        # Check for retracted articles
        comments_corrections = medline_citation.findall(".//CommentsCorrections")
        is_retracted = any(comment.attrib.get('RefType', '') in ["Retraction of", "Retraction in"] for comment in comments_corrections)
        if is_retracted:
            logging.info(f"Skipped article with PMID: {pmid}, Reason: Retracted article")
            continue

        # Extract Abstract
        abstract_elements = medline_citation.findall(".//Abstract/AbstractText")
        abstract_texts = [abstract.text for abstract in abstract_elements if abstract.text]
        if not abstract_texts:
            logging.info(f"Skipped article with PMID: {pmid}, Reason: No abstract")
            continue  # Skip articles without an abstract
        article_data['Abstract'] = ' '.join(abstract_texts)

        # Extract Article Title
        article_title_element = medline_citation.find(".//ArticleTitle")
        article_data['Title'] = article_title_element.text if article_title_element is not None else ''

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
            article_data['Journal'] = {'Title': '', 'Volume': '', 'PubDate': {'Year': '', 'Month': '', 'Day': ''}}

        # Extract Authors
        author_list_element = medline_citation.find(".//AuthorList")
        if author_list_element is not None:
            article_data['Authors'] = [{
                'LastName': author.find("LastName").text if author.find("LastName") is not None else '',
                'ForeName': author.find("ForeName").text if author.find("ForeName") is not None else '',
            } for author in author_list_element.findall("Author")]
        else:
            article_data['Authors'] = []

        # Extract Keywords
        keyword_elements = medline_citation.findall(".//KeywordList/Keyword")
        article_data['Keywords'] = [keyword.text for keyword in keyword_elements if keyword is not None]

        article_data['PMID'] = pmid
        article_data['PMID_Version'] = pmid_version

        # Generate payload and upsert to Qdrant
        payload = generate_payload(article_data)
        response = upsert(article_data, qdrant_client, payload, collection_name)
        if response:
            logging.info(f"Uploaded article with PMID: {payload['pmid']}")
            # Save processed PMID
            save_processed_pmid(pmid)

def generate_payload(article_data):
    payload = {
        "pmid": article_data.get('PMID', 'Unknown'),
        "pmid_version": article_data.get('PMID_Version', 'Unknown'),
        "title": article_data.get('Title', ''),
        "abstract": article_data.get('Abstract', ''),
        "authors": article_data.get('Authors', []),
        "journal": article_data.get('Journal', {}),
        "keywords": article_data.get('Keywords', []),
    }
    return payload

def upsert(article_data, client, payload, collection_name):
    bgem3_embedding = generate_bgem3_embedding(article_data['Abstract'])
    bge_large_embedding = generate_bge_large_embedding(article_data['Abstract'])

    point = PointStruct(id=int(payload['pmid']), vector={"bgem3_embedding": bgem3_embedding, "bge_large_embedding": bge_large_embedding}, payload=payload)
    response = client.upsert(collection_name=collection_name, points=[point])
    return response

def process_and_upload(file_name, compressed_data, collection_name):
    logging.info(f"Starting to process file: {file_name}")

    with gzip.GzipFile(fileobj=compressed_data, mode='rb') as f_in:
        extracted_data = f_in.read()

    logging.info(f"Parsing and processing articles from {file_name}")
    parse_pubmed_articles(extracted_data, collection_name)  # Pass collection name to handle articles immediately

    logging.info(f"Finished processing {file_name}")
    return True

def main():
    ftp = ftplib.FTP(ftp_server)
    logging.info("FTP connection successful")
    ftp.login()
    ftp.cwd(ftp_directory)
    logging.info(f"Changed directory to {ftp_directory}")

    collection_name = "PubMed_2024"
    ensure_collection_exists(qdrant_client, collection_name)

    for i in range(1, 2):  # Adjust the range as needed
        file_name = file_pattern.format(i)
        md5_file_name = md5_file_pattern.format(i)

        logging.info(f"Processing file {file_name}")
        
        # Add a delay between each FTP call
        time.sleep(1)
        
        # Retrieve and check MD5
        md5_data = BytesIO()
        logging.info(f"Retrieving MD5 for {file_name}")
        ftp.retrbinary(f"RETR {md5_file_name}", md5_data.write)
        md5_contents = md5_data.getvalue().decode().strip()
        expected_md5 = md5_contents.split('=')[1].strip() if '=' in md5_contents else md5_contents.split()[0].strip()

        # Add a delay between each FTP call
        time.sleep(1)
        
        # Retrieve the compressed file
        compressed_data = BytesIO()
        logging.info(f"Retrieving {file_name}")
        ftp.retrbinary(f"RETR {file_name}", compressed_data.write)

        # Calculate MD5 for the compressed data
        calculated_md5 = hashlib.md5(compressed_data.getvalue()).hexdigest()
        logging.info(f"MD5 for {file_name}: expected {expected_md5}, calculated {calculated_md5}")

        if calculated_md5 == expected_md5:
            compressed_data.seek(0)  # Reset buffer position
            logging.info(f"Checksums matched for {file_name}. Processing file...")
            process_and_upload(file_name, compressed_data, collection_name)
        else:
            logging.info(f"MD5 mismatch for {file_name}. Expected: {expected_md5}, Calculated: {calculated_md5}")

    ftp.quit()
    logging.info("FTP connection closed")

if __name__ == "__main__":
    main()
