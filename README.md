(AI Generated ReadMe)

PubMed Article Embedding and Storage Pipeline
This project is designed to download, process, and store PubMed article metadata into a Qdrant vector database. The key functionality includes fetching PubMed data from an FTP server, extracting relevant metadata, generating embeddings using OpenAI, and storing the processed data in Qdrant for efficient similarity search and retrieval.

Table of Contents
Features
Requirements
Installation
Usage
Code Overview
License
Features
FTP Integration: Downloads PubMed baseline data files from the NCBI FTP server.
MD5 Checksum Verification: Ensures data integrity by comparing MD5 checksums of downloaded files.
Data Extraction: Parses XML files to extract key metadata including PMID, abstract, authors, journal details, keywords, and more.
Text Embedding: Generates embeddings for article abstracts using OpenAI's text-embedding-3-small model.
Vector Database Integration: Stores the processed data and embeddings in a Qdrant vector database for efficient retrieval.
Collection Management: Ensures the Qdrant collection exists before inserting data.
Requirements
Python 3.8+
OpenAI Python SDK for embedding generation
Qdrant Client for interacting with the Qdrant vector database
ftplib for FTP operations
hashlib for MD5 checksum calculation
gzip for decompressing the data files
xml.etree.ElementTree for parsing XML data
