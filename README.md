
1. Reads PubMed baseline data file(s) from the NCBI FTP server into memory. Files go from pubmed24n0001 to pubmed24n1219. The number of files processed at once can be adjusted. The default is to only process pubmed24n0001.

2. Compares MD5 checksum of selected file with the provided checksum from Pubmed. Prints to the console if successful or not.

3. Parses XML file(s) to extract metadata including PMID, abstract, authors, journal details, keywords, and more. Skips articles that have been retracted or do not have an abstract. The number of articles per file can be adjusted. The default is 10.

4. Generates embeddings for article abstracts using OpenAI's text-embedding-3-small model.

5. Creates (if necessary) and stores the processed data and embeddings in a Qdrant vector database. (Data resides in the docker container...maybe find a different way to store it)
