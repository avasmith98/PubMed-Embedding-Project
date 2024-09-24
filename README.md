
To start Qdrant, navigate to the Qdrant directory and type ./target/release/qdrant. Then, run Qdrant_Updated. The program:

1. Reads PubMed baseline data file(s) from the NCBI FTP server into memory. Files go from pubmed24n0001 to pubmed24n1219. The number of files processed at once can be adjusted. The default is to only process pubmed24n0001.
2. Compares MD5 checksum of selected file with the provided checksum from Pubmed. Prints to the console if successful or not.
3. Parses XML file(s) to extract metadata including PMID, abstract, authors, journal details, keywords, and more. It skips articles that have been retracted or do not have an abstract. The number of articles per file can be adjusted. The default is 2.
4. Generates embeddings for article abstracts using bge-m3 (Ollama), and bge-large (Ollama).
5. Creates (if necessary) a collection and stores the processed data and embeddings in the collection/Qdrant vector database. 
6. If the connection is interrupted at any point, the program can be restarted and will know which PMID it left off on.

Next, run the User Interaction file. This program:
1. Takes a user's question and chosen embedding model.
2. Convers the user's question to an embedding (uses the chosen embedding model, which will also be the model used in the next step).
3. Accesses the Qdrant database and returns top abstracts. 
