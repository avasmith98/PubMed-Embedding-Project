import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import NamedVector
import ollama
import logging

# Qdrant client setup
qdrant_client = QdrantClient(host='localhost', port=6333)
collection_name = "PubMed"

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("embedding_search.log"), logging.StreamHandler()])

# Function to map model choice to vector name
def get_vector_name_for_model(model_choice):
    if model_choice == 'bge-m3':
        return 'bgem3_embedding'
    elif model_choice == 'bge-large':
        return 'bge_large_embedding'
    else:
        raise ValueError("Invalid model choice")

# User prompt to choose between models
def user_choose_model():
    """Prompt user to choose between bge-m3 or bge-large for embedding generation."""
    print("1. bge-m3 (default)")
    print("2. bge-large")
    
    choice = input("Enter the number of your choice (1 or 2): ").strip()

    if choice == '2':
        return 'bge-large'
    else:
        return 'bge-m3'  # Default to bge-m3 if user chooses 1 or any other input

# Generate embeddings for the user's input based on chosen model
def generate_bge_embedding(user_input, model='bge-m3'):
    """Generate text embeddings using the chosen bge model."""
    logging.info(f"Generating embedding for input: {user_input} using model: {model}")
    response = ollama.embeddings(model=model, prompt=user_input)
    return np.array(response['embedding'])

# Search Qdrant for top N results using cosine similarity
def search_qdrant_similar_abstracts(user_embedding, model_choice, top_n=5):
    """Search the Qdrant database for top N similar abstracts based on cosine similarity."""
    logging.info(f"Searching Qdrant using embedding from model: {model_choice}")

    vector_name = get_vector_name_for_model(model_choice)  # Dynamically select vector name

    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=NamedVector(
            name=vector_name,  # Specify the vector field to use for the search
            vector=user_embedding.tolist()
        ),
        limit=top_n,
        with_payload=True
    )
    return search_results

# Find top N similar abstracts based on user input and chosen model
def find_similar_abstracts(user_input, model_choice, top_n=5):
    """Find and return top N similar abstracts based on user input and chosen model."""
    
    # Step 1: Generate embedding for the user input using the selected model
    user_embedding = generate_bge_embedding(user_input, model=model_choice)

    # Step 2: Compare the embedding with the stored abstracts using cosine similarity
    search_results = search_qdrant_similar_abstracts(user_embedding, model_choice=model_choice, top_n=top_n)

    # Step 3: Return the top results (abstracts)
    logging.info(f"Top {top_n} similar abstracts found.")
    top_abstracts = []
    for result in search_results:
        top_abstracts.append({
            "pmid": result.payload['pmid'],
            "title": result.payload['title'],
            "abstract": result.payload['abstract'],
            "similarity": result.score
        })

    return top_abstracts

if __name__ == "__main__":
    # Step 1: Get the user's query
    user_input = input("Enter your query: ")

    # Step 2: Let the user choose the embedding model
    chosen_model = user_choose_model()
    print(f"You chose: {chosen_model}")

    # Step 3: Find top 5 similar abstracts based on user's input and chosen model
    top_abstracts = find_similar_abstracts(user_input, model_choice=chosen_model, top_n=5)

    # Step 4: Print the top results
    for i, abstract in enumerate(top_abstracts):
        print(f"\nTop {i+1} Abstract:")
        print(f"PMID: {abstract['pmid']}")
        print(f"Title: {abstract['title']}")
        print(f"Abstract: {abstract['abstract']}")
        print(f"Similarity Score: {abstract['similarity']}")
