import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage import StorageContext
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Define persistent storage directory
PERSIST_DIR = "./embeddings"
DATA_DIR = "./docs"
INDEX_FILE = os.path.join(PERSIST_DIR, "docstore.json")  # File expected for index

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_or_load_index():
    # Check if index file exists in PERSIST_DIR
    if not os.path.exists(PERSIST_DIR) or not os.path.exists(INDEX_FILE):
        print("Index file not found. Creating new index...")
        
        # Ensure DATA_DIR exists
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            print(f"Created {DATA_DIR} for storing documents.")
        
        # Check for files in DATA_DIR
        try:
            documents = SimpleDirectoryReader(DATA_DIR).load_data()
        except ValueError as e:
            # If no files found, create an empty index
            print(f"No documents found in {DATA_DIR}. Creating an empty index.")
            documents = []
        
        # Create index from documents
        if documents:
            index = VectorStoreIndex.from_documents(documents, show_progress=True, embed_model=embed_model)
            print("Index created from documents.")
        else:
            index = VectorStoreIndex(embed_model=embed_model)  # Create an empty index
            print("Empty index created.")

        # Create PERSIST_DIR if not exists
        if not os.path.exists(PERSIST_DIR):
            os.makedirs(PERSIST_DIR)

        # Persist the index to PERSIST_DIR
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"Index persisted to {PERSIST_DIR}")
        
    else:
        print(f"Loading existing index from {PERSIST_DIR}...")
        # Load the index from PERSIST_DIR if it exists
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        # Pass the embed_model explicitly when loading the index
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        print("Index loaded successfully.")

    return index

def main():
    index = create_or_load_index()
    print("Index is ready for use.")

if __name__ == "__main__":
    main()
