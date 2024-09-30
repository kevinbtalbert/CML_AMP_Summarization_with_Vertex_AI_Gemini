import os
import textwrap
import streamlit as st
import PyPDF2
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage import StorageContext
from llama_index.core.indices.loading import load_index_from_storage
import google.generativeai as genai
from google.generativeai import GenerationConfig
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core import Document

# Define persistent storage directory
PERSIST_DIR = "./embeddings"
DATA_DIR = "./docs"
INDEX_FILE = os.path.join(PERSIST_DIR, "docstore.json")  # File expected for index

# Set up Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Set up HuggingFace Embedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Update global settings
Settings.embed_model = embed_model

# Disable default LLM in llama_index to avoid OpenAI usage
Settings.llm = None

# List of allowed models
ALLOWED_MODELS = [
    "models/gemini-1.0-pro-latest",
    "models/gemini-1.0-pro",
    "models/gemini-pro",
    "models/gemini-1.0-pro-001",
    "models/gemini-1.5-pro-001",
    "models/gemini-1.5-pro-latest"
]


def get_metadata(file_path):
    """Function to attach metadata (e.g., filename) to each document."""
    return {"filename": os.path.basename(file_path)}

def get_content_generation_models():
    """Fetch suitable models for content generation."""
    models = [
        m.name for m in genai.list_models()
        if 'generateContent' in m.supported_generation_methods and m.name in ALLOWED_MODELS
    ]
    if not models:
        raise Exception("No suitable content generation models found.")
    return models

def create_or_load_index():
    """Create a new index or load the existing one."""
    # Check if index exists
    if os.path.exists(PERSIST_DIR) and os.path.exists(INDEX_FILE):
        # Load existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        # Initialize an empty index if no index exists
        st.info("No existing index found. Initializing an empty index...")
        index = None

    return index

def rebuild_index():
    """Rebuild the index from the current contents of the ./docs directory."""
    if not os.path.exists(DATA_DIR):
        st.error("Data directory does not exist.")
        return

    # Initialize an empty list to hold all documents
    documents = []

    # Loop through each PDF file in the directory
    for pdf_file in os.listdir(DATA_DIR):
        if pdf_file.endswith(".pdf"):
            # Construct full path to the file
            file_path = os.path.join(DATA_DIR, pdf_file)
            
            # # Read content of the file using PyPDF2
            # with open(file_path, "rb") as file:
            #     pdf_reader = PyPDF2.PdfReader(file)
            #     content = ""
            #     for page_num in range(len(pdf_reader.pages)):
            #         page = pdf_reader.pages[page_num]
            #         text = page.extract_text()
            #         if text:
            #             content += text
            
            # Check if content is valid

            # Create a Document object
            # doc = Document(content=content, metadata={"filename": pdf_file})
            # Load the new document for indexing
            reader = SimpleDirectoryReader(input_files=[file_path], file_metadata=get_metadata)
            doc = reader.load_data()
            documents.append(doc)

        else:
            st.warning(f"No content extracted from '{pdf_file}'. Skipping.")

    if not documents:
        st.error("No valid documents found for indexing.")
        return

    # Create a new index with the collected documents
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    
    # Persist the new index to storage
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    st.success("Rebuilt the vector store index successfully.")

def add_document_to_index(uploaded_file):
    """Add a new document to the docs folder and add it to the index."""
    # Save the uploaded file to the ./docs directory
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"PDF uploaded and saved as {file_path}")

    # Load the new document for indexing
    reader = SimpleDirectoryReader(input_files=[file_path], file_metadata=get_metadata)
    new_documents = reader.load_data()
    
    # Debugging: print document contents and metadata
    for doc in new_documents:
        print(f"[DEBUG] Loaded new document '{doc.metadata['filename']}' with content length: {len(doc.text)} characters")
    
    # Load the existing index or create a new one
    index = create_or_load_index()
    
    # If an index already exists, add the new documents
    index = VectorStoreIndex.from_documents(new_documents, embed_model=embed_model)
    
    # Persist the index with the new document added
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    st.success("Document added to the vector store.")


def delete_document_from_index(filename):
    """Delete a document from the vector store and ./docs directory."""
    # Delete the document from the ./docs folder
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        st.success(f"Deleted {filename} from file system.")

    # Rebuild the index after deletion
    rebuild_index()

def main():
    st.title("RAG with Gemini")

    # Only load index if it exists
    index = create_or_load_index()
    if index:
        st.write(f"Index loaded. Number of nodes: {len(index.docstore.docs)}")

        # Fetch models for dropdown selection
        content_models = get_content_generation_models()

    # Reordered tabs: 1. Summarize from Text Input, 2. Summarize from Doc Library, 3. Manage Vector Store
    tab1, tab2, tab3 = st.tabs(["Summarize from Text Input", "Summarize from Doc Library", "Manage Vector Store"])

    # Tab for summarizing from text input
    with tab1:
        st.header("Summarize from Text Input")

        # Select model for content generation
        selected_model_name = st.selectbox("Select a model for content generation:", content_models, key="text_model")

        # Text input area for user to provide content
        user_query = st.text_area("Enter text to summarize:")

        # Add sliders for generation configuration
        max_output_tokens = st.slider("Max Output Tokens", min_value=128, max_value=2056, value=256, key="text_max_tokens")
        temperature = st.slider("Temperature", min_value=0.01, max_value=1.0, value=0.2, key="text_temperature")

        # Add a button to generate the summary from text input
        if st.button("Generate Summary from Text"):
            if user_query.strip():
                query = "Summarize the following text:"
                prompt = textwrap.dedent(f"""
                You are a helpful and informative bot that answers questions using text from the reference passage included below.
                QUESTION: '{query}'
                PASSAGE: '{user_query.strip()}'
                ANSWER:
                """)

                try:
                    # Initialize the selected model
                    model = genai.GenerativeModel(selected_model_name)

                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=0.95,
                        top_k=20,
                        candidate_count=1,
                        max_output_tokens=max_output_tokens,
                        stop_sequences=["STOP!"],
                    )

                    # Generate content
                    answer = model.generate_content(prompt)
                    answer_text = "".join([response.text for response in answer])

                    # Display the AI-generated answer
                    st.write("Summary:", answer_text)

                except Exception as e:
                    st.error(f"Failed to generate summary: {e}")
            else:
                st.warning("Please enter some text to summarize.")

    # Tab for summarizing from the vector store
    with tab2:
        st.header("Summarize from Doc Library")

        if index:
            # Select model for content generation
            selected_model_name = st.selectbox("Select a model for content generation:", content_models, key="doc_model")

            # Dropdown for selecting a document from the index
            filenames = list(set([doc.metadata.get("filename") for doc in index.docstore.docs.values()]))

            # Use filenames in the dropdown for selection
            selected_filename = st.selectbox("Select a document for summarization:", filenames)
            
            # Add sliders for generation configuration
            max_output_tokens = st.slider("Max Output Tokens", min_value=128, max_value=2056, value=256, key="doc_max_tokens")
            temperature = st.slider("Temperature", min_value=0.01, max_value=1.0, value=0.2, key="doc_temperature")

            # Add a button to generate the summary
            if st.button("Generate Summary from Doc"):
                # Retrieve relevant content chunks for the selected document
                relevant_docs = [doc.get_content() for doc in index.docstore.docs.values() if doc.metadata.get("filename") == selected_filename]
                relevant_passage = " ".join(relevant_docs)
                query = "Summarize the content of this document."

                # Create a prompt for the Generative Model
                prompt = textwrap.dedent(f"""
                You are a helpful and informative bot that answers questions using text from the reference passage included below.
                QUESTION: '{query}'
                PASSAGE: '{relevant_passage}'
                ANSWER:
                """)

                try:
                    # Initialize the selected model
                    model = genai.GenerativeModel(selected_model_name)

                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=0.95,
                        top_k=20,
                        candidate_count=1,
                        max_output_tokens=max_output_tokens,
                        stop_sequences=["STOP!"],
                    )

                    # Generate content
                    answer = model.generate_content(prompt)
                    answer_text = "".join([response.text for response in answer])

                    # Display the AI-generated answer
                    st.write("Summary:", answer_text)

                except Exception as e:
                    st.error(f"Failed to generate summary: {e}")
        else:
            st.info("No documents found in the vector store for summarization.")

    # Tab for managing the vector store (uploading & deleting PDFs)
    with tab3:
        st.header("Manage Vector Store")

        # File uploader for PDFs
        uploaded_file = st.file_uploader("Choose a PDF file to add to the vector store:", type="pdf")
        if uploaded_file is not None:
            # Add the document directly to the index
            if st.button("Add Document"):
                add_document_to_index(uploaded_file)
                st.rerun()  # Refresh the page to reflect changes

        # Display existing PDFs in vector store
        st.subheader("Existing PDFs in Vector Store")
        existing_pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')] if os.path.exists(DATA_DIR) else []

        if existing_pdfs:
            for pdf in existing_pdfs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(pdf)
                with col2:
                    delete_button = st.button("Delete", key=f"delete_{pdf}")
                    if delete_button:
                        # Ensure there is more than one document before allowing deletion
                        if len(existing_pdfs) > 1:
                            delete_document_from_index(pdf)
                            st.rerun()  # Refresh the page after deletion
                        else:
                            st.warning("Cannot delete the last document in the vector store.")
        else:
            st.info("No PDFs currently in the vector store.")

if __name__ == "__main__":
    main()
