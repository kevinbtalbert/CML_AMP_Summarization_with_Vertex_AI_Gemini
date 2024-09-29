import os
import sqlite3
import textwrap
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage import StorageContext
from llama_index.core.indices.loading import load_index_from_storage
import google.generativeai as genai
from google.generativeai import GenerationConfig
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

# Define persistent storage directory and SQLite DB path
PERSIST_DIR = "./embeddings"
DATA_DIR = "./docs"
DB_PATH = "./doc_metadata.db"

# Set up Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Set up HuggingFace Embedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Update global settings
Settings.embed_model = embed_model

# List of allowed models
ALLOWED_MODELS = [
    "models/gemini-1.0-pro-latest",
    "models/gemini-1.0-pro",
    "models/gemini-pro",
    "models/gemini-1.0-pro-001",
    "models/gemini-1.5-pro-001",
    "models/gemini-1.5-pro-latest"
]

def get_content_generation_models():
    models = [
        m.name for m in genai.list_models()
        if 'generateContent' in m.supported_generation_methods and m.name in ALLOWED_MODELS
    ]
    if not models:
        raise Exception("No suitable content generation models found.")
    return models

def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent(f"""\
    You are a helpful and informative bot that answers questions using text from the reference passage included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and conversational tone. \
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGE: '{escaped}'

    ANSWER:
    """)
    return prompt

# Initialize the SQLite DB
def init_db():
    """Initialize the SQLite database for file name to document ID mapping."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS doc_metadata (
            doc_id TEXT PRIMARY KEY,
            file_name TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_metadata(doc_id, file_name):
    """Add file name and document ID mapping to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO doc_metadata (doc_id, file_name) VALUES (?, ?)', (doc_id, file_name))
    conn.commit()
    conn.close()

def get_metadata():
    """Retrieve all metadata mappings from the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT doc_id, file_name FROM doc_metadata')
    rows = cursor.fetchall()
    conn.close()
    return {doc_id: file_name for doc_id, file_name in rows}

def delete_metadata(doc_id):
    """Delete metadata mapping from the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM doc_metadata WHERE doc_id = ?', (doc_id,))
    conn.commit()
    conn.close()

def load_or_create_index():
    """Load existing index or create one if not present."""
    init_db()  # Initialize the database for metadata
    
    if not os.path.exists(PERSIST_DIR):
        st.error(f"Persistent directory '{PERSIST_DIR}' not found.")
        return None
    
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

    return index

def list_existing_pdfs():
    """List existing PDF files in the DATA_DIR."""
    return [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]

def delete_pdf(pdf_name):
    """Delete the selected PDF from the DATA_DIR."""
    file_path = os.path.join(DATA_DIR, pdf_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        st.success(f"Deleted {pdf_name}")

def repopulate_index():
    """Repopulate the vector store index based on current PDFs in DATA_DIR."""
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True, embed_model=embed_model)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    
    # Update metadata in SQLite
    for doc in documents:
        doc_id = doc.get_id()
        file_name = doc.get_content_metadata().get('file_name', 'Unknown')
        add_metadata(doc_id, file_name)
    
    st.success("Vector store index has been updated.")

def main():
    st.title("RAG with Gemini")

    # Load index (if any exists)
    index = load_or_create_index()
    if index is None:
        st.error("Failed to create or load index. Please check your 'docs' directory.")
        return

    st.write(f"Index loaded. Number of nodes: {len(index.docstore.docs)}")

    # Retrieve metadata for document names
    doc_metadata = get_metadata()

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
                prompt = make_prompt(query, user_query.strip())

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

        # Select model for content generation
        selected_model_name = st.selectbox("Select a model for content generation:", content_models, key="doc_model")

        # Dropdown for selecting a document for summarization
        if index is not None and doc_metadata:
            # Use file names in the dropdown
            file_names = list(doc_metadata.values())
            selected_file_name = st.selectbox("Select a document for summarization:", file_names)

            # Map file name back to document ID
            selected_doc_id = next((doc_id for doc_id, file_name in doc_metadata.items() if file_name == selected_file_name), None)

            # Add sliders for generation configuration
            max_output_tokens = st.slider("Max Output Tokens", min_value=128, max_value=2056, value=256, key="doc_max_tokens")
            temperature = st.slider("Temperature", min_value=0.01, max_value=1.0, value=0.2, key="doc_temperature")

            # Add a button to generate the summary
            if st.button("Generate Summary from Doc"):
                # Retrieve content from the selected document
                relevant_passage = index.docstore.docs[selected_doc_id].get_content()
                query = "Summarize the content of this document."

                # Create a prompt for the Generative Model
                prompt = make_prompt(query, relevant_passage)

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
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"PDF uploaded and saved as {file_path}")

        # Display existing PDFs
        st.subheader("Existing PDFs in Vector Store")
        existing_pdfs = list_existing_pdfs()
        if existing_pdfs:
            for pdf in existing_pdfs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(pdf)
                with col2:
                    delete_button = st.button("Delete", key=f"delete_{pdf}")
                    if delete_button:
                        # Delete file from disk and associated metadata
                        delete_pdf(pdf)

                        # Find and delete the corresponding document ID in the metadata
                        for doc_id, file_name in doc_metadata.items():
                            if file_name == pdf:
                                delete_metadata(doc_id)

                        st.experimental_rerun()  # Refresh the page after deletion

            # Button to repopulate the vector store index after any changes
            if st.button("Repopulate Vector Store Index"):
                repopulate_index()
        else:
            st.info("No PDFs currently in the vector store.")

if __name__ == "__main__":
    main()
