import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def build_vector_store():
    docs_dir = "docs"
    persist_dir = "chroma_db"

    if not os.path.exists(docs_dir):
        print(f"Error: Could not find the '{docs_dir}' folder.")
        return

    # SAFETY CHECK: Remove the old database so we don't create duplicate entries
    if os.path.exists(persist_dir):
        print(f"Clearing old '{persist_dir}' to prevent duplicates...")
        shutil.rmtree(persist_dir)

    print(f"1. Loading all .txt files from '{docs_dir}'...")
    # This reads EVERY .txt file in the folder
    loader = DirectoryLoader(
        docs_dir, 
        glob="*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()

    if not documents:
        print(f"No .txt files found in the '{docs_dir}' folder.")
        return

    print(f"Loaded {len(documents)} document(s).")

    print("2. Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print("3. Creating embeddings and saving to ChromaDB...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    
    print(f"Success! Created {len(chunks)} chunks from your text files and saved them to '{persist_dir}'.")

if __name__ == "__main__":
    db_file = "memoria_chat.db"
    cache_dir = "app/__pycache__"

    # 1. Clear the old SQLite database (File)
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Cleared old chat history: {db_file}")

    # 2. Clear the compiled Python cache (Folder)
    if os.path.exists(cache_dir):
        import shutil # Just in case it's not imported at the top
        shutil.rmtree(cache_dir)
        print(f"Cleared old Python cache: {cache_dir}")

    # 3. Build the fresh Vector Store
    build_vector_store()