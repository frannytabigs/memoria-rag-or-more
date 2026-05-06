import os
import shutil
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def build_core_vector_store():
    """Processes the official cemetery documents."""
    docs_dir = "docs"
    persist_dir = "chroma_db"

    if not os.path.exists(docs_dir):
        print(f"Error: Could not find the '{docs_dir}' folder.")
        return

    # SAFETY CHECK: Remove the old document database so we don't create duplicate entries
    if os.path.exists(persist_dir):
        print(f"Clearing old '{persist_dir}' to prevent duplicates...")
        shutil.rmtree(persist_dir)

    print(f"\n--- INGESTING OFFICIAL DOCUMENTS ---")
    print(f"1. Loading all .txt files from '{docs_dir}'...")
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
    
    print(f"Success! Created {len(chunks)} document chunks.")


def vectorize_all_user_chats():
    """Processes all logged-in user chats and saves them to a separate collection."""
    db_file = "memoria_chat.db"
    
    if not os.path.exists(db_file):
        print(f"\n--- INGESTING CHATS ---")
        print(f"No '{db_file}' found. Skipping chat vectorization.")
        return

    print(f"\n--- INGESTING CHATS ---")
    print("1. Connecting to SQLite and fetching users...")
    sqlite_conn = sqlite3.connect(db_file)
    cursor = sqlite_conn.cursor()

    try:
        # Get all distinct phone numbers that are NOT NULL
        cursor.execute("SELECT DISTINCT phone_number FROM chat_logs WHERE phone_number IS NOT NULL")
        phone_numbers = [row[0] for row in cursor.fetchall()]

        if not phone_numbers:
            print("No logged-in users with phone numbers found in chat logs. Skipping.")
            return

        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Ensure Chroma uses the exact same embedding model as Langchain
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        # Create or get the specific chat collection
        chat_collection = chroma_client.get_or_create_collection(
            name="chat_memory", 
            embedding_function=emb_fn
        )

        print(f"2. Vectorizing chat history for {len(phone_numbers)} user(s)...")
        for phone in phone_numbers:
            # Pull the user's chat history
            cursor.execute("SELECT user_message, bot_response FROM chat_logs WHERE phone_number = ?", (phone,))
            rows = cursor.fetchall()
            
            if rows:
                # Combine the messages into a single transcript chunk
                chat_transcript = "\n".join([f"User: {user_msg}\nBot: {bot_msg}" for user_msg, bot_msg in rows])
                
                # Use UPSERT so if you run ingest.py twice, it overwrites the old memory instead of duplicating it
                chat_collection.upsert(
                    documents=[chat_transcript],
                    metadatas=[{"phone_number": phone}], 
                    ids=[f"chat_summary_{phone}"] 
                )
                
        print("Success! User chats have been vectorized and stored in 'chat_memory'.")

    except sqlite3.OperationalError as e:
        print(f"Database error (Does the table exist yet?): {e}")
    finally:
        sqlite_conn.close()


if __name__ == "__main__":
    cache_dir = "app/__pycache__"

    # 1. Clear the compiled Python cache
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print(f"Cleared old Python cache: {cache_dir}")

    # 2. Build the Core Facts Vector Store (memoria.txt, etc.)
    build_core_vector_store()
    
    # 3. Build the User Chat Vector Store
    vectorize_all_user_chats()
    
    print("\n--- INGESTION COMPLETE ---")