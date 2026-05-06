import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ==========================================
# 1. INITIALIZE MODELS & GLOBAL RULES
# ==========================================
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

# Initialize ChromaDB client specifically for the chat memory
chroma_client = chromadb.PersistentClient(path="./chroma_db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chat_collection = chroma_client.get_or_create_collection(name="chat_memory", embedding_function=emb_fn)

def get_base_rules():
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    return f"""
    - TIME AWARENESS: The current date and time is {now}. Use this to calculate if a burial lease is expired, expiring soon, or active.
    - LANGUAGE: Your default language is STRICTLY English. You must reply in English unless the user's prompt is written entirely in Tagalog or Bisaya or any other local language or languages . Do not let local names or addresses change your language.
    - STRICT SCOPE: This is a government system strictly for HUMAN burial records. If the user asks about pets, politely inform them that Memoria does not accommodate animal burials.
    """

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_router_prompt():
    return """You are the routing brain for Memoria, the official chatbot for the Mandaue City Public Cemetery. Your job is to classify the user's input into exactly ONE of four categories. Do not answer the question. Output ONLY the category word. 

CATEGORIES: 
1. CEMETERY_INFO: General public information, requirements, operating hours, fees, process steps, government office locations, or contact details. 
2. CEMETERY_RECORDS: Specific deceased individuals, grave plot locations INSIDE the cemetery, availability of blocks/vacant graves, or personal burial records.
3. GENERAL_CHAT: Saying hello, small talk, or clear general knowledge questions entirely unrelated to the cemetery. 
4. CLARIFY: The user's input is ambiguous.

User Input: "{user_question}" 
Category:"""

def format_history(chat_history):
    messages = []
    for chat in chat_history:
        messages.append(HumanMessage(content=chat['user']))
        messages.append(AIMessage(content=chat['bot']))
    return messages

def retrieve_personal_memory(user_query, user_phone_number):
    if not user_phone_number or user_phone_number == "None":
        return "No past context available. User is anonymous."
    try:
        results = chat_collection.query(
            query_texts=[user_query],
            n_results=2,
            where={"phone_number": user_phone_number} 
        )
        if results['documents'] and len(results['documents'][0]) > 0:
            return "\n\n".join(results['documents'][0])
        else:
            return "No relevant past context found for this user."
    except Exception as e:
        print(f"DEBUG - Error fetching memory: {e}")
        return "No past context available."

# ==========================================
# 3. THE ROUTER
# ==========================================
def run_router(user_question):
    router_template = load_router_prompt()
    formatted_prompt = router_template.replace("{user_question}", user_question)
    messages = [SystemMessage(content=formatted_prompt)]
    result = model.invoke(messages)
    
    category = result.content.strip().upper()
    if category not in ["CEMETERY_INFO", "CEMETERY_RECORDS", "GENERAL_CHAT", "CLARIFY"]:
        return "GENERAL_CHAT"
    return category

# ==========================================
# 4. EXECUTION BRANCHES
# ==========================================
def run_rag_pipeline(user_question, formatted_history, past_memory):
    docs = db.similarity_search(user_question, k=3)
    results_from_core_facts = "\n\n".join([doc.page_content for doc in docs]) if docs else "No official facts found regarding this query."

    system_instruction = f"""
    You are Memoria, the official chatbot for the Mandaue City Public Cemetery.
    
    CRITICAL RULES:
    1. Answer using STRICTLY the context provided below.
    2. Do NOT invent premium services, tours, or monuments.
    {get_base_rules()}    
    
    PRIMARY FACTS (Absolute Truth - Never Contradict):
    {results_from_core_facts}
    
    PAST CONTEXT FROM PREVIOUS CONVERSATIONS:
    {past_memory}
    
    Rule: If the Past Context contradicts the Primary Facts, you MUST ignore the Past Context and state the Primary Fact.
    """
    messages = [SystemMessage(content=system_instruction)] + formatted_history + [HumanMessage(content=user_question)]
    return model.invoke(messages).content

def run_database_query(user_question, formatted_history, past_memory):
    json_path = os.path.join(os.path.dirname(__file__), '..', 'records.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            db_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return "I am currently unable to access the cemetery records system due to a connection error."

    json_context = json.dumps(db_data, indent=2)
    system_instruction = f"""
    You are Memoria's secure records assistant. 
    Use ONLY the following JSON records to answer the user's question. 
    {get_base_rules()}

    DATABASE RECORDS:
    {json_context}
    
    PAST CONTEXT FROM PREVIOUS CONVERSATIONS:
    {past_memory}
    """
    messages = [SystemMessage(content=system_instruction)] + formatted_history + [HumanMessage(content=user_question)]
    return model.invoke(messages).content

def run_general_chat(user_question, formatted_history, past_memory):
    system_instruction = f"""
    You are Memoria, the official chatbot for the Mandaue City Public Cemetery. 
    1. This is a local, public municipal cemetery. You DO NOT offer premium services.
    2. If the user makes small talk, politely greet them and remind them you can assist with official cemetery requests.
    {get_base_rules()}
    
    PAST CONTEXT FROM PREVIOUS CONVERSATIONS:
    {past_memory}
    """
    messages = [SystemMessage(content=system_instruction)] + formatted_history + [HumanMessage(content=user_question)]
    return model.invoke(messages).content

def run_clarification(user_question, formatted_history, past_memory):
    system_instruction = f"""
    You are Memoria. The user asked an ambiguous question.
    Politely ask them to clarify. Specifically, say something similar to: "Are you looking for a specific burial record in our cemetery? If so, please use the Login button above so I can securely search the database. Or are you just asking a general question?"
    {get_base_rules()}
    
    PAST CONTEXT FROM PREVIOUS CONVERSATIONS:
    {past_memory}
    """
    messages = [SystemMessage(content=system_instruction)] + formatted_history + [HumanMessage(content=user_question)]
    return model.invoke(messages).content

# ==========================================
# 5. MAIN ENTRY POINT
# ==========================================
def get_response(user_question, chat_history, is_logged_in=False, phone_number=None):
    formatted_history = format_history(chat_history)
    
    # 1. Fetch memory globally
    past_memory = retrieve_personal_memory(user_question, phone_number)
    category = run_router(user_question)
    
    print(f"DEBUG - Traffic Cop decided: {category}") 
    
    if category == "CEMETERY_INFO":
        return run_rag_pipeline(user_question, formatted_history, past_memory)
    
    elif category == "CEMETERY_RECORDS":
        if not is_logged_in:
             return "It sounds like you want me to search the official cemetery records. To protect families' privacy, you must be logged in to view specific burial data. Please click the **Login** button at the top! If you meant to ask a general public question, just let me know."
        return run_database_query(user_question, formatted_history, past_memory)
        
    elif category == "CLARIFY":
        return run_clarification(user_question, formatted_history, past_memory)
        
    else: 
        return run_general_chat(user_question, formatted_history, past_memory)