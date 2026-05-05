import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
key = os.getenv("GROQ_API_KEY")
print(f"DEBUG: API Key found: {key[:10] if key else 'None'}...")
# ==========================================
# 1. INITIALIZE MODELS & GLOBAL RULES
# ==========================================
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

def get_base_rules():
    """Generates the rules dynamically so the clock is always 100% accurate."""
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    
    return f"""
    - TIME AWARENESS: The current date and time is {now}. Use this to calculate if a burial lease is expired, expiring soon, or active, and to understand concepts like "today", "recently", or current time.
    - LANGUAGE MATCHING: You MUST respond in the exact same language the user is using.
    - STRICT SCOPE: This is a government system strictly for HUMAN burial records. If the user asks about pets (cats, dogs, etc.), politely inform them that Memoria: Mandaue City Public Cemetery does not accommodate or track animal burials.
    """

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_router_prompt():
    # FIXED: Added the closing quotes so it doesn't crash!
    return """You are the routing brain for Memoria, the official chatbot for the Mandaue City Public Cemetery. Your job is to classify the user's input into exactly ONE of four categories. Do not answer the question. Output ONLY the category word. 

CATEGORIES: 
1. CEMETERY_INFO: General public information, requirements, operating hours, fees, process steps, government office locations, or contact details. 
2. CEMETERY_RECORDS: Specific deceased individuals, grave plot locations INSIDE the cemetery, availability of blocks/vacant graves, or personal burial records. (Do NOT use this for government office locations). 
3. GENERAL_CHAT: Saying hello, small talk, or clear general knowledge questions entirely unrelated to the cemetery. 
4. CLARIFY: The user's input is ambiguous. It could refer to a general knowledge topic OR a specific cemetery record (e.g., "Where is my grandma?", "Who is Lapu-Lapu?", "Tell me about Rizal").

User Input: "{user_question}" 
Category:"""

def format_history(chat_history):
    messages = []
    for chat in chat_history:
        messages.append(HumanMessage(content=chat['user']))
        messages.append(AIMessage(content=chat['bot']))
    return messages

def save_to_global_memory(user_question, bot_response):
    chunk_text = f"[PAST USER Q&A]\nUser asked: {user_question}\nMemoria answered: {bot_response}"
    db.add_texts(texts=[chunk_text])
    print("DEBUG - Injected new knowledge into Global Memory!")


# ==========================================
# 3. THE ROUTER (TRAFFIC COP)
# ==========================================
def run_router(user_question):
    router_template = load_router_prompt()
    formatted_prompt = router_template.replace("{user_question}", user_question)
    
    messages = [SystemMessage(content=formatted_prompt)]
    result = model.invoke(messages)
    
    category = result.content.strip().upper()
    valid_categories = ["CEMETERY_INFO", "CEMETERY_RECORDS", "GENERAL_CHAT", "CLARIFY"]
    
    if category not in valid_categories:
        return "GENERAL_CHAT"
    return category


# ==========================================
# 4. THE EXECUTION BRANCHES
# ==========================================
def run_rag_pipeline(user_question, formatted_history):
    """Branch 1: Public Info (ChromaDB)"""
    docs = db.similarity_search(user_question, k=3)
    
    if not docs:
        return "I am Memoria. I don't see that in my official documents. Please contact the DGS office for further assistance."

    context = "\n\n".join([doc.page_content for doc in docs])
    
    system_instruction = f"""
    You are Memoria, the official chatbot for the Mandaue City Public Cemetery.
    
    CRITICAL RULES:
    1. Answer using STRICTLY the context provided below.
    2. Do NOT invent premium services, tours, or monuments.
    3. SMARTS: If the user asks broadly about a topic (like "DGS"), summarize all details found in the context.
    4. Prioritize official facts over [PAST USER Q&A].
    {get_base_rules()}    
    
    CONTEXT:
    {context}
    """
    
    messages = [SystemMessage(content=system_instruction)] + formatted_history + [HumanMessage(content=user_question)]
    return model.invoke(messages).content

def run_database_query(user_question, formatted_history):
    """Branch 2: Private Records (JSON file)"""
    json_path = os.path.join(os.path.dirname(__file__), '..', 'records.json')
    
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            db_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # FIXED: Now catches empty/corrupted JSON errors gracefully
        return "I am currently unable to access the cemetery records system due to a connection error."

    json_context = json.dumps(db_data, indent=2)
    system_instruction = f"""
    You are Memoria's secure records assistant. 
    Use ONLY the following JSON records to answer the user's question. 
    If the answer cannot be found, politely state that you cannot find the record.
    Do not expose raw JSON syntax. Act as an analyst parsing data for the user.
    {get_base_rules()}

    DATABASE RECORDS:
    {json_context}
    """
    
    messages = [SystemMessage(content=system_instruction)] + formatted_history + [HumanMessage(content=user_question)]
    return model.invoke(messages).content

def run_general_chat(user_question, formatted_history):
    """Branch 3: Normal conversational AI"""
    system_instruction = f"""
    You are Memoria, the official chatbot for the Mandaue City Public Cemetery. 
    CRITICAL RULES:
    1. This is a local, public municipal cemetery. You DO NOT offer premium services.
    2. If the user makes small talk, politely greet them and remind them you can assist with official cemetery requests and locations.
    {get_base_rules()}
    """
    
    messages = [SystemMessage(content=system_instruction)] + formatted_history + [HumanMessage(content=user_question)]
    return model.invoke(messages).content

def run_clarification(user_question, formatted_history):
    """Branch 4: Ambiguity Resolution"""
    system_instruction = f"""
    You are Memoria. The user asked an ambiguous question (like looking for a relative, e.g., "Where is my grandma?").
    
    Politely ask them to clarify. Specifically, say something similar to: "Are you looking for a specific burial record in our cemetery? If so, please use the Login button above so I can securely search the database. Or are you just asking a general question?"
    
    Keep your tone warm, empathetic, and professional.
    {get_base_rules()}
    """
    messages = [SystemMessage(content=system_instruction)] + formatted_history + [HumanMessage(content=user_question)]
    return model.invoke(messages).content


# ==========================================
# 5. MAIN ENTRY POINT
# ==========================================
def get_response(user_question, chat_history, is_logged_in=False):
    """Called by routes.py. chat_history comes from SQLite."""
    
    formatted_history = format_history(chat_history)
    category = run_router(user_question)
    
    valid_categories = ["CEMETERY_INFO", "CEMETERY_RECORDS", "GENERAL_CHAT", "CLARIFY"]
    if category not in valid_categories:
        category = "GENERAL_CHAT"
        
    print(f"DEBUG - Traffic Cop decided: {category}") 
    
    if category == "CEMETERY_INFO":
        bot_response = run_rag_pipeline(user_question, formatted_history)
        save_to_global_memory(user_question, bot_response)
        return bot_response
    
    elif category == "CEMETERY_RECORDS":
        if not is_logged_in:
             return "It sounds like you want me to search the official cemetery records. To protect families' privacy, you must be logged in to view specific burial data. Please click the **Login** button at the top! If you meant to ask a general public question, just let me know."
        return run_database_query(user_question, formatted_history)
        
    elif category == "CLARIFY":
        return run_clarification(user_question, formatted_history)
        
    else: 
        return run_general_chat(user_question, formatted_history)