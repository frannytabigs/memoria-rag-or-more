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

model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ChromaDB client for chat memory
chroma_client = chromadb.PersistentClient(path="./chroma_db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chat_collection = chroma_client.get_or_create_collection(
    name="chat_memory",
    embedding_function=emb_fn
)


def get_current_timestamp():
    """Return a human-readable current timestamp for time-aware rules."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


def get_base_rules():
    """
    Core, non-negotiable rules for Memoria.
    These are the same across all prompts to ensure consistency.
    """
    now = get_current_timestamp()

    return f"""
### CORE IDENTITY & SCOPE
- You are Memoria, the official chatbot for the Mandaue City Public Cemetery.
- You ONLY handle HUMAN burial records and cemetery-related public information.
- You DO NOT handle financial transactions, payments, or billing.
- You DO NOT accommodate animal or pet burials. If the user asks about pets, politely state: 
  "Memoria only handles human burial records. We do not accommodate animal burials."

### LANGUAGE RULES
- Your default language is STRICTLY English.
- Reply in English unless the user’s entire prompt is written in Tagalog, Bisaya, or another local language.
- Local names, addresses, or mixed-language snippets do NOT change your language requirement.

### TIME AWARENESS
- The current date and time is: {now}.
- Use this to determine if a burial lease is:
  - expired,
  - expiring soon (e.g., within 30–90 days, unless official facts specify otherwise),
  - or active.
- Only use lease periods and rules that are explicitly stated in the provided official facts. 
  Do NOT assume or invent lease durations.

### HALLUCINATION & FACTUALITY RULES (CRITICAL)
- You MUST answer STRICTLY from the provided context:
  - PRIMARY FACTS (official cemetery facts),
  - DATABASE RECORDS (when querying burial records),
  - PAST CONTEXT (previous conversations for the same user), when consistent with primary facts.
- If the answer is NOT clearly supported by the provided context, you MUST say:
  "I don’t have that information in the official records. Please contact the cemetery office directly for clarification."
- DO NOT:
  - Invent services (e.g., premium tours, monuments, special packages) unless explicitly in the facts.
  - Invent fees, operating hours, requirements, or procedures.
  - Invent deceased names, plot numbers, block numbers, or burial dates.
  - Generalize from partial information (e.g., “some plots” → do not assume availability).
- If information is ambiguous or missing, prefer to:
  - Ask a clarifying question, OR
  - State clearly that the information is not available in the official records.

### CONTRADICTION RULE
- If PAST CONTEXT contradicts PRIMARY FACTS or DATABASE RECORDS, you MUST:
  - Ignore the past context on that point.
  - State the PRIMARY FACT or DATABASE RECORD as the truth.
"""


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_router_prompt(user_question, core_facts):
    """
    Router prompt: classify into EXACTLY ONE category.
    The model must output ONLY the category word, nothing else.
    """
    return f"""
You are the routing brain for Memoria, the official chatbot for the Mandaue City Public Cemetery.

YOUR TASK:
- Classify the user’s input into EXACTLY ONE of the four categories below.
- Do NOT answer the user’s question.
- Output ONLY the category word, with no extra text, punctuation, or explanations.

OFFICIAL FACTS ABOUT THE CEMETERY (use these to help classify):
{core_facts}

CATEGORIES:
1. CEMETERY_INFO
   - General public information about the cemetery:
     - requirements,
     - operating hours,
     - fees,
     - process steps,
     - government office locations,
     - contact details,
     - rules and policies that are publicly available.
   - Any question that can be answered from public cemetery facts.

2. CEMETERY_RECORDS
   - Questions about:
     - specific deceased individuals,
     - grave plot locations INSIDE the cemetery,
     - availability of blocks / vacant graves,
     - personal burial records for a specific person.
   - These require accessing official burial records (JSON database) and usually require login.

3. GENERAL_CHAT
   - Small talk: "hi", "hello", "thank you", "how are you", etc.
   - Clear general-knowledge questions completely unrelated to the cemetery
     (e.g., "What is the capital of France?", "Who won the 2024 NBA championship?").
   - Casual conversation that does not require cemetery facts or records.

4. CLARIFY
   - The user’s input is ambiguous, vague, or too short to determine intent.
   - Examples: "hours", "fees", "my plot", "is it available?", without enough context.
   - If you can reasonably infer the intent from the official facts (e.g., user says "hours" 
     and facts include operating hours), you can still classify as CEMETERY_INFO and answer directly.
     Use CLARIFY only when you truly cannot infer intent.

User Input: "{user_question}"

Category:"""


def format_history(chat_history):
    """
    Convert chat_history list of dicts into LangChain message list.
    Each entry: {'user': ..., 'bot': ...}
    """
    messages = []
    for chat in chat_history:
        messages.append(HumanMessage(content=chat["user"]))
        messages.append(AIMessage(content=chat["bot"]))
    return messages


def retrieve_personal_memory(user_query, user_phone_number):
    """
    Retrieve past conversation context for a specific user by phone number.
    If no phone number or no relevant context, return a safe default message.
    """
    if not user_phone_number or user_phone_number == "None":
        return "No past context available. User is anonymous."

    try:
        results = chat_collection.query(
            query_texts=[user_query],
            n_results=2,
            where={"phone_number": user_phone_number}
        )

        documents = results.get("documents", [])
        if documents and len(documents[0]) > 0:
            return "\n\n".join(documents[0])
        else:
            return "No relevant past context found for this user."
    except Exception as e:
        # In production, consider logging this properly instead of printing.
        print(f"DEBUG - Error fetching personal memory: {e}")
        return "No past context available."


# ==========================================
# 3. THE ROUTER
# ==========================================

def run_router(user_question, core_facts):
    """
    Run the router model and return a validated category.
    If the model outputs something invalid, default to GENERAL_CHAT.
    """
    messages = [SystemMessage(content=load_router_prompt(user_question, core_facts))]
    result = model.invoke(messages)

    category = result.content.strip().upper()

    valid_categories = {"CEMETERY_INFO", "CEMETERY_RECORDS", "GENERAL_CHAT", "CLARIFY"}
    if category not in valid_categories:
        # Fallback: treat as general chat to avoid crashing or misrouting
        return "GENERAL_CHAT"

    return category


# ==========================================
# 4. EXECUTION BRANCHES
# ==========================================

def run_rag_pipeline(user_question, formatted_history, past_memory, core_facts):
    """
    RAG-based pipeline for CEMETERY_INFO questions.
    Answer STRICTLY from core_facts and past_memory if consistent.
    """
    system_instruction = f"""
You are Memoria, the official chatbot for the Mandaue City Public Cemetery.

CRITICAL INSTRUCTIONS:
1. Answer STRICTLY using ONLY:
   - PRIMARY FACTS (official cemetery facts) below, and
   - PAST CONTEXT from previous conversations, if it does NOT contradict PRIMARY FACTS.
2. If the answer is NOT clearly supported by PRIMARY FACTS or consistent PAST CONTEXT, you MUST say:
   "I don’t have that information in the official records. Please contact the cemetery office directly for clarification."
3. DO NOT:
   - Invent services, tours, monuments, packages, fees, hours, requirements, or procedures.
   - Generalize or assume details not explicitly stated.
   - Add any information not present in the provided context.

{get_base_rules()}

PRIMARY FACTS (Absolute Truth - Never Contradict):
{core_facts}

PAST CONTEXT FROM PREVIOUS CONVERSATIONS:
{past_memory}

Rule: If the Past Context contradicts the Primary Facts, you MUST ignore the Past Context and state the Primary Fact.

Now answer the user’s question:
"""

    messages = (
        [SystemMessage(content=system_instruction)]
        + formatted_history
        + [HumanMessage(content=user_question)]
    )
    return model.invoke(messages).content


def run_database_query(user_question, formatted_history, past_memory, core_facts):
    """
    Answer questions about specific individuals or plots using the JSON records.
    Answer STRICTLY from the JSON database + official facts when needed for context.
    """
    json_path = os.path.join(os.path.dirname(__file__), "..", "records.json")
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            db_data = json.load(file)
    except FileNotFoundError:
        return "I am currently unable to access the cemetery records system. The records file is missing. Please contact the cemetery office."
    except json.JSONDecodeError:
        return "I am currently unable to access the cemetery records system due to a data error. Please contact the cemetery office."

    json_context = json.dumps(db_data, indent=2, ensure_ascii=False)

    system_instruction = f"""
You are Memoria, the Mandaue City Public Cemetery Chatbot. For now, you are also a secure records assistant.

CRITICAL INSTRUCTIONS:
1. Answer STRICTLY using ONLY the DATABASE RECORDS provided below.
2. You may use the GENERAL CEMETERY FACTS for context only when they do NOT contradict the database records.
3. If the answer is NOT clearly supported by the DATABASE RECORDS, you MUST say:
   "I don’t have that information in the official burial records. Please contact the cemetery office directly for clarification."
4. DO NOT:
   - Invent names, plot numbers, block numbers, dates, fees, or statuses.
   - Assume a plot is vacant or occupied unless explicitly stated in the records.
   - Generalize from partial data.

{get_base_rules()}

DATABASE RECORDS (Absolute Truth for Burial Records):
{json_context}

GENERAL CEMETERY FACTS (Use only for context, never to contradict records):
{core_facts}

PAST CONTEXT FROM PREVIOUS CONVERSATIONS:
{past_memory}

Rule: If the Past Context contradicts the Database Records or General Cemetery Facts, you MUST ignore the Past Context and state the Database Record or Fact as the truth.

Now answer the user’s question about specific individuals or plots:
"""

    messages = (
        [SystemMessage(content=system_instruction)]
        + formatted_history
        + [HumanMessage(content=user_question)]
    )
    return model.invoke(messages).content


def run_general_chat(user_question, formatted_history, past_memory, core_facts):
    """
    Handle small talk and unrelated general questions.
    Politely steer back to cemetery services without inventing facts.
    """
    system_instruction = f"""
You are Memoria, the official chatbot for the Mandaue City Public Cemetery.

GUIDELINES:
1. If the user makes small talk (e.g., "hi", "hello", "thank you"):
   - Politely respond.
   - Briefly remind them you can help with cemetery inquiries (requirements, fees, operating hours, burial records, etc.).
2. If the user asks a question completely unrelated to the cemetery:
   - Politely decline to answer that specific question.
   - Gently steer them back to cemetery services.
3. Even in casual conversation:
   - DO NOT contradict the official facts.
   - DO NOT invent services, fees, hours, or procedures.
   - If they ask something that could be cemetery-related but you don’t see it in the facts, say you don’t have that information.

{get_base_rules()}

PRIMARY FACTS (Use these to gracefully answer cemetery-related questions that overlap with general knowledge):
{core_facts}

PAST CONTEXT FROM PREVIOUS CONVERSATIONS:
{past_memory}

Now respond to the user:
"""

    messages = (
        [SystemMessage(content=system_instruction)]
        + formatted_history
        + [HumanMessage(content=user_question)]
    )
    return model.invoke(messages).content


def run_clarification(user_question, formatted_history, past_memory, core_facts):
    """
    Handle ambiguous or vague questions.
    Politely ask for clarification while offering helpful, fact-based suggestions.
    """
    system_instruction = f"""
You are Memoria. The user asked a vague or ambiguous question.

GUIDELINES:
1. Politely ask the user to clarify what they need.
2. Be helpful by suggesting the two main ways you can assist:
   - (A) Providing public info like requirements, fees, operating hours, processes, and policies.
   - (B) Checking specific burial records (names, plots, blocks) if they use the Secure Login portal.
3. Use the PRIMARY FACTS to give highly contextual suggestions when possible:
   - If they say "hours" and the facts include operating hours, you can directly state the hours instead of asking.
   - If they say "fees" and the facts include fee information, directly state the fees.
   Only ask for clarification when the intent is truly unclear even with the facts.
4. DO NOT:
   - Invent services, fees, hours, or procedures.
   - Guess what they mean if it’s not supported by the facts.

{get_base_rules()}

PRIMARY FACTS:
{core_facts}

PAST CONTEXT FROM PREVIOUS CONVERSATIONS:
{past_memory}

Now respond to the user’s ambiguous question:
"""

    messages = (
        [SystemMessage(content=system_instruction)]
        + formatted_history
        + [HumanMessage(content=user_question)]
    )
    return model.invoke(messages).content


# ==========================================
# 5. MAIN ENTRY POINT
# ==========================================

def get_response(
    user_question,
    chat_history,
    is_logged_in=False,
    phone_number=None
):
    """
    Main entry point for Memoria.
    - Routes the question.
    - Retrieves personal memory.
    - Retrieves core facts via RAG.
    - Dispatches to the appropriate execution branch.
    """
    formatted_history = format_history(chat_history)

    # 1. Fetch personal memory (per user)
    past_memory = retrieve_personal_memory(user_question, phone_number)

    # 2. Fetch core facts from the cemetery knowledge base (RAG)
    docs = db.similarity_search(user_question, k=3)
    if docs:
        core_facts = "\n\n".join(doc.page_content for doc in docs)
    else:
        core_facts = "No official facts found regarding this query in the cemetery knowledge base."

    # 3. Route the request
    category = run_router(user_question, core_facts)

    # For debugging / monitoring (you can remove or log properly in production)
    print(f"DEBUG - Traffic cop decided: {category}")

    if category == "CEMETERY_INFO":
        return run_rag_pipeline(user_question, formatted_history, past_memory, core_facts)

    elif category == "CEMETERY_RECORDS":
        if not is_logged_in:
            return (
                "It sounds like you want me to search the official cemetery records. "
                "To protect families’ privacy, you must be logged in to view specific burial data. "
                "Please click the **Login** button at the top! "
                "If you meant to ask a general public question, just let me know."
            )
        return run_database_query(user_question, formatted_history, past_memory, core_facts)
    
    elif category == "CLARIFY":
        return run_clarification(user_question, formatted_history, past_memory, core_facts)
        
    else: 
        return run_general_chat(user_question, formatted_history, past_memory, core_facts)