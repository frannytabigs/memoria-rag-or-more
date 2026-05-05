import os
import random
import time
import requests
import re  # NEW: Import regex for validation
from flask import current_app as app, render_template, request, jsonify
from app.rag_engine import get_response
from app.database import get_db_connection
import uuid
from dotenv import load_dotenv

load_dotenv()  # Load .env variables
otp_storage = {}

def normalize_ph_number(phone):
    """
    Cleans and validates a Philippine mobile number.
    Returns the normalized +639XXXXXXXXX string, or None if invalid.
    """
    # 1. Strip all characters except digits and the '+' sign
    cleaned = re.sub(r'[^\d+]', '', phone)
    
    # 2. Check and convert formats
    if re.match(r'^09\d{9}$', cleaned):
        return '+63' + cleaned[1:]      # Converts 09123456789 -> +639123456789
    elif re.match(r'^\+639\d{9}$', cleaned):
        return cleaned                  # Already perfect
    elif re.match(r'^639\d{9}$', cleaned):
        return '+' + cleaned            # Missing the plus sign
    
    return None # Fails validation

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/request_otp", methods=["POST"])
def request_otp():
    data = request.get_json()
    raw_phone = data.get("phone", "").strip()
    
    # Validate and normalize
    phone = normalize_ph_number(raw_phone)
    if not phone:
        return jsonify({"error": "Invalid Philippine mobile number format."}), 400

    code = str(random.randint(100000, 999999))
    
    otp_storage[phone] = {
        "code": code,
        "expires_at": time.time() + 300
    }
    
    print(f"DEBUG - Generated OTP for {phone}: {code}")

    api_key = os.getenv("TEXTBEE_API_KEY")
    device_id = os.getenv("TEXTBEE_DEVICE_ID")
    
    if api_key and device_id and "your_textbee" not in api_key:
        try:
            url = f"https://api.textbee.dev/api/v1/gateway/devices/{device_id}/send-sms"
            headers = {  
                "x-api-key": api_key 
            }
            payload = {
                "recipients": [phone],
                "message": f"Your Memoria login code is {code}. It expires in 5 minutes."
            }
            x = requests.post(url, json=payload, headers=headers)
            print(x.text)  # Log TextBee response for debugging
        except Exception as e:
            print(f"TextBee Error: {e}")
            return jsonify({"error": "Failed to send SMS"}), 500

    return jsonify({"message": "OTP generated and sent"}), 200

@app.route("/api/verify_otp", methods=["POST"])
def verify_otp():
    data = request.get_json()
    raw_phone = data.get("phone", "").strip()
    user_code = data.get("code", "").strip()
    
    # We must normalize it here too, otherwise it won't match the storage!
    phone = normalize_ph_number(raw_phone)
    if not phone:
        return jsonify({"success": False, "error": "Invalid phone format."}), 400
    
    if phone not in otp_storage:
        return jsonify({"success": False, "error": "No OTP requested for this number"}), 400
        
    record = otp_storage[phone]
    
    if time.time() > record["expires_at"]:
        del otp_storage[phone]
        return jsonify({"success": False, "error": "OTP has expired"}), 400
        
    if record["code"] == user_code:
        del otp_storage[phone] 
        return jsonify({"success": True, "message": "Login successful"}), 200
    else:
        return jsonify({"success": False, "error": "Invalid code"}), 400

@app.route("/api/chat", methods=["POST"])
def chat():
    # 1. Fallback to an empty dictionary if JSON is missing
    data = request.get_json() or {} 
    
    user_message = data.get("message", "")
    is_logged_in = data.get("is_logged_in", False)
    
    # 2. THE FIX: Grab the session_id. If it is None, empty, or missing, generate a new one.
    session_id = data.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        
    if not user_message.strip():
        return jsonify({"error": "Message cannot be empty"}), 400

    try:
        conn = get_db_connection()
    
        # Load history from SQLite
        cursor = conn.execute(
            "SELECT user_message as user, bot_response as bot FROM chat_logs WHERE session_id = ? ORDER BY created_at ASC", 
            (session_id,)
        )
    
        # We fetch the rows, but the [-8:] tells Python to ONLY keep the last 8 conversations!
        # This prevents the AI's context window from getting overloaded.
        history = [dict(row) for row in cursor.fetchall()][-8:]

        # Pass the message, history, and login status to the engine!
        bot_response = get_response(user_message, history, is_logged_in)
    
        # Save the new message to SQLite so it remembers it next time
        conn.execute(
            "INSERT INTO chat_logs (session_id, user_message, bot_response) VALUES (?, ?, ?)",
            (session_id, user_message, bot_response)
        )
        conn.commit()
    finally:
        conn.close()

    return jsonify({
        "response": bot_response,
        "session_id": session_id
    }), 200