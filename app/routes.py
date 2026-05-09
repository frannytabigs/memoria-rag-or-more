import sqlite3
import os
import random
import time
import requests
import re
from flask import current_app as app, render_template, request, jsonify
from app.rag_engine import get_response
from app.database import get_db_connection
import uuid
from dotenv import load_dotenv

load_dotenv()

otp_storage = {}

def normalize_ph_number(phone):
    cleaned = re.sub(r'[^\d+]', '', phone)
    if re.match(r'^09\d{9}$', cleaned):
        return '+63' + cleaned[1:]
    elif re.match(r'^\+639\d{9}$', cleaned):
        return cleaned
    elif re.match(r'^639\d{9}$', cleaned):
        return '+' + cleaned
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/request_otp", methods=["POST"])
def request_otp():
    data = request.get_json()
    raw_phone = data.get("phone", "").strip()
    
    phone = normalize_ph_number(raw_phone)
    if not phone:
        return jsonify({"error": "Invalid Philippine mobile number format."}), 400

    code = 0
    if not otp_storage.get(phone):
        code = str(random.randint(100000, 999999))
        otp_storage[phone] = {
            "code": code,
            "expires_at": time.time() + 300
        }
    
        print(f"DEBUG - Generated OTP for {phone}: {code}")
    else:
        code = otp_storage[phone]['code']
        print(f"DEBUG - OTP already exists for {phone}. Not generating a new one. Existing OTP: {otp_storage[phone]['code']}")

    api_key = os.getenv("TEXTBEE_API_KEY")
    device_id = os.getenv("TEXTBEE_DEVICE_ID")

    if api_key and device_id and "your_textbee" not in api_key:
        # Add these lines to strip away hidden spaces or quotes
        api_key = api_key.strip().strip('"').strip("'")
        device_id = device_id.strip().strip('"').strip("'")
        
        try:
            url = f"https://api.textbee.dev/api/v1/gateway/devices/{device_id}/sendSMS"
            headers = { "x-api-key": api_key }
            payload = {
                "receivers": [phone],
                "smsBody": f"Your Memoria login code is {code}. It expires in 5 minutes."
            }
            x = requests.post(url, json=payload, headers=headers)
            print(f"DEBUG - TextBee Response: {x.status_code} - {x.text}")
            if x.status_code == 201:
                return jsonify({"message": "OTP generated and sent"}), 200
            if x.status_code == 200:
                return jsonify({"message": "OTP generated and sent"}), 200
            return jsonify({"error": "Failed to send SMS"}), 500
        except Exception as e:
            print(f"TextBee Error: {e}")
            return jsonify({"error": "Failed to send SMS"}), 500

    return jsonify({"error": "Failed to send SMS"}), 500
        
@app.route("/api/verify_otp", methods=["POST"])
def verify_otp():
    data = request.get_json()
    raw_phone = data.get("phone", "").strip()
    user_code = data.get("code", "").strip()
    session_id = data.get("session_id")
    
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

        conn = sqlite3.connect('memoria_chat.db')
        cursor = conn.cursor()
    
        cursor.execute('''
            UPDATE chat_logs 
            SET phone_number = ? 
            WHERE session_id = ?
        ''', (phone, session_id))
    
        conn.commit()
        conn.close()

        return jsonify({"success": True, "message": "Login successful","phone": phone}), 200
    else:
        return jsonify({"success": False, "error": "Invalid code"}), 400

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {} 
    
    user_message = data.get("message", "")
    is_logged_in = data.get("is_logged_in", False)
    phone_number = data.get("phone_number", None)
    
    session_id = data.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        
    if not user_message.strip():
        return jsonify({"error": "Message cannot be empty"}), 400

    try:
        conn = get_db_connection()
    
        cursor = conn.execute(
            "SELECT user_message as user, bot_response as bot FROM chat_logs WHERE session_id = ? OR phone_number = ? ORDER BY created_at ASC", 
            (session_id, phone_number,)
        )
    
        history = [dict(row) for row in cursor.fetchall()][-8:]

        # FIX 2: Pass the phone_number so RAG can access memory!
        bot_response = get_response(user_message, history, is_logged_in, phone_number)
    
        # FIX 3: Save the phone_number to the database so ingest.py can vectorize it later!
        conn.execute(
            "INSERT INTO chat_logs (session_id, user_message, bot_response, phone_number) VALUES (?, ?, ?, ?)",
            (session_id, user_message, bot_response, phone_number)
        )
        conn.commit()
    finally:
        conn.close()

    return jsonify({
        "response": bot_response,
        "session_id": session_id
    }), 200