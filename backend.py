import time
import numpy as np
import torch
import json
import os
# from faster_whisper import WhisperModel
import whisper
from groq import Groq
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import threading
import tempfile
import queue
import wave

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

load_dotenv()

BASE_SYSTEM_PROMPT = """
﻿You are an AI assistant designed to provide crucial agricultural guidance. Your primary focus is on wheat cultivation practices, but you also have an extensive hardcoded knowledge base for olive farming, rice (دھان) cultivation, and soil tillage techniques. All responses must be in Urdu, concise, and tailored to local conditions, seasons, and specific crop requirements.
General Guidelines:
1. Language & Style:


   * تمام جوابات اردو میں ہوں۔
   * آسان اور سیدھے الفاظ استعمال کریں تاکہ کسان آسانی سے سمجھ سکیں۔
   * جوابات 4-5 سطروں کے اندر رکھیں، جب تک کہ اضافی تفصیل نہ ضروری ہو۔
   2. Topic Enforcement:


      * اگر صارف کا سوال زرعی معاملات سے ہٹ کر ہو تو مہذب انداز میں زرعی مشورے دینے پر زور دیں۔
      3. Urgent Alerts:


         * ضروری اور فوری مسائل (مثلاً شدید کیڑوں کا حملہ، خراب موسم وغیرہ) کو موٹا لکھ کر نمایاں کریں۔




Crop-Specific Knowledge Base:
1. گندم (Wheat):
         * بوائی کا وقت: 1 سے 15 نومبر تک بہترین۔
         * بیج کی مقدار: اگر 15 نومبر تک بویا جائے تو 50 کلو/اکڑ؛ بعد میں 60 کلو/اکڑ۔
         * زمین کی تیاری: کاٹن، مکئی یا شوگر کین کے فصل صاف کرنے کے بعد روٹیویٹر اور دوہری ہل چلائیں۔
         * بیج کی اقسام:
         * بارش والے علاقوں میں: GA 2002، عقب 2000، NARC 2009
         * آبپاشی والے علاقوں میں: عقب 2000، پنجاب-1
         * آبپاشی: پہلی آبپاشی کے بعد "ہیریو" دو بار کریں؛ بارش والے علاقوں میں گہرے ہل چلانے سے بارش کا پانی زمین میں برقرار رہتا ہے۔
         * کیڑے مار دوا: اپھڈ کے حملے کا جائزہ لیں اور منظور شدہ زہریں استعمال کریں؛ دھند، ہوا یا بارش کے دوران سپرے سے گریز کریں۔
         * جڑی بوٹیوں کا کنٹرول: ہربیسائڈ سپرے کے لیے فلیٹ فین نوزلز استعمال کریں۔
2. زیتون (Olive):
         * زیتون کی کاشت:
         * Olive BARI کے اصولوں کی پیروی کریں، جس میں آبپاشی، چھان بین، اور چھانٹی کے اصول شامل ہیں۔
         * علاقے کے مطابق پانی، فصل کی کٹائی، اور کیڑوں/بیماریوں کے کنٹرول کے لیے مقامی رہنمائی اور Olive BARI کے دستاویز میں مذکور مشورے استعمال کریں۔
3. دھان (Rice) – فائن اقسام:
* فائن اقسام (باسمتی):
   * باسمتی 370، باسمتی پاک، سپر باسمتی، باسمتی 2000، شاہین باسمتی، باسمتی 515، پنجاب باسمتی، چناب باسمتی، کسان باسمتی، سپر باسمتی 2019، سپر گولڈ
* غیر باسمتی اقسام:
   * PK 1121 ایرومیٹک، PK 386
* کورس اقسام:
   * ARI 6، KS282، KS133، KS434
* دوغلی اقسام:
   * PHB 7، V26، شانہ شاہ 2، پرائیڈ 1، RIZ سوئفٹ
* ممنوعہ اقسام:
   * کشمیرا، مالٹا، سپر فائن، سپرا، اور سپری ہرگز کاشت نہ کریں۔
* شرح بیج فی ایکڑ:
   * باسمتی اقسام کے لیے: 4-5 کلوگرام
   * کورس اور دوغلی اقسام کے لیے: 6-7 کلوگرام
* نرسری لگانے کا وقت:
   * کورس اقسام: 20 مئی تا 7 جون
   * فائن اقسام: 1 جون تا 25 جون (شاہین اور کسان باسمتی کے لیے: 15 جون تا 30 جون)
* نرسری کے طریقے:
   * گیلا طریقہ (پانی میں بھیگانا): ہل چلانے اور پانی بھرنے کے بعد بیج کو مخصوص محلول میں 24 گھنٹے بھگوئیں، پھر مناسب وقت پر اکھاڑیں۔
   * خشک طریقہ: زمین کو خشک طریقے سے تیار کر کے چھوٹی کیاریوں میں بیج کا چھٹہ دیں اور گلی سڑی گوبر یا بھوسے کی ہلکی تہہ بچھا دیں۔
   * راب کا طریقہ: خشک زمین میں ہل چلانے کے بعد راکھ اور گوبر شامل کریں، پھر بیج کا چھٹہ دیں۔
* بیج کا زہر لگانا:
   * کاشت سے دو ہفتے قبل تجویز کردہ زہر (0.5 گرام/ملی لیٹر فی کلو بیج) لگائیں یا 24 گھنٹے پانی میں زہر ملا کر بیج بھگوئیں۔
* منتقلی کے وقت پودوں کی عمر:
   * مثالی عمر 25-40 دن؛ کم یا زیادہ عمر کے پودے پیداوار پر منفی اثر ڈالتے ہیں۔
* پودوں کی تعداد:
   * نرسری کی منتقلی کے وقت ایک سوراخ میں دو پودے لگائیں؛ مثالی طور پر ایک ایکڑ میں 80,000 پودے ہونے چاہئیں۔
________________
       
4. زمین کی تیاری (Till کی کاشت):
         * زمین کی تیاری کے اصول:
         * مقامی زمین کی خصوصیات کو مدِنظر رکھتے ہوئے، مناسب مشینری اور ہل چلانے کے طریقے اپنائیں۔
         * فصل کی پیداوار کو بڑھانے کے لیے زمین کو اچھی طرح ہموار کر کے چھوٹی کیاریوں میں تقسیم کریں۔
         * اضافی رہنمائی:
         * Till ki kasht کے متعلق بنیادی اصول اور تصاویر دستیاب ہیں؛ مقامی حالات کے مطابق مشورے دیں اور فصل کی بہتر نشوونما کے لیے جدید طریقے اختیار کریں۔




Example Interaction:
User Query (Urdu):
"مجھے یہ بتاؤ کہ گندم میں کیڑے مارنے کے لیے کون سی زہر یوز کرنی چاہیے۔"
Your Response (Urdu):
گندم کے کیڑوں کے لیے سفارش کردہ زہر:
         * ایندوسلفن: وسیع الطیف زہر، سست تیلے اور مکڑیوں کے خلاف مؤثر۔
         * سیالوٹرن: سست تیلے کے لیے بہترین۔
         * ایمیٹاف: تیزی سے اثر کرنے والا۔
ہدایات: زہر کا استعمال ہدایت کے مطابق کریں اور سپرے سے پہلے موسم کی پیشگوئی ضرور چیک کریں۔




Usage Instructions:
            * Crop-Specific Advice:
            * اگر سوال گندم کا ہو تو اوپر دیے گئے گندم کے اصول استعمال کریں۔
            * اگر سوال زیتون، دھان یا زمین کی تیاری سے متعلق ہو تو متعلقہ سیکشن سے مشورے فراہم کریں۔
            * موسم اور جغرافیائی حالات:
            * ہر مشورے کو مخصوص موسم، علاقائی حالات اور زرعی معیارات کے مطابق ڈھالیں۔
            * فوری مسائل:
            * اہم یا فوری مسائل کو موٹا لکھ کر نمایاں کریں۔
"""

CONVERSATION_HISTORY_FILE = "conversation_history.json"
conversation_history = []
history_lock = threading.Lock()
MAX_HISTORY_LENGTH = 5

if not os.path.exists(CONVERSATION_HISTORY_FILE):
    with open(CONVERSATION_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

try:
    tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY", "sk_80cc5c6c67fc1a5775ee75ff13409ad094ade46de30ad7aa"))
    client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))
    print("API clients initialized successfully")
except Exception as e:
    print(f"Error initializing API clients: {e}")

try:
    vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=True)
    print("VAD model loaded successfully")
except Exception as e:
    print(f"Error loading VAD model: {e}")
    vad_model = None

# try:
#     model_size = "large-v3-turbo"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     compute_type = "float16" if device == "cuda" else "int8"
#     whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
#     print(f"Whisper model loaded successfully on {device}")
# except Exception as e:
#     print(f"Error loading Whisper model: {e}")
#     whisper_model = None

# try:
whisper_model = whisper.load_model("large")
print("Whisper model loaded successfully")
# except Exception as e:
#     print(f"Error loading Whisper model: {e}")
#     whisper_model = None

RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 2.0
SPEECH_THRESHOLD = 0.7

audio_queue = queue.Queue(maxsize=10)
audio_buffer = bytearray()
last_voice_time = time.time()
is_listening = False
is_speaking = False
is_processing = False

def save_conversation_history():
    with open(CONVERSATION_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)

def update_conversation_history(user_message, assistant_response):
    global conversation_history
    with history_lock:
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": assistant_response})
        if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
            conversation_history = conversation_history[-MAX_HISTORY_LENGTH * 2:]
        save_conversation_history()

def get_conversation_context():
    with history_lock:
        history = conversation_history[-MAX_HISTORY_LENGTH * 2:] if len(conversation_history) > MAX_HISTORY_LENGTH * 2 else conversation_history
    if not history:
        return "ابھی تک کوئی گفتگو نہیں ہوئی۔"
    return "\n".join([f"{entry['role']}: {entry['content']}" for entry in history])

def process_audio_data(data):
    global last_voice_time, audio_buffer, is_speaking, is_processing
    if is_processing:
        print("Processing in progress, skipping new audio chunk")
        return False

    try:
        audio_np = np.frombuffer(data, dtype=np.float32)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_data = audio_int16.tobytes()

        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        speech_prob = vad_model(audio_tensor, sr=RATE).mean().item()
        socketio.emit('speech_probability', {'probability': speech_prob})
        print(f"Speech probability: {speech_prob}")

        if speech_prob > SPEECH_THRESHOLD and not is_speaking:
            print("Speech started, beginning accumulation")
            is_speaking = True
            audio_buffer = bytearray()  # Reset buffer at start of speech
            audio_buffer.extend(audio_data)
            last_voice_time = time.time()
        elif speech_prob > SPEECH_THRESHOLD and is_speaking:
            audio_buffer.extend(audio_data)
            last_voice_time = time.time()
        elif speech_prob <= SPEECH_THRESHOLD and is_speaking:
            audio_buffer.extend(audio_data)
            if (time.time() - last_voice_time) > SILENCE_THRESHOLD:
                print(f"Silence detected after speech, buffer size: {len(audio_buffer)} bytes")
                is_speaking = False
                is_processing = True
                socketio.emit('start_processing')
                return True
        return False
    except Exception as e:
        print(f"Error processing audio chunk for VAD: {e}")
        is_speaking = False  # Reset speaking state on error
        return False

def transcribe_audio(audio_data):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                wf.writeframes(audio_data)
            wav_filename = temp_file.name

        result = whisper_model.transcribe(wav_filename, language="ur")
        transcribed_text = result["text"]
        print(f"Transcribed Text: {transcribed_text}")
        os.unlink(wav_filename)
        return transcribed_text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def get_ai_response(text):
    try:
        conversation_context = get_conversation_context()
        full_prompt = f"{BASE_SYSTEM_PROMPT}\n## گفتگو کا خلاصہ:\n{conversation_context}"
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
        )
        response_text = chat_completion.choices[0].message.content
        print(f"Groq Response: {response_text}")
        update_conversation_history(text, response_text)
        return response_text
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return "معذرت، میں آپ کی درخواست پر عمل نہیں کر سکا۔"

def generate_audio_response(text):
    try:
        audio_response = tts_client.text_to_speech.convert(
            text=text,
            voice_id="Sxk6njaoa7XLsAFT7WcN",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        audio_bytes = b''.join(chunk for chunk in audio_response)
        print(f"Generated audio response, size: {len(audio_bytes)} bytes")
        return audio_bytes
    except Exception as e:
        print(f"Error generating audio response: {e}")
        return None

def process_voice_input(audio_segment):
    global is_processing, is_listening
    try:
        transcribed_text = transcribe_audio(audio_segment)
        if transcribed_text and transcribed_text.strip():
            socketio.emit('transcribed_text', {'text': transcribed_text})
            response_text = get_ai_response(transcribed_text)
            audio_data = generate_audio_response(response_text)
            if audio_data:
                socketio.emit('audio_playing')
                socketio.emit('response', {'text': response_text, 'audio': audio_data})
            else:
                socketio.emit('response', {'text': response_text})
        else:
            print("No valid transcription; skipping processing")
            is_processing = False
            socketio.emit('system_ready', {'ready': True})
    except Exception as e:
        print(f"Error processing voice input: {e}")
        is_processing = False
        socketio.emit('system_ready', {'ready': True})

@socketio.on('audio_ended')
def handle_audio_ended():
    global is_processing, is_listening
    print("Audio playback confirmed ended")
    is_processing = False
    socketio.emit('system_ready', {'ready': True})
    if not is_listening:
        print("Listening stopped after audio playback")

def audio_processing_worker():
    while True:
        audio_segment = audio_queue.get()
        if audio_segment is None:
            break
        process_voice_input(audio_segment)
        audio_queue.task_done()

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('system_ready', {'ready': True})

@socketio.on('disconnect')
def handle_disconnect():
    global is_listening
    print('Client disconnected')
    is_listening = False

@socketio.on('start_listening')
def handle_start_listening():
    global is_listening, audio_buffer, last_voice_time, is_speaking, is_processing
    if is_listening or is_processing:
        print("Already listening or processing, ignoring start_listening")
        return
    print("Starting listening...")
    audio_buffer = bytearray()
    last_voice_time = time.time()
    is_speaking = False
    is_listening = True
    is_processing = False

@socketio.on('stop_listening')
def handle_stop_listening():
    global is_listening, audio_buffer, is_speaking
    if not is_listening:
        print("Not listening, ignoring stop_listening")
        return
    print("Stopping listening...")
    is_listening = False
    if audio_buffer and is_speaking and not is_processing:
        if not audio_queue.full():
            print(f"Enqueuing buffer on stop: {len(audio_buffer)} bytes")
            audio_queue.put(bytes(audio_buffer))
        else:
            print("Audio queue full. Dropping segment.")
    audio_buffer = bytearray()
    is_speaking = False

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    global is_listening, audio_buffer, is_processing
    if not is_listening or is_processing:
        print("Skipping audio chunk: not listening or processing")
        return
    try:
        if process_audio_data(data['audio']):
            if not audio_queue.full():
                print(f"Enqueuing buffer: {len(audio_buffer)} bytes")
                audio_queue.put(bytes(audio_buffer))
            else:
                print("Audio queue full. Dropping segment.")
            audio_buffer = bytearray()  # Reset buffer after enqueuing
    except Exception as e:
        print(f"Error handling audio chunk: {e}")
        audio_buffer = bytearray()  # Reset buffer on error
        is_processing = False
        socketio.emit('system_ready', {'ready': True})

@socketio.on('check_ready')
def handle_check_ready():
    global is_processing
    socketio.emit('system_ready', {'ready': not is_processing})

@app.route('/process_text', methods=['POST'])
def process_text():
    global is_processing, is_listening
    if is_processing:
        return jsonify({'error': 'System is currently processing'}), 429
    try:
        is_processing = True
        socketio.emit('start_processing')
        data = request.json
        text = data.get('text', '')
        if not text:
            is_processing = False
            socketio.emit('system_ready', {'ready': True})
            return jsonify({'error': 'No text provided'}), 400
        response_text = get_ai_response(text)
        audio_data = generate_audio_response(response_text)
        if audio_data:
            socketio.emit('audio_playing')
            socketio.emit('response', {'text': response_text, 'audio': audio_data})
        else:
            socketio.emit('response', {'text': response_text})
        return jsonify({'response': response_text})
    except Exception as e:
        print(f"Error processing text: {e}")
        is_processing = False
        socketio.emit('system_ready', {'ready': True})
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        worker_thread = threading.Thread(target=audio_processing_worker, daemon=True)
        worker_thread.start()
        print("Starting server on http://localhost:5000")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
    finally:
        audio_queue.put(None)
