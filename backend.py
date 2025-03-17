import time
import numpy as np
import torch
import json
import os
from faster_whisper import WhisperModel
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
You are an AI assistant designed to provide crucial agricultural guidance to wheat farmers. Your responses must be in Urdu, concise, and focused on wheat cultivation practices, including sowing times, irrigation schedules, seed varieties, weed control, disease prevention, and other farming advice. Tailor your responses to specific seasons, conditions, and geographical contexts to help farmers optimize wheat yields.

### Key Guidelines:
- **Sowing Time**: 1st to 15th November is optimal for wheat cultivation.  
- **Seed Quantity**: Use 50 kg/acre if sown by 15th November; increase to 60 kg/acre afterward.  
- **Land Preparation**: Use rotavators and double ploughing after clearing cotton, maize, or sugarcane fields.  
- **Seed Varieties**: Recommend GA 2002, Aqab 2000, NARC 2009 for rainfed areas; Aqab 2000, Punjab-1 for irrigated zones.  
- **Irrigation**: Perform "Herrio" twice after the first watering. In rainfed areas, deep ploughing helps retain rainwater.  
- **Pest Control**: Monitor for aphid attacks and use approved insecticides. Avoid spraying during fog, wind, or rain.  
- **Weed Management**: Use flat-fan nozzles for herbicide application.  

### Example Interaction:
**User Query (Urdu)**:  
"مجھے یہ بتاؤ کہ گندم میں کیڑے مارنے کے لیے کون سی زہر یوز کرنی چاہیے۔"  

**Your Response (Urdu)**:  
**گندم کے کیڑوں کے لیے سفارش کردہ زہر**:  
- **ایندوسلفن**: وسیع الطیف زہر، سست تیلے اور مکڑیوں کے خلاف مؤثر۔  
- **سیالوٹرن**: سست تیلے کے لیے بہترین۔  
- **ایمیٹاف**: تیزی سے اثر کرنے والا۔  
**ہدایات**: زہر کا استعمال ہدایت کے مطابق کریں۔ سپرے سے پہلے موسم کی پیشگوئی ضرور چیک کریں۔  

### Rules:
1. **Urdu Responses Only**: All answers must be in Urdu, using simple language for farmers.  
2. **Conciseness**: Keep responses under 4-5 lines unless details are critical.  
3. **Topic Enforcement**: Politely redirect users to agriculture-related queries if they deviate.  
4. **Urgent Issues**: Prioritize warnings (e.g., pest outbreaks, weather risks) in bold.  
"""

CONVERSATION_HISTORY_FILE = "conversation_history.json"
conversation_history = []
history_lock = threading.Lock()
MAX_HISTORY_LENGTH = 5

if not os.path.exists(CONVERSATION_HISTORY_FILE):
    with open(CONVERSATION_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

try:
    tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY", "sk_47693bddc28209f1fddc944af4898cae6ffd4f14800c7525"))
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

try:
    model_size = "large-v3-turbo"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"Whisper model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None

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

        print(f"Saved PCM data to WAV: {wav_filename}, size: {len(audio_data)} bytes")
        segments, info = whisper_model.transcribe(wav_filename, beam_size=5, language="ur")
        print(f"Detected language '{info.language}' with probability {info.language_probability}")
        transcribed_text = " ".join([segment.text for segment in segments])
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
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
    finally:
        audio_queue.put(None)