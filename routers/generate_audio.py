 
import pyttsx3
import uuid
import os
import hashlib

AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def clear_audio_folder():
    """Delete all .mp3 files in the audio directory"""
    for file in os.listdir(AUDIO_DIR):
        if file.endswith(".mp3"):
            os.remove(os.path.join(AUDIO_DIR, file))

def generate_tts_audio(text: str) -> str:
    """Generate a TTS audio file with female voice for the given text"""
    filename = f"{hashlib.md5(text.encode()).hexdigest()}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    if not os.path.exists(filepath):
        engine = pyttsx3.init()

 
        voices = engine.getProperty('voices')
        female_voice = None
        for voice in voices:
            if 'female' in voice.name.lower() or 'female' in voice.id.lower():
                female_voice = voice.id
                break

        if female_voice:
            engine.setProperty('voice', female_voice)
        else:
            print("⚠️ Female voice not found. Using default voice.")

 
        engine.setProperty('rate', 170) 

        engine.save_to_file(text, filepath)
        engine.runAndWait()

    return f"/static/audio/{filename}"
