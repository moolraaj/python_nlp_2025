# routers/generate_audio.py
 
import os
 
import hashlib
 
import logging
 
import requests
 
logger = logging.getLogger(__name__)
 
AUDIO_DIR = "static/audio"
 
os.makedirs(AUDIO_DIR, exist_ok=True)
 
def clear_audio_folder():
 
    """Delete all .mp3 files in the audio directory"""
 
    try:
 
        for file in os.listdir(AUDIO_DIR):
 
            if file.endswith((".mp3", ".aiff", ".wav")):
 
                os.remove(os.path.join(AUDIO_DIR, file))
 
        logger.info("✅ Cleared old audio files")
 
    except Exception as e:
 
        logger.error(f"❌ Error clearing audio files: {e}")
 
def generate_tts_audio(text: str) -> str:
 
    """Generate TTS using web service (cross-platform solution)"""
 
    try:
 
        filename = f"{hashlib.md5(text.encode()).hexdigest()}.mp3"
 
        filepath = os.path.join(AUDIO_DIR, filename)
 
        if not os.path.exists(filepath):
 
            logger.info(f"🔊 Using web TTS service for: '{text}'")
 
            # Option A: Use Google TTS (free)
 
            try:
 
                from gtts import gTTS
 
                tts = gTTS(text=text, lang='en', slow=False)
 
                tts.save(filepath)
 
                logger.info(f"🎵 Generated web TTS audio: {filename}")
 
            except ImportError:
 
                logger.warning("gTTS not available, using fallback")
 
                # Fallback: create empty file
 
                with open(filepath, 'w') as f:
 
                    f.write('')
 
        return f"/static/audio/{filename}"
 
    except Exception as e:
 
        logger.error(f"❌ Web TTS generation failed: {e}")
 
        # Create fallback file
 
        filename = f"{hashlib.md5(text.encode()).hexdigest()}.mp3"
 
        filepath = os.path.join(AUDIO_DIR, filename)
 
        with open(filepath, 'w') as f:
 
            f.write('')
 
        return f"/static/audio/{filename}"