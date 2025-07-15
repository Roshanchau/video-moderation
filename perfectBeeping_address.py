from moviepy import *
import numpy as np
import os
import json
import subprocess
import sys
import re
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")

def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        if 'ffmpeg version' not in result.stdout:
            raise Exception("FFmpeg not properly installed")
    except Exception as e:
        print("\nERROR: FFmpeg is required but not properly installed.")
        print("Please install FFmpeg:")
        print("Windows: choco install ffmpeg")
        print("Mac: brew install ffmpeg")
        print("Linux: sudo apt-get install ffmpeg")
        sys.exit(1)

VULGAR_WORDS = {
    "fucking": ["fucking"],
    "fuck": ["fuck", "fucking", "fucker", "motherfucker"],
    "shit": ["shit", "bullshit", "shitty"],
    "asshole": ["asshole", "arsehole"],
    "bitch": ["bitch", "bitches", "bitching"],
    "damn": ["damn", "goddamn"],
    "crap": ["crap", "crappy"],
    "phone_number": [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        r'\b\d{3}\s\d{3}\s\d{4}\b',
        r'\b\d{10}\b',
        r'\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b'
    ]
}

def convert_timestamp_to_seconds(timestamp_str):
    parts = list(map(float, timestamp_str.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0

def detect_addresses(text_with_timestamps):
    """Use Gemini to detect addresses in the transcribed text"""
    client = genai.Client(api_key=gemini_api_key)
    
    prompt = f"""
    Analyze this transcript and identify any addresses mentioned. Return JSON with:
    - original_text: The exact phrase containing the address
    - timestamp: The start timestamp (mm:ss)
    - duration: Duration in seconds
    
    Transcript:
    {text_with_timestamps}
    
    Return only valid JSON array with the structure:
    [{{"original_text": "...", "timestamp": "...", "duration": ...}}]
    """
    
    try:
        response = client.models.generate_content(
            model='models/gemini-2.0-flash',
            contents=[types.Part(text=prompt)]
        )
        
        if hasattr(response, 'candidates') and response.candidates:
            parts = response.candidates[0].content.parts
            response_text = '\n'.join(part.text for part in parts if hasattr(part, 'text'))
        else:
            response_text = getattr(response, 'text', '')
        
        # Extract JSON from response
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        json_data = response_text[json_start:json_end]
        
        return json.loads(json_data)
    except Exception as e:
        print(f"⚠️ Address detection failed: {e}")
        return []

def transcribe_video(input_path):
    try:
        file_size = os.path.getsize(input_path) / (1024 * 1024)
        if file_size > 20:
            raise ValueError("Video file must be smaller than 20MB for direct upload")

        client = genai.Client(api_key=gemini_api_key)

        print("\nProcessing video for multilingual transcription...")
        video_bytes = open(input_path, 'rb').read()

        # First pass: Get detailed multilingual transcription
        response = client.models.generate_content(
            model='models/gemini-2.0-flash',
            contents=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                ),
                types.Part(text="""Transcribe this video into JSON with keys: timestamp, duration, text. 
                Include all spoken words in their original language (Hindi, Marathi, Telugu, English, etc.).
                Each entry should include:
                - timestamp (mm:ss)
                - duration (in seconds)
                - text (spoken words in original language)
                Return only valid JSON.""")
            ]
        )

        transcript_text = ''
        if hasattr(response, 'candidates') and response.candidates:
            parts = response.candidates[0].content.parts
            transcript_text = '\n'.join(part.text for part in parts if hasattr(part, 'text'))
        else:
            transcript_text = getattr(response, 'text', '')

        print("\n📜 Raw Transcription Response >>>")
        print(transcript_text[:500] + "...")  # Print first 500 chars to avoid flooding console
        print("📜 Raw Transcription Response End <<<\n")

        # Clean and parse JSON
        json_start = transcript_text.find("[")
        json_end = transcript_text.rfind("]") + 1
        json_data = transcript_text[json_start:json_end]

        try:
            parsed = json.loads(json_data)
        except Exception as e:
            print("❌ Failed to parse JSON:", e)
            return []

        # Second pass: Detect addresses in the transcription
        print("\n🔍 Detecting addresses in transcription...")
        address_segments = detect_addresses(transcript_text)
        
        word_timestamps = []
        full_transcript = ""

        for entry in parsed:
            try:
                start_sec = convert_timestamp_to_seconds(entry['timestamp'])
                duration = float(entry['duration'])
                text = entry['text']
                full_transcript += f"{text}\n"
                
                # Process phone numbers first
                phone_patterns = VULGAR_WORDS["phone_number"]
                for pattern in phone_patterns:
                    for match in re.finditer(pattern, text):
                        full_number = match.group()
                        match_start = match.start()
                        match_end = match.end()
                        
                        total_text_length = max(len(text), 1)
                        number_start_sec = start_sec + (match_start / total_text_length) * duration
                        number_end_sec = start_sec + (match_end / total_text_length) * duration
                        
                        if not any(abs(ts[0] - number_start_sec) < 0.01 for ts in word_timestamps):
                            word_timestamps.append((number_start_sec, number_end_sec, "phone_number"))
                            print(f"⚠️ Phone number detected: '{full_number}' at {number_start_sec:.2f}s - {number_end_sec:.2f}s")

                # Process vulgar words
                words = text.split()
                per_word_dur = duration / len(words) if words else 0
                for i, word in enumerate(words):
                    clean_word = word.strip().lower().strip(".,!?\"'")
                    for base, variants in VULGAR_WORDS.items():
                        if base != "phone_number" and clean_word in variants:
                            word_start = start_sec + i * per_word_dur
                            word_end = word_start + per_word_dur
                            word_timestamps.append((word_start, word_end, base))
                            print(f"⚠️ Vulgar word detected: '{clean_word}' ({base}) at {word_start:.2f}s - {word_end:.2f}s")
                            
            except Exception as e:
                print(f"⚠️ Skipped malformed entry: {entry} due to {e}")

        # Add detected addresses to word_timestamps
        for addr in address_segments:
            try:
                start_sec = convert_timestamp_to_seconds(addr['timestamp'])
                duration = float(addr['duration'])
                word_timestamps.append((start_sec, start_sec + duration, "address"))
                print(f"⚠️ Address detected: '{addr['original_text']}' at {start_sec:.2f}s - {start_sec + duration:.2f}s")
            except Exception as e:
                print(f"⚠️ Skipped malformed address entry: {addr} due to {e}")

        if word_timestamps:
            print("\n✅ Censorable content found:", word_timestamps)
        else:
            print("\n✅ No censorable content detected. Original video is clean.")

        return word_timestamps

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return []

def generate_beep(duration, frequency=1000, sample_rate=44100):
    """Generate a precise, attention-grabbing beep sound with sharp attack"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    envelope = np.ones_like(t)
    attack_samples = int(0.01 * sample_rate)
    decay_samples = int(0.05 * sample_rate)
    
    if len(t) > attack_samples + decay_samples:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
    
    beep_sound = (0.7 * np.sin(2 * np.pi * frequency * t) + 
                 (0.2 * np.sin(2 * np.pi * 2 * frequency * t)) + 
                 (0.1 * np.sin(2 * np.pi * 3 * frequency * t)))
    
    beep_sound *= envelope
    
    left_channel = beep_sound * 0.8
    right_channel = beep_sound * 0.6
    return np.column_stack((left_channel, right_channel))

def censor_video(input_path, word_timestamps, output_path):
    """Enhanced video censoring with precise beep placement"""
    video = VideoFileClip(input_path)
    original_audio = video.audio
    
    if not word_timestamps:
        video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print("✅ No content to censor. Original video saved.")
        return
    
    audio_array = original_audio.to_soundarray()
    sample_rate = original_audio.fps
    num_channels = audio_array.shape[1] if len(audio_array.shape) > 1 else 1
    
    for start, end, word in word_timestamps:
        duration = end - start
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        
        beep_samples = generate_beep(duration, sample_rate=sample_rate)
        
        needed_samples = end_sample - start_sample
        if len(beep_samples) > needed_samples:
            beep_samples = beep_samples[:needed_samples]
        elif len(beep_samples) < needed_samples:
            padding = np.zeros((needed_samples - len(beep_samples), 2))
            beep_samples = np.vstack((beep_samples, padding))
        
        if num_channels == 1:
            beep_samples = np.mean(beep_samples, axis=1, keepdims=True)
        
        crossfade_samples = min(100, len(beep_samples) // 4)
        
        if crossfade_samples > 0:
            fade_in = np.linspace(0, 1, crossfade_samples)
            fade_out = np.linspace(1, 0, crossfade_samples)
            
            beep_samples[:crossfade_samples] *= fade_in[:, np.newaxis] if num_channels == 2 else fade_in.reshape(-1, 1)
            
            if start_sample + crossfade_samples <= len(audio_array):
                audio_array[start_sample:start_sample+crossfade_samples] *= fade_out[:, np.newaxis] if num_channels == 2 else fade_out.reshape(-1, 1)
            
            beep_samples[-crossfade_samples:] *= fade_out[:, np.newaxis] if num_channels == 2 else fade_out.reshape(-1, 1)
            
            if start_sample + len(beep_samples) <= len(audio_array):
                audio_array[start_sample+len(beep_samples)-crossfade_samples:start_sample+len(beep_samples)] *= fade_in[:, np.newaxis] if num_channels == 2 else fade_in.reshape(-1, 1)
        
        if start_sample + len(beep_samples) <= len(audio_array):
            audio_array[start_sample:start_sample+len(beep_samples)] = beep_samples
    
    censored_audio = AudioArrayClip(audio_array, fps=sample_rate)
    final_video = video.with_audio(censored_audio)
    
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=[
            '-crf', '18',
            '-preset', 'fast',
            '-movflags', '+faststart'
        ],
        threads=4,
        bitrate="5000k"
    )
    print(f"🎥 Successfully censored video saved to: {output_path}")

if __name__ == "__main__":
    check_ffmpeg()
    
    input_path = os.path.abspath("test4.mp4")
    fixed_path = os.path.abspath("fixed_input2.mp4")
    output_path = os.path.abspath("censored_output3.mp4")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    try:
        print("🔄 Pre-processing video for better transcription...")
        input_video = VideoFileClip(input_path)
        input_video.write_videofile(
            fixed_path,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=['-ar', '44100', '-ac', '2']
        )
        input_video.close()
        
        print("🔍 Detecting sensitive content...")
        censor_timestamps = transcribe_video(fixed_path)
        
        if not censor_timestamps:
            print("✅ No sensitive content detected. Original video is clean.")
        else:
            print(f"\n🔊 Found {len(censor_timestamps)} segments to censor:")
            for i, (start, end, word_type) in enumerate(censor_timestamps, 1):
                print(f"{i}. {word_type} at {start:.2f}s-{end:.2f}s")
            
            print("\n🎬 Creating censored version...")
            censor_video(fixed_path, censor_timestamps, output_path)
        
        if os.path.exists(fixed_path):
            os.remove(fixed_path)
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        sys.exit(1)