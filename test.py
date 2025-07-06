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

gemini_api_key= os.getenv("GOOGLE_API_KEY")

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
    "fucking":["fucking"],
    "fuck": ["fuck", "fucking", "fucker", "motherfucker"],
    "shit": ["shit", "bullshit", "shitty"],
    "asshole": ["asshole", "arsehole"],
    "bitch": ["bitch", "bitches", "bitching"],
    "damn": ["damn", "goddamn"],
    "crap": ["crap", "crappy"],
    "phone_number": [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # 205-234-1333 or 205.234.1333 or 205 234 1333
        r'\b\d{3}\s\d{3}\s\d{4}\b',  # 205 234 1333
        r'\b\d{10}\b',  # 2052341333
        r'\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b'  # (205) 234-1333
    ]
}


def convert_timestamp_to_seconds(timestamp_str):
    parts = list(map(float, timestamp_str.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0

def transcribe_video(input_path):
    try:
        file_size = os.path.getsize(input_path) / (1024 * 1024)
        if file_size > 20:
            raise ValueError("Video file must be smaller than 20MB for direct upload")

        client = genai.Client(api_key=gemini_api_key)

        print("\nProcessing video for speech transcription...")
        video_bytes = open(input_path, 'rb').read()

        response = client.models.generate_content(
            model='models/gemini-2.0-flash',
            contents=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                ),
                types.Part(text="Transcribe this video into JSON with keys: timestamp, duration, words. Each entry should include timestamp (mm:ss), duration (in seconds), and spoken words (string).")
            ]
        )

        transcript_text = ''
        if hasattr(response, 'candidates') and response.candidates:
            parts = response.candidates[0].content.parts
            transcript_text = '\n'.join(part.text for part in parts if hasattr(part, 'text'))
        else:
            transcript_text = getattr(response, 'text', '')

        print("\n📜 Raw Gemini Response >>>")
        print(transcript_text)
        print("📜 Raw Gemini Response End <<<\n")

        # Clean and parse JSON
        json_start = transcript_text.find("[")
        json_end = transcript_text.rfind("]") + 1
        json_data = transcript_text[json_start:json_end]

        try:
            parsed = json.loads(json_data)
        except Exception as e:
            print("❌ Failed to parse JSON:", e)
            return []

        word_timestamps = []

        for entry in parsed:
            try:
                start_sec = convert_timestamp_to_seconds(entry['timestamp'])
                duration = float(entry['duration'])
                text = entry['words']
                words = text.split()

                # Enhanced phone number detection
                phone_patterns = [
                    r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # 205-859-2113 or 205 859 2113
                    r'\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b'   # (205) 859-2113
                ]

                # Check all phone patterns against the full text
                for pattern in phone_patterns:
                    for match in re.finditer(pattern, text):
                        full_number = match.group()
                        match_start = match.start()
                        match_end = match.end()
                        
                        # Calculate exact timing for the entire phone number
                        total_text_length = max(len(text), 1)  # Avoid division by zero
                        number_start_sec = start_sec + (match_start / total_text_length) * duration
                        number_end_sec = start_sec + (match_end / total_text_length) * duration
                        
                        # Only add if we haven't already detected this number
                        if not any(abs(ts[0] - number_start_sec) < 0.01 for ts in word_timestamps):
                            word_timestamps.append((number_start_sec, number_end_sec, "phone_number"))
                            print(f"⚠️ Phone number detected: '{full_number}' at {number_start_sec:.2f}s - {number_end_sec:.2f}s")

                # Vulgar word detection
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
    
    # Create envelope for sharp attack and quick decay
    envelope = np.ones_like(t)
    attack_samples = int(0.01 * sample_rate)  # 10ms attack
    decay_samples = int(0.05 * sample_rate)   # 50ms decay
    
    if len(t) > attack_samples + decay_samples:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
    
    # Create beep with multiple harmonics for better audibility
    beep_sound = (0.7 * np.sin(2 * np.pi * frequency * t) + \
                 (0.2 * np.sin(2 * np.pi * 2 * frequency * t)) + \
                 (0.1 * np.sin(2 * np.pi * 3 * frequency * t)))
    
    # Apply envelope
    beep_sound *= envelope
    
    # Convert to stereo with slight panning for better spatial awareness
    left_channel = beep_sound * 0.8
    right_channel = beep_sound * 0.6
    return np.column_stack((left_channel, right_channel))

def censor_video(input_path, word_timestamps, output_path):
    """Enhanced video censoring with precise beep placement"""
    video = VideoFileClip(input_path)
    original_audio = video.audio
    
    if not word_timestamps:
        video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print("✅ No vulgar words detected. Original video saved.")
        return
    
    # Create a new audio array with beeps
    audio_array = original_audio.to_soundarray()
    sample_rate = original_audio.fps
    
    # Get number of channels (1 for mono, 2 for stereo)
    num_channels = audio_array.shape[1] if len(audio_array.shape) > 1 else 1
    
    
    for start, end, word in word_timestamps:
        duration = end - start
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        
        # Generate precise beep for this segment
        beep_samples = generate_beep(duration, sample_rate=sample_rate)
        
        # Ensure perfect length matching
        needed_samples = end_sample - start_sample
        if len(beep_samples) > needed_samples:
            beep_samples = beep_samples[:needed_samples]
        elif len(beep_samples) < needed_samples:
            # Pad with silence if needed (shouldn't happen with our generation)
            padding = np.zeros((needed_samples - len(beep_samples), 2))
            beep_samples = np.vstack((beep_samples, padding))
        
        # Match channels
        if num_channels == 1:
            beep_samples = np.mean(beep_samples, axis=1, keepdims=True)
        
        # Apply crossfade to avoid clicks
        crossfade_samples = min(100, len(beep_samples) // 4)  # ~2.3ms at 44100Hz
        
        if crossfade_samples > 0:
            # Create fade curves
            fade_in = np.linspace(0, 1, crossfade_samples)
            fade_out = np.linspace(1, 0, crossfade_samples)
            
            # Apply fade-in to beep
            beep_samples[:crossfade_samples] *= fade_in[:, np.newaxis] if num_channels == 2 else fade_in.reshape(-1, 1)
            
            # Apply fade-out to original audio
            if start_sample + crossfade_samples <= len(audio_array):
                audio_array[start_sample:start_sample+crossfade_samples] *= fade_out[:, np.newaxis] if num_channels == 2 else fade_out.reshape(-1, 1)
            
            # Apply fade-out to beep
            beep_samples[-crossfade_samples:] *= fade_out[:, np.newaxis] if num_channels == 2 else fade_out.reshape(-1, 1)
            
            # Apply fade-in to original audio after beep
            if start_sample + len(beep_samples) <= len(audio_array):
                audio_array[start_sample+len(beep_samples)-crossfade_samples:start_sample+len(beep_samples)] *= fade_in[:, np.newaxis] if num_channels == 2 else fade_in.reshape(-1, 1)
        
        # Replace the segment with beep
        if start_sample + len(beep_samples) <= len(audio_array):
            audio_array[start_sample:start_sample+len(beep_samples)] = beep_samples
    
    # Create new audio clip with higher quality settings
    censored_audio = AudioArrayClip(audio_array, fps=sample_rate)
    
    # Combine with video
    final_video = video.with_audio(censored_audio)
    
    # Write with optimized settings
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=[
            '-crf', '18',        # Better quality
            '-preset', 'fast',   # Good balance between speed and compression
            '-movflags', '+faststart'  # For web streaming
        ],
        threads=4,
        bitrate="5000k"
    )
    print(f"🎥 Successfully censored video saved to: {output_path}")

if __name__ == "__main__":
    check_ffmpeg()
    
    input_path = os.path.abspath("test3.mp4")
    fixed_path = os.path.abspath("fixed_input.mp4")
    output_path = os.path.abspath("censored_output3.mp4")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    try:
        # Pre-process video for better transcription
        print("🔄 Pre-processing video for better transcription...")
        input_video = VideoFileClip(input_path)
        input_video.write_videofile(
            fixed_path,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=['-ar', '44100', '-ac', '2']  # Standardize audio
        )
        input_video.close()
        
        # Detect vulgar words with enhanced accuracy
        print("🔍 Detecting vulgar words...")
        vulgar_word_timestamps = transcribe_video(fixed_path)
        
        if not vulgar_word_timestamps:
            print("✅ No vulgar words detected. Original video is clean.")
        else:
            print(f"\n🔊 Found {len(vulgar_word_timestamps)} segments to censor:")
            for i, (start, end, word) in enumerate(vulgar_word_timestamps, 1):
                print(f"{i}. {word} at {start:.2f}s-{end:.2f}s")
            
            # Generate censored video
            print("\n🎬 Creating censored version...")
            censor_video(fixed_path, vulgar_word_timestamps, output_path)
        
        # Clean up temporary file
        if os.path.exists(fixed_path):
            os.remove(fixed_path)
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        sys.exit(1)