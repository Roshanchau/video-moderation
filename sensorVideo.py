from moviepy import *
import numpy as np
import os
import json
import subprocess
import sys
import re
import google.genai as genai
from google.genai import types


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
    "fuck": ["fuck", "fucking", "fucker", "motherfucker"],
    "shit": ["shit", "bullshit", "shitty"],
    "asshole": ["asshole", "arsehole"],
    "bitch": ["bitch", "bitches", "bitching"],
    "damn": ["damn", "goddamn"],
    "crap": ["crap", "crappy"]
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

        client = genai.Client(api_key='AIzaSyC7RPvt2RM7rCGsRuB7dHe8u2kXVVCbmE8')

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
                words = entry['words'].split()

                per_word_dur = duration / len(words) if words else 0

                for i, word in enumerate(words):
                    clean_word = word.strip().lower().strip(".,!?\"'")
                    for base, variants in VULGAR_WORDS.items():
                        if clean_word in variants:
                            word_start = start_sec + i * per_word_dur
                            word_end = word_start + per_word_dur
                            word_timestamps.append((word_start, word_end, base))
                            print(f"⚠️ Vulgar word detected: '{clean_word}' ({base}) at {word_start:.2f}s - {word_end:.2f}s")
            except Exception as e:
                print(f"⚠️ Skipped malformed entry: {entry} due to {e}")

        if word_timestamps:
            print("\n✅ Vulgar words found:", word_timestamps)
        else:
            print("\n✅ No vulgar words detected. Original video is clean.")

        return word_timestamps

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return []



def generate_beep(duration, frequency=1000, sample_rate=44100):
    """Generate a more noticeable beep sound"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a more attention-grabbing beep with harmonics
    beep_sound = (0.6 * np.sin(2 * np.pi * frequency * t) + 
                  0.3 * np.sin(4 * np.pi * frequency * t) + 
                  0.1 * np.sin(6 * np.pi * frequency * t))
    return np.column_stack((beep_sound, beep_sound))  # Stereo format

def censor_video(input_path, word_timestamps, output_path):
    """Censor video directly using moviepy"""
    video = VideoFileClip(input_path)
    original_audio = video.audio
    
    if not word_timestamps:
        video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print("✅ No vulgar words detected. Original video saved.")
        return
    
    # Create a new audio array with beeps
    audio_array = original_audio.to_soundarray()
    sample_rate = original_audio.fps
    
    for start, end in word_timestamps:
        duration = end - start
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        
        # Generate beep for this segment
        beep_samples = generate_beep(duration, sample_rate=sample_rate)
        
        # Ensure proper shape (stereo)
        if len(audio_array.shape) == 1:
            beep_samples = np.column_stack((beep_samples, beep_samples))
        else:
            beep_samples = np.column_stack((beep_samples, beep_samples))
        
        # Calculate the actual number of samples we can replace
        replace_length = min(len(beep_samples), end_sample - start_sample)
        
        # Replace the segment with beep
        if start_sample + replace_length <= len(audio_array):
            audio_array[start_sample:start_sample+replace_length] = beep_samples[:replace_length]
    
    # Create new audio clip
    censored_audio = AudioArrayClip(audio_array, fps=sample_rate)
    
    # Combine with video - This is the corrected line
    final_video = video.with_audio(censored_audio)
    
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=['-crf', '18'],  # Higher quality
        threads=4  # Use multiple threads for faster processing
    )
    print(f"🎥 Censored video saved to: {output_path}")

if __name__ == "__main__":
    check_ffmpeg()
    
    input_path = os.path.abspath("test.mp4")
    fixed_path = os.path.abspath("fixed_input.mp4")
    output_path = os.path.abspath("censored_output.mp4")
    
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