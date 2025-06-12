from moviepy import *
import numpy as np
import os
import json
import subprocess
import sys
import re
import google.generativeai as genai
from google.generativeai import types

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

        client = genai.Client(api_key='AIzaSyCxGVXk_qmDB4PQBtKkfdqaS7qPL5e1t1M')

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

        print("\nRaw Gemini Response >>>")
        print(transcript_text)
        print("Raw Gemini Response End <<<\n")

        json_start = transcript_text.find("[")
        json_end = transcript_text.rfind("]") + 1
        json_data = transcript_text[json_start:json_end]

        try:
            parsed = json.loads(json_data)
        except Exception as e:
            print("Failed to parse JSON:", e)
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
                            print(f"Vulgar word detected: '{clean_word}' ({base}) at {word_start:.2f}s - {word_end:.2f}s")
            except Exception as e:
                print(f"Skipped malformed entry: {entry} due to {e}")

        if word_timestamps:
            print("\nVulgar words found:", word_timestamps)
        else:
            print("\nNo vulgar words detected. Original video is clean.")

        return word_timestamps

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def generate_beep(duration, frequency=1000, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    envelope = np.ones_like(t)
    attack_samples = int(0.01 * sample_rate)
    decay_samples = int(0.05 * sample_rate)
    
    if len(t) > attack_samples + decay_samples:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
    
    beep_sound = (0.7 * np.sin(2 * np.pi * frequency * t) +
                 0.2 * np.sin(2 * np.pi * 2 * frequency * t) +
                 0.1 * np.sin(2 * np.pi * 3 * frequency * t))
    
    beep_sound *= envelope
    
    left_channel = beep_sound * 0.8
    right_channel = beep_sound * 0.6
    return np.column_stack((left_channel, right_channel))

def censor_video(input_path, word_timestamps, output_path):
    video = VideoFileClip(input_path)
    original_audio = video.audio
    
    if not word_timestamps:
        video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print("No vulgar words detected. Original video saved.")
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
    print(f"Successfully censored video saved to: {output_path}")

if __name__ == "__main__":
    check_ffmpeg()
    
    input_path = os.path.abspath("test.mp4")
    fixed_path = os.path.abspath("fixed_input.mp4")
    output_path = os.path.abspath("censored_output.mp4")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    try:
        print("Pre-processing video for better transcription...")
        input_video = VideoFileClip(input_path)
        input_video.write_videofile(
            fixed_path,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=['-ar', '44100', '-ac', '2']
        )
        input_video.close()
        
        print("Detecting vulgar words...")
        vulgar_word_timestamps = transcribe_video(fixed_path)
        
        if not vulgar_word_timestamps:
            print("No vulgar words detected. Original video is clean.")
        else:
            print(f"\nFound {len(vulgar_word_timestamps)} segments to censor:")
            for i, (start, end, word) in enumerate(vulgar_word_timestamps, 1):
                print(f"{i}. {word} at {start:.2f}s-{end:.2f}s")
            print("\nCreating censored version...")
            censor_video(fixed_path, vulgar_word_timestamps, output_path)
        
        if os.path.exists(fixed_path):
            os.remove(fixed_path)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
