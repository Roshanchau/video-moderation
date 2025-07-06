import cv2
import numpy as np
import os
import json
import re
import google.genai as genai
from google.genai import types
from moviepy import *
from dotenv import load_dotenv
import shutil
from typing import List, Tuple

load_dotenv()

# Initialize Gemini Client
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Sensitive content patterns
SENSITIVE_PATTERNS = {
    "phone_number": [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        r'\b\d{3}\s\d{3}\s\d{4}\b',
        r'\b\d{10}\b',
        r'\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b'
    ],
    "profanity": [
        r'\bfuck\w*\b',
        r'\bshit\w*\b',
        r'\basshole\b',
        r'\bbitch\w*\b',
        r'\bdamn\b',
        r'\bcrap\b'
    ]
}

def convert_timestamp_to_seconds(timestamp: str) -> float:
    """Convert mm:ss timestamp to seconds"""
    if ':' in timestamp:
        minutes, seconds = timestamp.split(':')
        return float(minutes) * 60 + float(seconds)
    return float(timestamp)

def detect_sensitive_content(video_path: str) -> List[Tuple[float, float, str, Tuple[int, int, int, int]]]:
    """Detect sensitive content in video frames using Gemini"""
    print("Analyzing video for sensitive content...")
    
    try:
        # file_size = os.path.getsize(video_path) / (1024 * 1024)
        # if file_size > 20:
        #     raise ValueError("Video file must be smaller than 20MB for direct upload")

        client = genai.Client(api_key=GEMINI_API_KEY)

        print("\nProcessing video for sensitive content detection...")
        video_bytes = open(video_path, 'rb').read()

        response = client.models.generate_content(
            model='models/gemini-2.0-flash',
            contents=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                ),
                types.Part(text="""Analyze this video frame and the corresponding transcript to detect any addresses , phone numbers or sensitive information that appears on screen
                           in different languages including (Hindi, Telugu, Marathi, etc.).
        
        For any detected sensitive information (addresses, phone numbers, vulgar or profanity words etc.), provide:
        - exact_text: The text content visible on screen
        - content_type: Type of sensitive content (phone_number, address, profanity, etc.)
        - timestamp: Estimated time in seconds when this appears
        - duration: How long it remains visible + 3 to 5 ms for error handling(precision) (in seconds)
        - position: Precise position as {"x1": %, "y1": %, "x2": %, "y2": %} relative coordinates (0-100%)
        - confidence: Confidence score (0-1)
        
        Return only valid JSON array with the structure:
        [{"exact_text": "...", "content_type": "...", "timestamp": ..., "duration": ..., "position": {...}, "confidence": ...}]""")
            ]
        )

        transcript_text = ''
        if hasattr(response, 'candidates') and response.candidates:
            parts = response.candidates[0].content.parts
            transcript_text = '\n'.join(part.text for part in parts if hasattr(part, 'text'))
        else:
            transcript_text = getattr(response, 'text', '')

        print("\n📜 Raw Analysis Response >>>")
        print(transcript_text[:500] + "...")  # Print first 500 chars
        print("📜 Raw Analysis Response End <<<\n")

        # Clean and parse JSON
        json_start = transcript_text.find("[")
        json_end = transcript_text.rfind("]") + 1
        json_data = transcript_text[json_start:json_end]

        try:
            parsed = json.loads(json_data)
        except Exception as e:
            print("❌ Failed to parse JSON:", e)
            return []

        sensitive_segments = []
        cap = cv2.VideoCapture(video_path)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        for entry in parsed:
            try:
                start_sec = float(entry['timestamp'])  # Already in seconds
                duration = float(entry['duration'])
                content_type = entry.get('content_type', 'sensitive_info')

                # Convert normalized bbox to pixel coordinates
                pos = entry['position']
                x1 = int(pos['x1']/100 * frame_width)
                y1 = int(pos['y1']/100 * frame_height)
                x2 = int(pos['x2']/100 * frame_width)
                y2 = int(pos['y2']/100 * frame_height)

                # Ensure valid coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                
                if x1 >= x2 or y1 >= y2:
                    print(f"⚠️ Invalid bounding box: {x1},{y1} - {x2},{y2}")
                    continue

                sensitive_segments.append((start_sec, start_sec + duration, content_type, (x1, y1, x2, y2)))
                print(f"⚠️ {content_type} detected at {start_sec:.2f}s - {start_sec+duration:.2f}s: {x1},{y1} to {x2},{y2}")
        
            except Exception as e:
                print(f"⚠️ Skipped malformed entry: {entry} due to {e}")

        if sensitive_segments:
            print("\n✅ Sensitive content found:", sensitive_segments)
        else:
            print("\n✅ No sensitive content detected. Original video is clean.")

        return sensitive_segments

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return []

def blur_region(frame: np.ndarray, bbox: Tuple[int, int, int, int], blur_strength: int = 31) -> np.ndarray:
    """Blur a specific region in a frame with validation"""
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are valid and region has positive size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    
    # Ensure region has positive dimensions
    if x2 <= x1 or y2 <= y1:
        print(f"⚠️ Invalid region: {bbox} in frame of size {frame.shape}")
        return frame
    
    # Ensure blur strength is odd and positive
    blur_strength = max(3, blur_strength | 1)  # Forces odd number ≥3
    
    try:
        region = frame[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)
        frame[y1:y2, x1:x2] = blurred
    except Exception as e:
        print(f"⚠️ Blur failed for {bbox}: {str(e)}")
    
    return frame

def process_video(input_path: str, output_path: str):
    """Main processing function with enhanced debugging"""
    sensitive_segments = detect_sensitive_content(input_path)
    
    if not sensitive_segments:
        print("No sensitive content found - copying original video")
        shutil.copy(input_path, output_path)
        return

    print(f"\n🔍 Found {len(sensitive_segments)} segments to blur:")
    for i, (start, end, content_type, bbox) in enumerate(sensitive_segments):
        print(f"{i+1}. {content_type} @ {start:.1f}-{end:.1f}s: {bbox}")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp_blurred.mp4', fourcc, fps, (width, height))
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = current_frame / fps
        
        # Debug print every 5 seconds
        if current_frame % int(fps*5) == 0:
            print(f"\n⏱️ Processing frame {current_frame} @ {current_time:.1f}s")
        
        # Apply all relevant blurring for this frame
        for start, end, content_type, bbox in sensitive_segments:
            if start <= current_time <= end:
                if current_frame % int(fps) == 0:  # Print once per second per segment
                    print(f"  • Blurring {content_type} @ {current_time:.1f}s: {bbox}")
                frame = blur_region(frame, bbox)
        
        out.write(frame)
        current_frame += 1
    
    cap.release()
    out.release()
    print("\n✅ Blurring complete - finalizing video...")
    
    # Combine with original audio
    try:
        original_clip = VideoFileClip(input_path)
        video_clip = VideoFileClip('temp_blurred.mp4')
        
        final_clip = video_clip.with_audio(original_clip.audio)
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            threads=4,
            preset='fast'
        )
        
        original_clip.close()
        video_clip.close()
    except Exception as e:
        print(f"❌ Final composition failed: {e}")
        shutil.copy('temp_blurred.mp4', output_path)
    
    os.remove('temp_blurred.mp4')
    print(f"✅ Processing complete! Saved to {output_path}")

if __name__ == "__main__":
    input_video = "censored_output3.mp4"
    output_video = "final_output_blurred.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Error: Input file {input_video} not found!")
    else:
        process_video(input_video, output_video)