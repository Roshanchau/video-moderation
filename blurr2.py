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
            model='models/gemini-2.5-pro',
            contents=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                ),
                types.Part(text="""Analyze this video frame by frame to detect and locate any of the following sensitive information that appears visually on screen.
The analysis should cover all languages present in the video, including but not limited to English, Hindi, Telugu, and Marathi.

For each piece of sensitive information detected, you must provide the following details in a structured JSON format:

-   **exact_text**: The exact text of the sensitive information as it appears on the screen.
-   **content_type**: The category of the sensitive information. This must be one of the following:
    -   `phone_number`
    -   `email_address`
    -   `physical_address`
    -   `profanity_or_vulgarism`
    -   `credit_card_number`
    -   `personal_name` (if it seems to be part of sensitive contact info)
    -   `other_sensitive_info`
-   **timestamp**: The precise start time in seconds when the information first becomes visible.
-   **duration**: The total duration in seconds for which the information remains visible. Add a small buffer (e.g., 0.5 seconds) to ensure it's fully covered.
-   **position**: A highly accurate bounding box that tightly encloses the entire sensitive area. The coordinates must be relative to the video frame's dimensions (0-100%). The format is `{"x1": %, "y1": %, "x2": %, "y2": %}`.
    -   `x1`, `y1`: Top-left corner of the bounding box.
    -   `x2`, `y2`: Bottom-right corner of the bounding box.
    -   It is crucial that this box is precise. For multi-line text like an address, the box must cover all lines completely.
-   **confidence**: A confidence score from 0 to 1 indicating the certainty of the detection.

Return ONLY a valid JSON array containing an object for each detected piece of sensitive information. The structure must be:
`[{"exact_text": "...", "content_type": "...", "timestamp": ..., "duration": ..., "position": {...}, "confidence": ...}]`

If no sensitive content is found, return an empty array `[]`.""")
            ]
        )

        transcript_text = ''
        if hasattr(response, 'candidates') and response.candidates:
            parts = response.candidates[0].content.parts
            transcript_text = '\n'.join(part.text for part in parts if hasattr(part, 'text'))
        else:
            transcript_text = getattr(response, 'text', '')

        print("\n[INFO] Raw Analysis Response >>>")
        print(transcript_text[:500] + "...")  # Print first 500 chars
        print("[INFO] Raw Analysis Response End <<<\n")

        # Clean and parse JSON
        json_start = transcript_text.find("[")
        json_end = transcript_text.rfind("]") + 1
        json_data = transcript_text[json_start:json_end]

        try:
            parsed = json.loads(json_data)
        except Exception as e:
            print("[ERROR] Failed to parse JSON:", e)
            return []

        sensitive_segments = []
        cap = cv2.VideoCapture(video_path)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        if parsed:
            all_x1 = [int(p['position']['x1']/100 * frame_width) for p in parsed]
            all_y1 = [int(p['position']['y1']/100 * frame_height) for p in parsed]
            all_x2 = [int(p['position']['x2']/100 * frame_width) for p in parsed]
            all_y2 = [int(p['position']['y2']/100 * frame_height) for p in parsed]

            min_x1 = min(all_x1)
            min_y1 = min(all_y1)
            max_x2 = max(all_x2)
            max_y2 = max(all_y2)

            start_time = min(float(p['timestamp']) for p in parsed)
            end_time = max(float(p['timestamp']) + float(p['duration']) for p in parsed)

            # Ensure valid coordinates after consolidation
            min_x1, min_y1 = max(0, min_x1), max(0, min_y1)
            max_x2, max_y2 = min(frame_width, max_x2), min(frame_height, max_y2)
            
            # Swap coordinates if they are in the wrong order
            if min_x1 > max_x2:
                min_x1, max_x2 = max_x2, min_x1
            if min_y1 > max_y2:
                min_y1, max_y2 = max_y2, min_y1

            if min_x1 >= max_x2 or min_y1 >= max_y2:
                print(f"[WARNING] Invalid consolidated bounding box: {min_x1},{min_y1} - {max_x2},{max_y2}")
            else:
                sensitive_segments.append((start_time, end_time, 'sensitive_info', (min_x1, min_y1, max_x2, max_y2)))

        print(f"[INFO] Sensitive content found:", sensitive_segments)
        return sensitive_segments

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        return []


def blur_region(frame: np.ndarray, bbox: Tuple[int, int, int, int], blur_strength: int = 151) -> np.ndarray:
    """Blur a specific region in a frame with validation and debug visualization"""
    # Make a copy to avoid modifying original
    frame = frame.copy()
    x1, y1, x2, y2 = bbox
    
    # Debug visualization before blurring (red rectangle)
    debug_frame = frame.copy()
    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('Debug - Region to Blur', debug_frame)
    cv2.waitKey(1)  # Briefly show the region
    
    # Ensure coordinates are valid and region has positive size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
    
    # Ensure region has positive dimensions
    if x2 <= x1 or y2 <= y1:
        print(f"[WARNING] Invalid region: {bbox} in frame of size {frame.shape}")
        return frame
    
    # Ensure blur strength is odd and positive (larger for visibility)
    blur_strength = max(31, blur_strength | 1)  # Forces odd number \u226531
    
    try:
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            print("[WARNING] Empty region - skipping blur")
            return frame
            
        blurred = cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)
        frame[y1:y2, x1:x2] = blurred
        
        # Debug visualization after blurring
        cv2.imshow('Debug - After Blur', frame[y1:y2, x1:x2])
        cv2.waitKey(1)
        
    except Exception as e:
        print(f"[WARNING] Blur failed for {bbox}: {str(e)}")
    
    return frame

def consolidate_bounding_boxes(segments: List[Tuple[float, float, str, Tuple[int, int, int, int]]]) -> List[Tuple[float, float, str, Tuple[int, int, int, int]]]:
    """Consolidate overlapping or nearby bounding boxes."""
    if not segments:
        return []

    # Sort segments by start time
    segments.sort(key=lambda x: x[0])

    consolidated = []
    current_segment = list(segments[0])

    for next_segment in segments[1:]:
        # If the next segment overlaps or is close in time
        if next_segment[0] <= current_segment[1]:
            # Merge the time intervals
            current_segment[1] = max(current_segment[1], next_segment[1])

            # Merge the bounding boxes
            x1_curr, y1_curr, x2_curr, y2_curr = current_segment[3]
            x1_next, y1_next, x2_next, y2_next = next_segment[3]
            new_bbox = (
                min(x1_curr, x1_next),
                min(y1_curr, y1_next),
                max(x2_curr, x2_next),
                max(y2_curr, y2_next)
            )
            current_segment[3] = new_bbox
        else:
            consolidated.append(tuple(current_segment))
            current_segment = list(next_segment)

    consolidated.append(tuple(current_segment))
    return consolidated

def process_video(input_path: str, output_path: str):
    """Main processing function with enhanced debugging"""
    sensitive_segments = detect_sensitive_content(input_path)

    # Expand the consolidated bounding boxes by a fixed pixel amount
    expanded_segments = []
    PIXEL_EXPANSION = 100  # Expand by 100 pixels in each direction
    for start, end, content_type, bbox in sensitive_segments:
        x1, y1, x2, y2 = bbox
        x1 = int(x1 - PIXEL_EXPANSION)
        y1 = int(y1 - PIXEL_EXPANSION)
        x2 = int(x2 + PIXEL_EXPANSION)
        y2 = int(y2 + PIXEL_EXPANSION)
        expanded_segments.append((start, end, content_type, (x1, y1, x2, y2)))
    sensitive_segments = expanded_segments
    
    if not sensitive_segments:
        print("No sensitive content found - copying original video")
        shutil.copy(input_path, output_path)
        return

    print(f"\n[INFO] Found {len(sensitive_segments)} segments to blur:")
    for i, (start, end, content_type, bbox) in enumerate(sensitive_segments):
        print(f"{i+1}. {content_type} @ {start:.1f}-{end:.1f}s: {bbox}")
        # Visualize the bounding box
        print(f"   Bounding box dimensions: width={bbox[2]-bbox[0]}, height={bbox[3]-bbox[1]}")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a window for debugging
    cv2.namedWindow('Processing Debug', cv2.WINDOW_NORMAL)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp_blurred.mp4', fourcc, fps, (width, height))
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = current_frame / fps
        
        # Apply all relevant blurring for this frame
        frame_debug = frame.copy()
        TIME_BUFFER = 0.5  # Add a 0.5-second buffer to catch frames more reliably
        for start, end, content_type, bbox in sensitive_segments:
            if (start - TIME_BUFFER) <= current_time <= (end + TIME_BUFFER):
                # Draw bounding box on debug frame
                cv2.rectangle(frame_debug, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                frame = blur_region(frame, bbox)
                # Draw bounding box on the final frame
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        
        # Show processing debug
        cv2.imshow('Processing Debug', frame_debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        out.write(frame)
        current_frame += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\n[SUCCESS] Blurring complete - finalizing video...")
    
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
        print(f"[ERROR] Final composition failed: {e}")
        shutil.copy('temp_blurred2.mp4', output_path)
    
    os.remove('temp_blurred.mp4')
    print(f"[SUCCESS] Processing complete! Saved to {output_path}")

if __name__ == "__main__":
    input_video = "censored_output3.mp4"
    output_video = "final_output_blurred2.mp4"
    
    if not os.path.exists(input_video):
        print(f"[ERROR] Error: Input file {input_video} not found!")
    else:
        process_video(input_video, output_video)
