"""
Complete Video Profanity Censoring System - Single File Solution
Save this as: video_censor_complete.py
Fixed for Windows console encoding and MoviePy compatibility
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
from google.cloud import videointelligence
from moviepy import *

# Fix for Windows console encoding
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_censoring.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CensorSegment:
    """Data class for storing censoring segments"""
    start_time: float
    end_time: float
    word: str
    confidence: float = 0.0

class VideoCensoringError(Exception):
    """Custom exception for video censoring errors"""
    pass

class VideoProfileCensor:
    """
    A comprehensive video profanity censoring system that:
    1. Transcribes video using Google Cloud Video Intelligence
    2. Identifies profane words with timestamps
    3. Replaces profane audio segments with beep sounds
    4. Maintains video quality and synchronization
    """
    
    # Comprehensive profanity list (can be loaded from external file)
    DEFAULT_PROFANITY_LIST = [
        "shit", "fuck", "fucking", "fucked", "fucker", "asshole", "bitch", 
        "damn", "crap", "piss", "bastard", "hell", "bloody", "goddamn"
    ]
    
    def __init__(self, 
                 profanity_list: Optional[List[str]] = None,
                 beep_frequency: int = 1000,
                 beep_volume: float = 0.3,
                 confidence_threshold: float = 0.8):
        """
        Initialize the video censoring system
        
        Args:
            profanity_list: Custom list of words to censor
            beep_frequency: Frequency of beep sound in Hz
            beep_volume: Volume of beep sound (0.0 to 1.0)
            confidence_threshold: Minimum confidence for word detection
        """
        self.profanity_list = [word.lower() for word in (profanity_list or self.DEFAULT_PROFANITY_LIST)]
        self.beep_frequency = beep_frequency
        self.beep_volume = beep_volume
        self.confidence_threshold = confidence_threshold
        
        # Verify dependencies
        self._verify_dependencies()
        
    def _verify_dependencies(self) -> None:
        """Verify all required dependencies are available"""
        try:
            # Check FFmpeg
            subprocess.run(['ffmpeg', '-version'], check=True, 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("✓ FFmpeg verified")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise VideoCensoringError(
                "FFmpeg is required but not installed. Please install FFmpeg:\n"
                "Windows: choco install ffmpeg\n"
                "Mac: brew install ffmpeg\n"
                "Linux: sudo apt-get install ffmpeg"
            )
        
        # Check Google Cloud credentials
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            logger.warning("⚠ GOOGLE_APPLICATION_CREDENTIALS not set. Ensure you have proper authentication.")
    
    @contextmanager
    def _safe_video_handling(self, video_path: str):
        """Context manager for safe video file handling"""
        video = None
        try:
            video = VideoFileClip(video_path)
            yield video
        finally:
            if video:
                video.close()
    
    def _preprocess_video(self, input_path: str, temp_path: str) -> None:
        """
        Preprocess video to ensure compatibility with Google Cloud API
        
        Args:
            input_path: Path to original video
            temp_path: Path for preprocessed video
        """
        logger.info(f"Preprocessing video: {input_path}")
        
        with self._safe_video_handling(input_path) as video:
            # Re-encode with standard settings for better API compatibility
            video.write_videofile(
                temp_path,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
                ffmpeg_params=['-crf', '23'],  # Balanced quality/size
                logger=None
            )
        
        logger.info(f"✓ Video preprocessed and saved to: {temp_path}")
    
    def transcribe_and_detect_profanity(self, video_path: str) -> List[CensorSegment]:
        """
        Transcribe video and detect profanity with timestamps
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of CensorSegment objects containing profanity timestamps
        """
        logger.info("Starting video transcription...")
        
        try:
            # Initialize Google Cloud Video Intelligence client
            client = videointelligence.VideoIntelligenceServiceClient()
            
            # Configure speech transcription
            speech_config = videointelligence.SpeechTranscriptionConfig(
                language_code="en-US",
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                enable_speaker_diarization=False,  # Can be enabled if needed
                audio_tracks=[0]  # Process first audio track
            )
            
            video_context = videointelligence.VideoContext(
                speech_transcription_config=speech_config
            )
            
            # Read video file
            with open(video_path, 'rb') as video_file:
                input_content = video_file.read()
            
            # Submit transcription request
            operation = client.annotate_video(
                request={
                    "features": [videointelligence.Feature.SPEECH_TRANSCRIPTION],
                    "input_content": input_content,
                    "video_context": video_context,
                }
            )
            
            logger.info("⏳ Processing video transcription (this may take several minutes)...")
            result = operation.result(timeout=900)  # 15 minutes timeout
            
            # Process results
            censor_segments = []
            annotation_results = result.annotation_results[0]
            
            for speech_transcription in annotation_results.speech_transcriptions:
                for alternative in speech_transcription.alternatives:
                    transcript = alternative.transcript
                    logger.info(f"Transcript confidence: {alternative.confidence:.2f}")
                    logger.info(f"Full transcript: {transcript}")
                    
                    for word_info in alternative.words:
                        word = word_info.word.lower().strip('.,!?";:')
                        confidence = word_info.confidence if hasattr(word_info, 'confidence') else 1.0
                        
                        if word in self.profanity_list and confidence >= self.confidence_threshold:
                            start_time = (word_info.start_time.seconds + 
                                        word_info.start_time.microseconds * 1e-6)
                            end_time = (word_info.end_time.seconds + 
                                      word_info.end_time.microseconds * 1e-6)
                            
                            segment = CensorSegment(
                                start_time=start_time,
                                end_time=end_time,
                                word=word,
                                confidence=confidence
                            )
                            censor_segments.append(segment)
                            
                            logger.warning(f"X Profanity detected: '{word}' "
                                         f"({start_time:.2f}s-{end_time:.2f}s) "
                                         f"confidence: {confidence:.2f}")
            
            # Merge overlapping segments
            censor_segments = self._merge_overlapping_segments(censor_segments)
            
            logger.info(f"✓ Transcription complete. Found {len(censor_segments)} segments to censor")
            return censor_segments
            
        except Exception as e:
            logger.error(f"X Transcription failed: {str(e)}")
            raise VideoCensoringError(f"Transcription failed: {str(e)}")
    
    def _merge_overlapping_segments(self, segments: List[CensorSegment]) -> List[CensorSegment]:
        """Merge overlapping or adjacent censoring segments"""
        if not segments:
            return segments
        
        # Sort by start time
        segments.sort(key=lambda x: x.start_time)
        merged = [segments[0]]
        
        for current in segments[1:]:
            last = merged[-1]
            
            # Merge if segments overlap or are very close (within 0.1 seconds)
            if current.start_time <= last.end_time + 0.1:
                last.end_time = max(last.end_time, current.end_time)
                last.word += f" + {current.word}"
            else:
                merged.append(current)
        
        return merged
    
    def _generate_beep_audio(self, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a beep sound with specified duration
        
        Args:
            duration: Duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Stereo beep audio array
        """
        if duration <= 0:
            return np.array([]).reshape(0, 2)
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        beep_mono = np.sin(2 * np.pi * self.beep_frequency * t) * self.beep_volume
        
        # Create stereo beep
        return np.column_stack((beep_mono, beep_mono))
    
    def censor_video(self, input_path: str, censor_segments: List[CensorSegment], 
                    output_path: str) -> None:
        """
        Apply censoring to video by replacing profanity with beeps
        
        Args:
            input_path: Path to input video
            censor_segments: List of segments to censor
            output_path: Path for output video
        """
        logger.info(f"Starting video censoring with {len(censor_segments)} segments")
        
        if not censor_segments:
            logger.info("No segments to censor, copying original video")
            with self._safe_video_handling(input_path) as video:
                video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
            return
        
        try:
            with self._safe_video_handling(input_path) as video:
                if not video.audio:
                    raise VideoCensoringError("Input video has no audio track")
                
                # Get audio properties
                audio_array = video.audio.to_soundarray()
                sample_rate = video.audio.fps
                
                logger.info(f"Audio properties: {audio_array.shape} samples at {sample_rate}Hz")
                
                # Apply censoring to each segment
                for i, segment in enumerate(censor_segments, 1):
                    logger.info(f"Processing segment {i}/{len(censor_segments)}: "
                              f"{segment.word} ({segment.start_time:.2f}s-{segment.end_time:.2f}s)")
                    
                    start_sample = int(segment.start_time * sample_rate)
                    end_sample = int(segment.end_time * sample_rate)
                    duration = segment.end_time - segment.start_time
                    
                    # Validate sample bounds
                    start_sample = max(0, min(start_sample, len(audio_array)))
                    end_sample = max(start_sample, min(end_sample, len(audio_array)))
                    actual_duration = (end_sample - start_sample) / sample_rate
                    
                    if actual_duration > 0:
                        # Generate beep for this segment
                        beep_audio = self._generate_beep_audio(actual_duration, sample_rate)
                        
                        # Ensure beep matches the segment length exactly
                        beep_length = min(len(beep_audio), end_sample - start_sample)
                        if beep_length > 0:
                            audio_array[start_sample:start_sample + beep_length] = beep_audio[:beep_length]
                            logger.info(f"✓ Censored {actual_duration:.2f}s at {segment.start_time:.2f}s")
                    else:
                        logger.warning(f"⚠ Skipped invalid segment: {segment.word}")
                
                # Create new video with censored audio
                censored_audio = AudioArrayClip(audio_array, fps=sample_rate)
                final_video = video.set_audio(censored_audio)
                
                # Write final video
                final_video.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    preset="medium",
                    ffmpeg_params=['-crf', '20'],  # High quality
                    threads=4,
                    logger=None
                )
                
                logger.info(f"✓ Censored video saved to: {output_path}")
                
        except Exception as e:
            logger.error(f"X Video censoring failed: {str(e)}")
            raise VideoCensoringError(f"Video censoring failed: {str(e)}")
    
    def process_video(self, input_path: str, output_path: str, 
                     keep_temp_files: bool = False) -> Dict[str, any]:
        """
        Complete video processing pipeline
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            keep_temp_files: Whether to keep temporary files
            
        Returns:
            Dictionary with processing results and statistics
        """
        # Validate input
        if not Path(input_path).exists():
            raise VideoCensoringError(f"Input file not found: {input_path}")
        
        # Create temp directory
        temp_dir = Path("temp_video_processing")
        temp_dir.mkdir(exist_ok=True)
        temp_video_path = temp_dir / f"preprocessed_{Path(input_path).name}"
        
        try:
            logger.info(f"Starting video processing: {input_path}")
            
            # Step 1: Preprocess video
            self._preprocess_video(input_path, str(temp_video_path))
            
            # Step 2: Transcribe and detect profanity
            censor_segments = self.transcribe_and_detect_profanity(str(temp_video_path))
            
            # Step 3: Apply censoring
            self.censor_video(str(temp_video_path), censor_segments, output_path)
            
            # Compile results
            results = {
                "input_file": input_path,
                "output_file": output_path,
                "segments_censored": len(censor_segments),
                "total_censored_duration": sum(s.end_time - s.start_time for s in censor_segments),
                "profanity_detected": [{"word": s.word, "start": s.start_time, 
                                      "end": s.end_time, "confidence": s.confidence} 
                                     for s in censor_segments],
                "processing_successful": True
            }
            
            logger.info(f"Processing completed successfully!")
            logger.info(f"Statistics: {results['segments_censored']} segments censored, "
                       f"{results['total_censored_duration']:.2f}s total duration")
            
            return results
            
        except Exception as e:
            logger.error(f"X Processing failed: {str(e)}")
            return {
                "input_file": input_path,
                "processing_successful": False,
                "error": str(e)
            }
        finally:
            # Cleanup temp files
            if not keep_temp_files and temp_video_path.exists():
                temp_video_path.unlink()
                logger.info("Temporary files cleaned up")

# Usage Examples and Main Function
def basic_usage_example():
    """Basic usage example"""
    
    # Initialize the censoring system
    censor = VideoProfileCensor(
        beep_frequency=1000,      # 1kHz beep
        beep_volume=0.3,          # 30% volume
        confidence_threshold=0.8   # 80% confidence minimum
    )
    
    # Process a video
    results = censor.process_video(
        input_path="test.mp4",  # Change this to your input file
        output_path="censored_output2.mp4"
    )
    
    # Check results
    if results["processing_successful"]:
        print(f"✓ Success! Censored {results['segments_censored']} segments")
        print(f"Total censored duration: {results['total_censored_duration']:.2f} seconds")
        
        # Save detailed results
        with open("processing_results.json", "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Detailed results saved to processing_results.json")
    else:
        print(f"X Error: {results['error']}")

def custom_profanity_example():
    """Example with custom profanity list"""
    
    # Custom profanity list - add your specific words here
    custom_words = [
        "shit", "fuck", "damn", "hell", "crap",
        # Add more words as needed
    ]
    
    censor = VideoProfileCensor(
        profanity_list=custom_words,
        beep_frequency=800,       # Different beep sound
        beep_volume=0.4,
        confidence_threshold=0.7   # More sensitive detection
    )
    
    results = censor.process_video("test.mp4", "custom_censored.mp4")
    return results

def main():
    """Main function - modify this to suit your needs"""
    print("Video Profanity Censoring System")
    print("=" * 50)
    
    # Basic usage - just change the file names
    input_file = "test.mp4"  # Change this to your input video file
    output_file = "censored_output.mp4"  # Change this to your desired output file
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"X Error: Input file '{input_file}' not found!")
        print("Please make sure your video file exists and update the 'input_file' variable.")
        return
    
    try:
        # Initialize censoring system
        censor = VideoProfileCensor(
            beep_frequency=1000,      # Beep frequency in Hz
            beep_volume=0.3,          # Beep volume (0.0 to 1.0)
            confidence_threshold=0.8   # Confidence threshold for detection
        )
        
        # Process the video
        print(f"Processing video: {input_file}")
        results = censor.process_video(input_file, output_file)
        
        # Display results
        if results["processing_successful"]:
            print("\nSUCCESS!")
            print(f"✓ Input: {results['input_file']}")
            print(f"✓ Output: {results['output_file']}")
            print(f"✓ Segments censored: {results['segments_censored']}")
            print(f"✓ Total censored duration: {results['total_censored_duration']:.2f} seconds")
            
            if results['profanity_detected']:
                print("\nX Profanity detected:")
                for item in results['profanity_detected']:
                    print(f"   - '{item['word']}' at {item['start']:.2f}s-{item['end']:.2f}s")
        else:
            print(f"\nX FAILED: {results['error']}")
            
    except Exception as e:
        print(f"\nX Unexpected error: {str(e)}")
        print("Make sure you have:")
        print("1. Installed all dependencies: pip install google-cloud-videointelligence moviepy numpy")
        print("2. Set up Google Cloud credentials")
        print("3. Installed FFmpeg")

if __name__ == "__main__":
    # Run the main function
    main()
    
    # Uncomment these lines to run other examples:
    # print("\n" + "="*50)
    # print("Running custom profanity example...")
    # custom_profanity_example()