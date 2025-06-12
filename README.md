# 🎬 Video Moderation

This project automatically detects and censors vulgar words from videos using Google Gemini for transcription and MoviePy for audio/video editing. It identifies timestamps of profanities and replaces them with a custom beep sound, outputting a clean version of the video.

---

## 🚀 Features

- ✅ Detects vulgar words using Gemini AI
- 🔊 Replaces offensive words with custom-generated beep sound
- 🎧 Smooth audio transitions using crossfades
- 📼 Outputs a clean video with censored audio
- 📁 Works on local `.mp4` video files (max 20MB for transcription)

---

## 🛠️ Requirements

- Python 3.8+
- FFmpeg (must be installed and available in your system path)

### Install FFmpeg:

- **Windows:**  
  `choco install ffmpeg`

- **macOS:**  
  `brew install ffmpeg`

- **Linux:**  
  `sudo apt-get install ffmpeg`

---

## 📦 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Roshanchau/video-moderation.git
   cd video-moderation
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your Google Gemini API key:

   Open `main.py` and replace the following line with your own key:

   ```python
   client = genai.Client(api_key='YOUR_API_KEY_HERE')
   ```

---

## 🧪 Usage

1. Place your input video as `test.mp4` in the root directory.
2. Run the script:

   ```bash
   python main.py
   ```

3. Output:

   - A censored version of the video will be saved as `censored_output.mp4`.
   - If no vulgar words are detected, the original video is saved.

---

## 🧠 How It Works

- The input video is preprocessed for consistent audio quality.
- Video is transcribed using Gemini.
- Vulgar words are identified with timestamps.
- Audio is censored precisely using generated beep sounds with envelope smoothing.
- Final video is exported with the censored audio.

---

## 📝 Profanity List

You can customize the list of vulgar words in the `VULGAR_WORDS` dictionary inside `videoModeration.py`.

```python
VULGAR_WORDS = {
    "shit": ["shit", "shitty"],
    ...
}
```

---

## 📁 File Structure

```
video-moderation/
│
├── main.py         # Main script
├── requirements.txt           # Python dependencies
├── test.mp4                   # Your input video (place here)
└── censored_output.mp4        # Output after processing
```

---

## ⚠️ Limitations

- Only supports videos smaller than 20MB (Gemini inline upload limit).
- Gemini transcription may vary in accuracy.
- Only handles listed vulgar words — not slangs/variations unless specified.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Author

Built with ❤️ by [Roshan Chaudhary](https://github.com/Roshanchau)