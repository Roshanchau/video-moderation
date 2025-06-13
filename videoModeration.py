import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

def moderate_youtube_video(youtube_url):
    client = genai.Client(os.getenv("GOOGLE_API_KEY"))

    response = client.models.generate_content(  
        model='models/gemini-2.0-flash',
        contents=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(file_uri=youtube_url)
                ),
                types.Part(text="You are a content moderator. Your task is to analyze the provided video "
                        "and classify it based on the following harm types related to harassment and abuse: "
                        "Harassment (harass, intimidate, or bully others), Toxic (rude, disrespectful, or unreasonable), "
                        "Hate (promotes violence or attacks based on protected characteristics). "
                        "Output the result in JSON format with fields: violation (yes or no), harm_type, and explanation.")
            ]
        )
    )

    print(response.text)

if __name__ == "__main__":
    # Replace with your target YouTube video URL
    moderate_youtube_video("https://www.youtube.com/watch?v=pPw_izFr5PA")
