from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

# --- Setup ---
api_key = os.getenv("GEMINI_API_KEY")
# The print statement below is commented out for safety but kept for debugging context
# print("Loaded GEMINI_API_KEY:", api_key) 

# Initialize the client
client = genai.Client(api_key=api_key)

# --- API Call ---
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["Hello, test this API"],
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant. Respond concisely to the user's greeting.",
            temperature=0.7,
            max_output_tokens=50
        )
    )

    # --- Result Extraction (Corrected) ---
    if response.text:
        print("Model Response:")
        print(response.text)
    else:
        # Check if the response was blocked by safety settings
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            print(f"API call blocked. Reason: {response.prompt_feedback.block_reason.name}")
        else:
            print("No text content returned (possibly empty response or an issue occurred).")

except Exception as e:
    print(f"An error occurred during the API call: {e}")