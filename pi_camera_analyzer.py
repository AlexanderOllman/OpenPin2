"""
This script runs on a Raspberry Pi with a camera to capture an image,
upload it to Google's Gemini API, and ask a question about the image.

Before running:
1. Make sure your Pi Camera is connected and enabled in raspi-config.
2. Install the necessary Python libraries:
   - For the Gemini API: pip install google-generativeai
   - For the camera: sudo apt update && sudo apt install -y python3-picamera2
3. Set your Google API key as an environment variable for security:
   export GOOGLE_API_KEY="YOUR_API_KEY"
   (You can add this to your ~/.bashrc file to make it permanent)
"""
import time
import os
import sys

try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("Could not import picamera2. Run 'sudo apt install -y python3-picamera2'")

try:
    from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
    from google import genai
except ImportError:
    sys.exit("Could not import google.genai. Run 'pip install google-generativeai'")


# --- Configuration ---
# Fetches the API key from environment variables.
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running the script.")

MODEL_ID = "gemini-2.0-flash"
IMAGE_FILE_PATH = "captured_image.jpg"
PROMPT = "What is in this image?"

# --- Main Execution ---
def main():
    """
    Initializes the camera, and then enters a loop to capture and analyze
    images with Gemini each time the user presses Enter.
    """
    # --- Initialization ---
    client = genai.Client(api_key=API_KEY)
    picam2 = Picamera2()

    # --- Camera Setup ---
    print("Initializing and starting camera...")
    # A lower resolution will result in a smaller file and faster uploads.
    config = picam2.create_still_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Allow camera time to auto-adjust.
    print("Camera ready.")

    try:
        while True:
            # --- Wait for user trigger ---
            input("\nPress Enter to capture an image, or Ctrl+C to exit...")

            # The main logic is now in a try/except/finally block to handle
            # errors in one cycle without crashing the whole program.
            try:
                # --- Camera Capture ---
                print(f"Capturing image to {IMAGE_FILE_PATH}...")
                picam2.capture_file(IMAGE_FILE_PATH)
                print("Capture complete.")

                # --- Gemini API Interaction ---
                print(f"Uploading {IMAGE_FILE_PATH} to the Gemini API...")
                image_file_resource = client.files.upload(file=IMAGE_FILE_PATH)

                google_search_tool = Tool(
                    google_search=GoogleSearch()
                )

                print(f"Asking Gemini: '{PROMPT}'")
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[image_file_resource, PROMPT],
                    config=GenerateContentConfig(
                        tools=[google_search_tool],
                        response_modalities=["TEXT"],
                    )
                )

                # --- Print Response ---
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        print("\n--- Gemini's Response ---")
                        print(part.text)
                        print("-------------------------\n")
                else:
                    print("No content generated. Check your API key and model configuration.")
                    if response.prompt_feedback:
                         print(f"Prompt Feedback: {response.prompt_feedback}")

            except Exception as e:
                print(f"An error occurred during capture or analysis: {e}")

            finally:
                # --- File Cleanup ---
                if os.path.exists(IMAGE_FILE_PATH):
                    os.remove(IMAGE_FILE_PATH)
                    print(f"Cleaned up temporary file: {IMAGE_FILE_PATH}")

    except KeyboardInterrupt:
        print("\nExiting program.")
    finally:
        # --- Camera Cleanup ---
        picam2.stop()
        print("Camera stopped.")


if __name__ == "__main__":
    main() 