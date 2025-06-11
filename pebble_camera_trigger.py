"""
This script runs on a Raspberry Pi and uses a Pebble watch as a remote
trigger for the Pi Camera and for voice commands.
- A press of the MIDDLE button triggers an image capture and analysis.
- A press of the BOTTOM button starts and stops a voice recording, which is
  then sent to Google's Gemini API for a response.

---------------------------
--- IMPORTANT: PEBBLE SETUP ---
---------------------------
Follow these steps ONCE to pair your watch.
1.  INSTALL SYSTEM TOOLS:
    sudo apt-get update
    sudo apt-get install -y blueman ffmpeg

2.  START THE BLUETOOTH COMMAND-LINE TOOL:
    sudo bluetoothctl

3.  IN THE BLUETOOTH SHELL, PAIR THE WATCH:
    (Replace [mac_address] with your watch's address)
    remove [mac_address]  # Only if you've tried pairing before
    scan on
    # Wait for your Pebble to appear, then...
    pair [mac_address]
    trust [mac_address]
    exit

4.  BIND THE WATCH TO A SERIAL PORT (after every reboot):
    sudo rfcomm bind 0 [mac_address] 1

-------------------------
--- PYTHON REQUIREMENTS ---
-------------------------
You will need to install these libraries in your virtual environment:
pip install google-generativeai libpebble2 pydub

"""
import time
import os
import sys
import threading
import subprocess
import re
import uuid

try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("Could not import picamera2. Run 'pip install picamera2' or 'sudo apt install -y python3-picamera2'")

try:
    from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
    from google import genai
except ImportError:
    sys.exit("Could not import google.genai. Run 'pip install google-generativeai'")

try:
    from libpebble2.communication import PebbleConnection
    from libpebble2.communication.transports.serial import SerialTransport
    from libpebble2.services.notifications import Notifications
    from libpebble2.services.voice import VoiceService
    from libpebble2.protocol.timeline import TimelineItem, TimelineAttribute, TimelineAction
except ImportError:
    sys.exit("Could not import libpebble2. Run 'pip install libpebble2'")

try:
    from pydub import AudioSegment
except ImportError:
    sys.exit("Could not import pydub. Run 'pip install pydub'")

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

MODEL_ID = "gemini-2.0-flash"
IMAGE_FILE_PATH = "captured_image.jpg"
AUDIO_FILE_PATH_RAW = "recorded_audio.opus"
AUDIO_FILE_PATH_CONVERTED = "converted_audio.mp3"
PROMPT_IMAGE = "What is in this image? Be concise."
PROMPT_AUDIO = "Fulfill the request in this audio. If it's a question, answer it. If it's a command, confirm it."

PEBBLE_SERIAL_PORT = "/dev/rfcomm0"
MIDDLE_BUTTON_PACKET = b'\x00\x11\x004\x01\xde\xc0BL\x06%Hx\xb1\xf2\x14~W\xe86\x88'
BOTTOM_BUTTON_PACKET = bytes.fromhex('00110034023af858c316cb456191e7f1ad2df8725f')


class PebbleGeminiBridge:
    """
    Manages connections and triggers for image and voice requests.
    """
    def __init__(self):
        self._pebble = None
        self._notifications = None
        self._gemini_client = genai.Client(api_key=API_KEY)
        self._picam2 = Picamera2()
        self._is_recording = False
        self._voice_service = None
        self._audio_file = None

    def connect(self):
        print("Initializing camera...")
        config = self._picam2.create_still_configuration(main={"size": (1280, 720)})
        self._picam2.configure(config)
        self._picam2.start()
        time.sleep(2)
        print("Camera ready.")

        print(f"Connecting to Pebble on {PEBBLE_SERIAL_PORT}...")
        self._pebble = PebbleConnection(SerialTransport(PEBBLE_SERIAL_PORT))
        self._pebble.connect()
        self._notifications = Notifications(self._pebble)
        
        # Initialize VoiceService immediately and wait for the watch to connect.
        self._voice_service = VoiceService(self._pebble)
        self._voice_service.register_handler("audio_data", self._handle_audio_data)
        self._voice_service.register_handler("session_ended", self._stop_recording_and_analyze)
        
        print("Pebble connected successfully!")

    def _raw_packet_handler(self, packet):
        print(f"DEBUG: Received packet: {packet.hex()}")
        if packet == MIDDLE_BUTTON_PACKET:
            print("\n>>> Middle button press detected! Starting image analysis...")
            threading.Thread(target=self._capture_and_analyze_image).start()
        elif packet == BOTTOM_BUTTON_PACKET:
            print("\n>>> Bottom button press detected! Sending voice prompt to watch...")
            self._send_voice_prompt()
            
    def _send_voice_prompt(self):
        """
        Sends a notification with a "Reply with Voice" action to the watch.
        This is the new trigger for the voice workflow.
        """
        try:
            # Construct a timeline pin with a voice reply action
            pin_id = str(uuid.uuid4())
            action = TimelineAction(action_id=0, type=TimelineAction.Type.Response)
            
            # The debug output has shown us the correct names.
            # - The parameter is 'type'.
            # - The value is 'Notification' (with a capital N).
            pin = TimelineItem(
                item_id=pin_id,
                parent_id=pin_id,
                timestamp=int(time.time()),
                duration=0,
                type=TimelineItem.Type.Notification,
                flags=0,
                layout=0x01,  # Generic Notification Layout
                attributes=[
                    TimelineAttribute(attribute_id=1, content="Voice Command".encode('utf-8')),
                    TimelineAttribute(attribute_id=3, content="Reply with voice to send a command to Gemini.".encode('utf-8')),
                ],
                actions=[action]
            )
            self._pebble.send_packet(pin)
            print("Voice prompt sent to watch. Please open it and select 'Reply with Voice'.")
        except Exception as e:
            print(f"Failed to send voice prompt: {type(e).__name__}: {e}")

    def _handle_audio_data(self, data):
        """
        Receives audio chunks from the watch and writes them to a file.
        """
        if not self._is_recording:
            print("Audio stream started. Recording...")
            self._is_recording = True
            self._audio_file = open(AUDIO_FILE_PATH_RAW, 'wb')

        if self._audio_file:
            self._audio_file.write(data)

    def _stop_recording_and_analyze(self):
        """
        Triggered when the watch signals the end of the voice session.
        """
        if not self._is_recording:
            return # Avoids triggering on session setup failures

        print("Voice session ended. Processing audio...")
        self._is_recording = False
        if self._audio_file:
            self._audio_file.close()
        
        threading.Thread(target=self._process_audio_with_gemini).start()

    def _process_audio_with_gemini(self):
        try:
            print("Converting Opus audio to MP3...")
            sound = AudioSegment.from_file(AUDIO_FILE_PATH_RAW, format="opus")
            sound.export(AUDIO_FILE_PATH_CONVERTED, format="mp3")
            print("Conversion complete.")

            print(f"Uploading {AUDIO_FILE_PATH_CONVERTED} to the Gemini API...")
            audio_file_resource = self._gemini_client.files.upload(file=AUDIO_FILE_PATH_CONVERTED)

            print(f"Asking Gemini: '{PROMPT_AUDIO}'")
            response = self._gemini_client.models.generate_content(
                model=MODEL_ID,
                contents=[PROMPT_AUDIO, audio_file_resource]
            )
            self._handle_gemini_response(response, "Audio Result")
        except Exception as e:
            print(f"An error occurred during audio processing: {e}")
            self._notifications.send_notification("Gemini Error", str(e), "Raspberry Pi")
        finally:
            self._cleanup_file(AUDIO_FILE_PATH_RAW)
            self._cleanup_file(AUDIO_FILE_PATH_CONVERTED)

    def _capture_and_analyze_image(self):
        try:
            self._notifications.send_notification("Image Trigger", "Capturing image...", "Raspberry Pi")
            print(f"Capturing image to {IMAGE_FILE_PATH}...")
            self._picam2.capture_file(IMAGE_FILE_PATH)
            print("Capture complete.")

            print(f"Uploading {IMAGE_FILE_PATH} to the Gemini API...")
            image_file_resource = self._gemini_client.files.upload(file=IMAGE_FILE_PATH)

            google_search_tool = Tool(google_search=GoogleSearch())
            print(f"Asking Gemini: '{PROMPT_IMAGE}'")
            response = self._gemini_client.models.generate_content(
                model=MODEL_ID,
                contents=[image_file_resource, PROMPT_IMAGE],
                config=GenerateContentConfig(tools=[google_search_tool], response_modalities=["TEXT"])
            )
            self._handle_gemini_response(response, "Image Result")
        except Exception as e:
            print(f"An error occurred during image analysis: {e}")
            self._notifications.send_notification("Gemini Error", str(e), "Raspberry Pi")
        finally:
            self._cleanup_file(IMAGE_FILE_PATH)

    def _handle_gemini_response(self, response, result_title):
        if response.candidates and response.candidates[0].content.parts:
            result_text = response.candidates[0].content.parts[0].text
            print(f"\n--- Gemini's {result_title} ---")
            print(result_text)
            print("-------------------------\n")
            self._notifications.send_notification(result_title, result_text, "Raspberry Pi")
        else:
            print("No content generated.")
            self._notifications.send_notification(result_title, "Error: No content generated.", "Raspberry Pi")
    
    def _cleanup_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")

    def run(self):
        print("Registering raw packet handler...")
        self._pebble.register_raw_inbound_handler(self._raw_packet_handler)
        print("\nReady. Middle button for image, Bottom button to toggle voice.")
        self._pebble.run_sync()

    def shutdown(self):
        print("\nShutting down...")
        if hasattr(self, '_picam2') and self._picam2.started:
            self._picam2.stop()
            print("Camera stopped.")

def main():
    trigger = PebbleGeminiBridge()
    try:
        trigger.connect()
        trigger.run()
    except Exception as e:
        if "No such file or directory" in str(e):
            # The discover_and_setup function is removed for brevity,
            # assuming the user follows the manual setup instructions.
            print("Error: Could not find /dev/rfcomm0.")
            print("Please ensure the Pebble is paired and bound correctly.")
            print("See the setup instructions at the top of this script.")
        else:
            print(f"\nA critical error occurred: {e}")
    finally:
        trigger.shutdown()

if __name__ == "__main__":
    main() 