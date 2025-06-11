"""
This script runs on a Raspberry Pi and uses a Pebble watch as a voice
input device for the Google Gemini API.

----------------------
--- HOW IT WORKS ---
----------------------
1.  Press the MIDDLE (Select) button on your Pebble to start recording.
    A "Recording..." notification will appear on your watch.
2.  Speak your prompt or question.
3.  Press the MIDDLE button again to stop recording.
4.  The script sends your recorded audio to the Gemini API for transcription.
5.  The transcribed text is sent back to your watch as a notification.

---------------------------
--- IMPORTANT: PEBBLE SETUP ---
---------------------------
If you have not done so already, you must manually pair your Pebble watch
with the Raspberry Pi. This is a one-time setup. Please refer to the
discover_and_setup() function for the interactive guide.

-------------------------
--- PYTHON REQUIREMENTS ---
-------------------------
You will need to install the required libraries in your virtual environment:
pip install libpebble2 google-generativeai

"""
import time
import os
import sys
import threading
import subprocess
import re
import traceback

try:
    from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
    from google import genai
except ImportError:
    sys.exit("Could not import google.genai. Run 'pip install google-generativeai'")

try:
    from libpebble2.communication import PebbleConnection
    from libpebble2.communication.transports.serial import SerialTransport
    from libpebble2.services.notifications import Notifications
    from libpebble2.services.voice import VoiceService, SetupResult, VoiceTranscription
except ImportError:
    sys.exit("Could not import libpebble2. Run 'pip install libpebble2'")


# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

MODEL_ID = "gemini-2.0-flash"
PEBBLE_SERIAL_PORT = "/dev/rfcomm0"


def discover_and_setup():
    """
    Scans for Bluetooth devices, allows the user to select a Pebble,
    and prints the necessary setup commands.
    """
    print("Could not connect to a paired Pebble. Starting discovery...")
    pebbles = {}
    try:
        print("Scanning for Bluetooth devices for 10 seconds...")
        proc = subprocess.Popen(['sudo', 'bluetoothctl'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        proc.stdin.write("scan on\n")
        proc.stdin.flush()
        time.sleep(10)
        proc.stdin.write("scan off\nexit\n")
        proc.stdin.flush()
        output, _ = proc.communicate()

        for line in output.split('\n'):
            if 'Pebble' in line:
                mac_match = re.search(r'([0-9A-F]{2}:){5}[0-9A-F]{2}', line)
                if mac_match:
                    mac_address = mac_match.group(0)
                    pebbles[mac_address] = line.split(mac_address)[1].strip()
    
    except FileNotFoundError:
        print("Error: 'bluetoothctl' not found. Please install bluetooth tools with 'sudo apt-get install blueman'")
        return
    except Exception as e:
        print(f"An error occurred during Bluetooth scan: {e}")
        return

    if not pebbles:
        print("\nNo Pebble watches found. Make sure your watch is on and discoverable.")
        return

    print("\n--- Found Pebble Watches ---")
    devices = list(pebbles.items())
    for i, (mac, name) in enumerate(devices):
        print(f"{i+1}: {name} ({mac})")
    
    try:
        choice = int(input("Select a watch to pair with (enter number): ")) - 1
        selected_mac, _ = devices[choice]
    except (ValueError, IndexError):
        print("Invalid input.")
        return

    print("\n--- REQUIRED SETUP COMMANDS ---")
    print("Please run these commands in another terminal to pair your watch.")
    print("1. Remove any old pairings:")
    print(f"   sudo bluetoothctl remove {selected_mac}")
    print("2. Pair and Trust the device:")
    print(f"   sudo bluetoothctl pair {selected_mac}")
    print(f"   sudo bluetoothctl trust {selected_mac}")
    print("3. Bind the watch to a serial port (run this after every reboot):")
    print(f"   sudo rfcomm bind 0 {selected_mac} 1")
    print("\nAfter running these commands, start this script again.")


class PebbleGeminiController:
    """
    Manages the connection to the Pebble and the voice dictation workflow.
    """
    def __init__(self):
        self._pebble = None
        self._notifications = None
        self._voice_service = None
        self._gemini_client = genai.Client(api_key=API_KEY)
        self._audio_buffer = bytearray()

    def connect(self):
        """ Initializes the connection to the Pebble watch. """
        print(f"Connecting to Pebble on {PEBBLE_SERIAL_PORT}...")
        self._pebble = PebbleConnection(SerialTransport(PEBBLE_SERIAL_PORT))
        self._pebble.connect()
        self._notifications = Notifications(self._pebble)
        self._voice_service = VoiceService(self._pebble)
        print("Pebble connected successfully!")

    def _on_voice_control(self, data):
        """
        This is the main entry point, triggered when the watch wants to start
        a voice session.
        """
        print("\n>>> Voice session requested by watch. Clearing buffer.")
        self._audio_buffer.clear()
        
        # We must register handlers here, inside the request context.
        self._voice_service.register_handler("audio_data", self._handle_audio_data)
        self._voice_service.register_handler("session_end", self._handle_session_end)
        
        # Now we can safely tell the watch we are ready.
        self._voice_service.send_session_setup_result(result=SetupResult.Success)

    def _handle_audio_data(self, data):
        """ Appends incoming audio data chunks to the buffer. """
        self._audio_buffer.extend(data)
        
    def _handle_session_end(self, data):
        """
        Triggered when the watch stops sending audio. This is where we
        start the analysis.
        """
        print(">>> Voice session ended from watch. Processing audio...")

        audio_filename = "dictated_audio.ogg"
        with open(audio_filename, "wb") as f:
            f.write(self._audio_buffer)

        # It's crucial to unregister the handlers to prevent duplicates
        # on the next session.
        self._voice_service.unregister_handler("audio_data")
        self._voice_service.unregister_handler("session_end")

        threading.Thread(target=self._analyze_audio, args=(audio_filename,)).start()

    def _analyze_audio(self, filename):
        """ Uploads the audio to Gemini and sends the result to the Pebble. """
        try:
            print(f"Uploading {filename} to the Gemini API...")
            audio_file = self._gemini_client.files.upload(file=filename)

            print("Asking Gemini to transcribe the audio...")
            prompt = "Transcribe this audio and return only the transcribed text."
            response = self._gemini_client.models.generate_content(
                model=MODEL_ID,
                contents=[prompt, audio_file]
            )

            result_text = response.text
            print("\n--- Gemini's Transcription ---")
            print(result_text)
            print("------------------------------\n")
            # The VoiceService requires a specific object type for the result.
            self._voice_service.send_dictation_result(
                transcriptions=[VoiceTranscription(text=result_text, score=1.0)]
            )

        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            self._notifications.send_notification("Gemini Error", f"Error: {e}", "Raspberry Pi")
        finally:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Cleaned up temporary file: {filename}")

    def run(self):
        """ Registers the main event handler and starts the event loop. """
        # We listen for the watch to ask to start a session.
        self._voice_service.register_handler("voice_control", self._on_voice_control)

        print("\nReady. Activate voice dictation on your Pebble to begin.")
        print("(This is usually a long-press of the back or select button).")
        
        self._pebble.run_sync()

    def shutdown(self):
        """ Cleans up resources gracefully. """
        print("\nShutting down...")
        # run_sync() blocks until disconnection, so no explicit close is needed.

def main():
    controller = PebbleGeminiController()
    try:
        controller.connect()
        controller.run()
    except FileNotFoundError:
        discover_and_setup()
    except Exception as e:
        print(f"\nA critical error occurred. Exception type: {type(e)}")
        print("--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------")
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main() 