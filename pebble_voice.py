"""
This script runs on a Raspberry Pi and uses a Pebble watch as a remote
trigger for the Pi Camera. When a button is pressed on the Pebble, this
script captures an image, uploads it to Google's Gemini API, and asks a
question about the image.

---------------------------
--- IMPORTANT: PEBBLE SETUP ---
---------------------------
Before running this script for the first time, you must manually pair your
Pebble watch with the Raspberry Pi. This is a one-time setup.

1.  INSTALL BLUETOOTH TOOLS:
    sudo apt-get update && sudo apt-get install blueman

2.  START THE BLUETOOTH COMMAND-LINE TOOL:
    sudo bluetoothctl

3.  IN THE BLUETOOTH SHELL, TYPE THE FOLLOWING COMMANDS:
    (This makes the Pi discoverable and scans for nearby devices)
    
    agent on
    default-agent
    scan on

4.  FIND YOUR WATCH:
    After a few seconds, you will see a list of devices. Find your
    Pebble. It will look something like "Pebble Time XXYY". Note the
    MAC address (e.g., 00:11:22:33:AA:BB).

5.  PAIR AND TRUST THE WATCH:
    Replace [mac_address] with your watch's address.
    
    pair [mac_address]
    trust [mac_address]

    A prompt will appear on both the Pi and the Pebble. Confirm that the
    pairing codes match on both devices.

6.  EXIT BLUETOOTHCTL:
    disconnect [mac_address]
    exit

7.  BIND THE WATCH TO A SERIAL PORT:
    This makes the paired watch available at a consistent system path.
    This command must be run each time the Pi reboots, so it's a good
    idea to add it to a startup script (like /etc/rc.local).
    
    sudo rfcomm bind 0 [mac_address] 1

After completing these steps, you are ready to run this Python script.

-------------------------
--- PYTHON REQUIREMENTS ---
-------------------------
You will need to install libpebble2 in your virtual environment:
pip install libpebble2 google-generativeai sounddevice soundfile numpy

PEBBLE MAC ADDRESS: 51:7E:64:C0:B6:5E
"""
import time
import os
import sys
import threading
from functools import partial
import subprocess
import re

try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("Could not import picamera2. If in a virtual environment, run 'pip install picamera2'. Otherwise, run 'sudo apt install -y python3-picamera2'")

try:
    from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
    from google import genai
except ImportError:
    sys.exit("Could not import google.genai. Run 'pip install google-generativeai'")

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError:
    sys.exit("Could not import audio libraries. Please run 'pip install sounddevice soundfile numpy'.")

try:
    from libpebble2.communication import PebbleConnection
    from libpebble2.communication.transports.serial import SerialTransport
    from libpebble2.services.notifications import Notifications
except ImportError:
    sys.exit("Could not import libpebble2. Run 'pip install libpebble2'")


# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

MODEL_ID = "gemini-2.0-flash"
IMAGE_FILE_PATH = "captured_image.jpg"
AUDIO_FILE_PATH = "captured_audio.wav"
# This is the serial port created by the `rfcomm bind` command.
PEBBLE_SERIAL_PORT = "/dev/rfcomm0"

# This is the raw byte sequence discovered to correspond to the middle button.
MIDDLE_BUTTON_PACKET = b'\x00\x11\x004\x01\xde\xc0BL\x06%Hx\xb1\xf2\x14~W\xe86\x88'


def discover_and_setup():
    """
    Scans for Bluetooth devices, allows the user to select a Pebble,
    and prints the necessary setup commands.
    """
    print("Could not connect to a paired Pebble. Starting discovery...")
    pebbles = {}
    try:
        # Use a subprocess to run bluetoothctl and scan for devices
        print("Scanning for Bluetooth devices for 10 seconds...")
        proc = subprocess.Popen(['sudo', 'bluetoothctl'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        
        # We need to read and parse the output in real-time
        proc.stdin.write("scan on\n")
        proc.stdin.flush()
        
        time.sleep(10) # Scan for 10 seconds
        
        proc.stdin.write("scan off\n")
        proc.stdin.flush()
        
        proc.stdin.write("exit\n")
        proc.stdin.flush()
        
        output, _ = proc.communicate()

        # Parse the output to find Pebble devices
        for line in output.split('\n'):
            if 'Pebble' in line:
                mac_match = re.search(r'([0-9A-F]{2}:){5}[0-9A-F]{2}', line)
                if mac_match:
                    mac_address = mac_match.group(0)
                    device_name = line.split(mac_address)[1].strip()
                    pebbles[mac_address] = device_name
    
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
    print("--------------------------")

    try:
        choice = int(input("Select a watch to pair with (enter number): ")) - 1
        if not 0 <= choice < len(devices):
            print("Invalid selection.")
            return
        
        selected_mac, selected_name = devices[choice]
        print(f"\nYou selected: {selected_name}")
        
    except (ValueError, IndexError):
        print("Invalid input.")
        return

    print("\n--- REQUIRED SETUP COMMANDS ---")
    print("Please run the following commands in another terminal to pair and bind your watch.")
    print("You will only need to do this once.")
    print("\n1. Pair and Trust the device:")
    print(f"   sudo bluetoothctl pair {selected_mac}")
    print(f"   sudo bluetoothctl trust {selected_mac}")
    print("\n   (Confirm the pairing code on your watch and in the terminal if prompted)")
    print("\n2. Bind the watch to a serial port (run this after every reboot):")
    print(f"   sudo rfcomm bind 0 {selected_mac} 1")
    print("\nAfter running these commands, start this script again.")


class PebbleCameraTrigger:
    """
    Manages the connection to the Pebble, camera, and Gemini API.
    """
    def __init__(self):
        self._pebble = None
        self._notifications = None
        self._gemini_client = genai.Client(api_key=API_KEY)
        self._picam2 = Picamera2()
        self._is_recording = False
        self._audio_stream = None
        self._audio_frames = []
        # Check for available microphones
        try:
            sd.query_devices()
        except Exception as e:
            sys.exit(f"Could not query audio devices. Is a microphone connected? Error: {e}")
        self._samplerate = 44100

    def connect(self):
        """
        Initializes the camera and connects to the Pebble watch.
        """
        # --- Camera Setup ---
        print("Initializing camera...")
        config = self._picam2.create_still_configuration(main={"size": (1280, 720)})
        self._picam2.configure(config)
        self._picam2.start()
        time.sleep(2)
        print("Camera ready.")

        # --- Pebble Connection ---
        print(f"Connecting to Pebble on {PEBBLE_SERIAL_PORT}...")
        self._pebble = PebbleConnection(SerialTransport(PEBBLE_SERIAL_PORT))
        self._pebble.connect()
        self._notifications = Notifications(self._pebble)
        print("Pebble connected successfully!")

    def _debug_handler(self, packet):
        """
        Generic raw handler to print details of any packet received.
        This is used to identify the correct packet type for button presses.
        """
        print("\n--- DEBUG: Raw Packet Received ---")
        print(f"  Packet Type: {type(packet)}")
        print(f"  Packet Content: {packet}")
        print("----------------------------------\n")
        # After running and pressing a button, look for a packet that seems
        # to correspond to the button press. Then we can write a specific
        # handler for it.

    def _start_recording(self):
        """
        Starts recording audio from the default input device.
        """
        self._audio_frames = []  # Clear previous recording

        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            self._audio_frames.append(indata.copy())

        try:
            self._audio_stream = sd.InputStream(
                callback=callback,
                samplerate=self._samplerate,
                channels=1,
                dtype='int16'  # Standard for WAV files
            )
            self._audio_stream.start()
            print("Audio recording started.")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self._notifications.send_notification("Gemini Trigger", "Audio error. Check mic.", "Raspberry Pi")
            self._is_recording = False

    def _stop_recording(self):
        """
        Stops the audio recording and saves it to a WAV file.
        """
        if not self._audio_stream:
            return None
        
        self._audio_stream.stop()
        self._audio_stream.close()
        print("Audio recording stopped.")

        if not self._audio_frames:
            print("No audio frames recorded.")
            return None

        # Concatenate all recorded frames
        recording = np.concatenate(self._audio_frames, axis=0)
        
        # Save the recording as a WAV file
        sf.write(AUDIO_FILE_PATH, recording, self._samplerate)
        print(f"Audio saved to {AUDIO_FILE_PATH}")
        return AUDIO_FILE_PATH

    def _raw_packet_handler(self, packet):
        """
        Handles raw packets from the Pebble. The first press starts audio
        recording, the second press stops it and triggers the analysis.
        """
        # We can leave this print statement here for debugging other buttons.
        print(f"DEBUG: Received packet: {packet.hex()}")

        if packet == MIDDLE_BUTTON_PACKET:
            if not self._is_recording:
                # --- START RECORDING ---
                print("\n>>> Middle button press detected! Starting audio recording...")
                self._is_recording = True
                self._start_recording()
                if self._is_recording: # Check if recording actually started
                    self._notifications.send_notification("Gemini Trigger", "Recording... Press again to stop.", "Raspberry Pi")
            else:
                # --- STOP RECORDING AND ANALYZE ---
                print("\n>>> Middle button press detected! Stopping recording and starting analysis...")
                self._is_recording = False
                audio_path = self._stop_recording()

                if audio_path:
                    # Run the main logic in a separate thread to avoid blocking
                    threading.Thread(target=self._capture_and_analyze, args=(audio_path,)).start()
                else:
                    print("Audio recording failed, aborting analysis.")
                    self._notifications.send_notification("Gemini Trigger", "Recoding failed.", "Raspberry Pi")

    def _capture_and_analyze(self, audio_file_path):
        """
        The core logic: notifies the watch, captures, analyzes, and cleans up.
        """
        try:
            # --- Notify Pebble that the action has started ---
            self._notifications.send_notification("Gemini Trigger", "Capturing & analyzing...", "Raspberry Pi")

            # --- Camera Capture ---
            print(f"Capturing image to {IMAGE_FILE_PATH}...")
            self._picam2.capture_file(IMAGE_FILE_PATH)
            print("Capture complete.")

            # --- Gemini API Interaction ---
            print(f"Uploading {IMAGE_FILE_PATH} to the Gemini API...")
            image_file_resource = self._gemini_client.files.upload(file=IMAGE_FILE_PATH)
            
            print(f"Uploading {audio_file_path} to the Gemini API...")
            audio_file_resource = self._gemini_client.files.upload(file=audio_file_path)

            google_search_tool = Tool(google_search=GoogleSearch())

            print("Asking Gemini to analyze the image based on your audio query...")
            response = self._gemini_client.models.generate_content(
                model=MODEL_ID,
                contents=[audio_file_resource, image_file_resource],
                config=GenerateContentConfig(tools=[google_search_tool], response_modalities=["TEXT"])
            )

            # --- Print and Send Response ---
            if response.candidates:
                result_text = response.candidates[0].content.parts[0].text
                print("\n--- Gemini's Response ---")
                print(result_text)
                print("-------------------------\n")
                # Send the final result back to the watch
                self._notifications.send_notification("Gemini Result", result_text, "Raspberry Pi")
            else:
                print("No content generated.")
                if response.prompt_feedback:
                    print(f"Prompt Feedback: {response.prompt_feedback}")
                self._notifications.send_notification("Gemini Result", "Error: No content generated.", "Raspberry Pi")

        except Exception as e:
            print(f"An error occurred during capture or analysis: {e}")
            try:
                self._notifications.send_notification("Gemini Result", f"Error: {e}", "Raspberry Pi")
            except Exception as notif_e:
                print(f"Failed to send error notification to Pebble: {notif_e}")

        finally:
            # --- File Cleanup ---
            if os.path.exists(IMAGE_FILE_PATH):
                os.remove(IMAGE_FILE_PATH)
                print(f"Cleaned up temporary file: {IMAGE_FILE_PATH}")
            if os.path.exists(AUDIO_FILE_PATH):
                os.remove(AUDIO_FILE_PATH)
                print(f"Cleaned up temporary file: {AUDIO_FILE_PATH}")

    def run(self):
        """
        Registers a raw event handler and starts the event loop.
        """
        print("Registering raw packet handler...")
        self._pebble.register_raw_inbound_handler(self._raw_packet_handler)

        print("\nReady. Press the SELECT (middle) button on your Pebble to start recording your question.")
        print("Press it a second time to stop recording and trigger the Gemini analysis.")
        
        self._pebble.run_sync()

    def shutdown(self):
        """
        Cleans up resources gracefully.
        """
        print("\nShutting down...")
        # According to the docs, run_sync() blocks until disconnection.
        # No explicit disconnect/close call is needed for the Pebble.
        if hasattr(self, '_picam2') and self._picam2.started:
            self._picam2.stop()
            print("Camera stopped.")

def main():
    trigger = PebbleCameraTrigger()
    try:
        trigger.connect()
        trigger.run()
    except Exception as e:
        # Check if the error is due to the rfcomm port not existing, which
        # is the most common first-time setup issue.
        if "No such file or directory" in str(e):
            discover_and_setup()
        else:
            print(f"\nA critical error occurred: {e}")
            print("If this is your first time, the Pebble may not be paired correctly.")
            print("Please restart the script to run the interactive setup process if needed.")
    finally:
        trigger.shutdown()


if __name__ == "__main__":
    main() 