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
pip install libpebble2 google-generativeai

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
    sys.exit("Could not import google.genai. Run 'pip install google-genai'")

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
PROMPT = "What is in this image? Be concise."
# This is the serial port created by the `rfcomm bind` command.
PEBBLE_SERIAL_PORT = "/dev/rfcomm0"


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


    def _button_handler(self, event):
        """
        Handles button press events from the Pebble.
        Triggers on the SELECT (middle) button.
        """
        if event.button == 'SELECT':
            print("\n>>> Select button pressed! Starting capture and analysis...")
            # Run the main logic in a separate thread to avoid blocking the
            # Pebble's event loop. This keeps the watch responsive.
            threading.Thread(target=self._capture_and_analyze).start()

    def _capture_and_analyze(self):
        """
        The core logic: notifies the watch, captures, analyzes, and cleans up.
        """
        try:
            # --- Notify Pebble that the action has started ---
            self._notifications.send_notification("Gemini Trigger", "Capturing image...", "Raspberry Pi")

            # --- Camera Capture ---
            print(f"Capturing image to {IMAGE_FILE_PATH}...")
            self._picam2.capture_file(IMAGE_FILE_PATH)
            print("Capture complete.")

            # --- Gemini API Interaction ---
            print(f"Uploading {IMAGE_FILE_PATH} to the Gemini API...")
            image_file_resource = self._gemini_client.files.upload(file=IMAGE_FILE_PATH)

            google_search_tool = Tool(google_search=GoogleSearch())

            print(f"Asking Gemini: '{PROMPT}'")
            response = self._gemini_client.models.generate_content(
                model=MODEL_ID,
                contents=[image_file_resource, PROMPT],
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

    def run(self):
        """
        Registers the event handler and starts the main event loop.
        """
        self._pebble.register_handler("button", self._button_handler)
        print("Ready. Press the SELECT (middle) button on your Pebble to trigger an image capture.")
        self._pebble.run_forever()

    def shutdown(self):
        """
        Cleans up resources gracefully.
        """
        print("\nShutting down...")
        if self._pebble and self._pebble.is_connected:
            self._pebble.disconnect()
            print("Pebble disconnected.")
        if self._picam2.started:
            self._picam2.stop()
            print("Camera stopped.")

def main():
    trigger = PebbleCameraTrigger()
    try:
        trigger.connect()
        trigger.run()
    except FileNotFoundError:
        # This error is expected if /dev/rfcomm0 doesn't exist.
        # Trigger the interactive setup process.
        discover_and_setup()
    except Exception as e:
        print(f"\nA critical error occurred: {e}")
        print("If this is your first time, the Pebble may not be paired correctly.")
        print("Please restart the script to run the interactive setup process.")
    finally:
        trigger.shutdown()


if __name__ == "__main__":
    main() 