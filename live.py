"""
This script runs on a Raspberry Pi and uses a Pebble watch as a remote
trigger for the Pi Camera. When a button is pressed on the Pebble, this
script starts a real-time, multimodal session with Google's Gemini API.

---------------------------
--- IMPORTANT: PEBBLE SETUP ---
---------------------------
Before running this script for the first time, you must manually pair your
Pebble watch with the Raspberry Pi. This is a one-time setup.

1.  INSTALL BLUETOOTH TOOLS:
    sudo apt-get update && sudo apt-get install blueman

2.  START THE BLUETOETOOTH COMMAND-LINE TOOL:
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
You will need to install the following libraries in your virtual environment:
pip install libpebble2 google-generativeai pyaudio Pillow "taskgroup;python_version<'3.11'" "exceptiongroup;python_version<'3.11'"

And install system dependencies for audio:
sudo apt-get install libportaudio2 portaudio19-dev

PEBBLE MAC ADDRESS: 51:7E:64:C0:B6:5E
"""
import time
import os
import sys
import threading
import subprocess
import re
import asyncio
import base64
import io
import traceback

try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("Could not import picamera2. If in a virtual environment, run 'pip install picamera2'. Otherwise, run 'sudo apt install -y python3-picamera2'")

try:
    from google import genai
except ImportError:
    sys.exit("Could not import google.genai. Run 'pip install google-generativeai'")

try:
    import pyaudio
    import PIL.Image
except ImportError:
    sys.exit("Could not import audio/image libraries. Please run 'pip install pyaudio Pillow'.")

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

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

# Use the new Live model
MODEL_ID = "models/gemini-2.0-flash-live-001"
# This is the serial port created by the `rfcomm bind` command.
PEBBLE_SERIAL_PORT = "/dev/rfcomm0"

# This is the raw byte sequence discovered to correspond to the middle button.
MIDDLE_BUTTON_PACKET = b'\x00\x11\x004\x01\xde\xc0BL\x06%Hx\xb1\xf2\x14~W\xe86\x88'

# Audio settings from the example
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# --- Gemini Client Setup ---
# Using v1beta for the Live API
client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1beta"})
LIVE_SESSION_CONFIG = {"response_modalities": ["AUDIO"]}
pya = pyaudio.PyAudio()


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


class PebbleLiveSession:
    """
    Manages the connection to the Pebble and the Gemini Live session.
    """
    def __init__(self):
        self._pebble = None
        self._notifications = None
        self._picam2 = Picamera2()
        
        self.is_session_running = False
        self._session_future = None
        self._loop = None
        
        # For the live session
        self._session = None
        self._audio_in_queue = None
        self._out_queue = None
        self._audio_stream = None

    def connect(self):
        """
        Initializes the camera and attempts to connect to the Pebble watch.
        If the watch is not found, it continues allowing for keyboard-only control.
        """
        # --- Camera Setup ---
        print("Initializing camera...")
        config = self._picam2.create_still_configuration(main={"size": (1280, 720)})
        self._picam2.configure(config)
        print("Camera ready.")

        # --- Pebble Connection ---
        try:
            print(f"Connecting to Pebble on {PEBBLE_SERIAL_PORT}...")
            self._pebble = PebbleConnection(SerialTransport(PEBBLE_SERIAL_PORT))
            self._pebble.connect()
            self._notifications = Notifications(self._pebble)
            print("Pebble connected successfully!")
        except Exception as e:
            print(f"Could not connect to Pebble watch: {e}")
            print("Continuing without Pebble. Use the [Enter] key to trigger the session.")
            self._pebble = None
            self._notifications = None

    def _get_frame_picamera(self):
        """Captures a frame from Picamera2, converts, and encodes it."""
        frame_array = self._picam2.capture_array()
        img = PIL.Image.fromarray(frame_array)
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        """Async task to capture frames from the camera."""
        self._picam2.start()
        time.sleep(2)  # Allow camera to warm up
        while self.is_session_running:
            try:
                frame = await asyncio.to_thread(self._get_frame_picamera)
                await self._out_queue.put(frame)
                await asyncio.sleep(1.0)  # 1 FPS
            except Exception as e:
                print(f"Error getting frame: {e}")
                break
        if self._picam2.started:
            self._picam2.stop()

    async def listen_audio(self):
        """Async task to listen to the microphone."""
        mic_info = pya.get_default_input_device_info()
        self._audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
            input=True, input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE
        )
        print("Audio recording started.")
        while self.is_session_running:
            try:
                data = await asyncio.to_thread(self._audio_stream.read, CHUNK_SIZE, exception_on_overflow=False)
                await self._out_queue.put({"data": data, "mime_type": "audio/pcm"})
            except (IOError, asyncio.CancelledError):
                break # Stop on error or cancellation
        print("Audio recording stopped.")

    async def send_realtime(self):
        """Async task to send data from the queue to Gemini."""
        while self.is_session_running:
            try:
                msg = await self._out_queue.get()
                await self._session.send_realtime_input(input=msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error sending data: {e}")

    async def receive_audio(self):
        """Async task to receive responses from Gemini."""
        while self.is_session_running:
            try:
                turn = self._session.receive()
                async for response in turn:
                    for part in response.parts:
                        if part.text:
                            print(part.text, end="", flush=True)
                        if hasattr(part, 'inline_data') and part.inline_data.data:
                            self._audio_in_queue.put_nowait(part.inline_data.data)

                while not self._audio_in_queue.empty():
                    self._audio_in_queue.get_nowait()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error receiving audio: {e}")

    async def play_audio(self):
        """Async task to play back audio from Gemini."""
        stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE, output=True
        )
        while self.is_session_running:
            try:
                bytestream = await self._audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
            except asyncio.CancelledError:
                break
        stream.close()

    async def run_live_session(self):
        """The main async method that orchestrates the live session."""
        try:
            async with (
                client.aio.live.connect(model=MODEL_ID, config=LIVE_SESSION_CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self._session = session
                self._audio_in_queue = asyncio.Queue()
                self._out_queue = asyncio.Queue(maxsize=10)

                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.get_frames())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                print("\n>>> Gemini Live session started. Speak your question. <<<")
                if self._notifications:
                    self._notifications.send_notification("Gemini Live", "Session Active", "Raspberry Pi")

        except (asyncio.CancelledError, asyncio.ExceptionGroup):
            print("\nLive session task group cancelled.")
        except Exception as e:
            print(f"An error occurred in the live session: {e}")
            traceback.print_exc()
        finally:
            if self._audio_stream and self._audio_stream.is_active():
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            print("Live session cleanup complete.")
            # This will trigger the _session_done_callback
            self.is_session_running = False

    def _session_done_callback(self, future):
        """Callback executed when the session future completes."""
        try:
            future.result()
        except asyncio.CancelledError:
            print("Session cancelled by user.")
        except Exception as e:
            print(f"Live session ended with an error: {e}")
        
        self.is_session_running = False
        self._session_future = None
        print("\n>>> Gemini Live session ended. Ready for next command. <<<")
        if self._notifications:
            self._notifications.send_notification("Gemini Live", "Session Ended", "Raspberry Pi")

    def _raw_packet_handler(self, packet):
        """Handles raw packets from the Pebble to start/stop the session."""
        if packet == MIDDLE_BUTTON_PACKET:
            self._toggle_session()

    def _toggle_session(self):
        """Starts or stops the Gemini Live session."""
        if not self.is_session_running:
            if not self._loop:
                print("Error: Event loop is not running.")
                return
            
            print("\n>>> Middle button press detected! Starting Gemini Live session...")
            self.is_session_running = True
            coro = self.run_live_session()
            self._session_future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            self._session_future.add_done_callback(self._session_done_callback)
        else:
            print("\n>>> Middle button press detected! Stopping Gemini Live session...")
            if self._session_future:
                # This signals all async tasks to stop
                self.is_session_running = False
                # And cancels the main task group
                self._loop.call_soon_threadsafe(self._session_future.cancel)

    def run(self):
        """
        Registers handlers, starts the asyncio event loop in a background thread,
        and listens for Pebble events and keyboard input.
        """
        # Start asyncio event loop in a background thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        loop_thread.start()

        pebble_thread = None
        if self._pebble:
            print("Registering raw packet handler...")
            self._pebble.register_raw_inbound_handler(self._raw_packet_handler)

            print("\nReady. Press the SELECT (middle) button on your Pebble to start a Gemini Live session.")
            print("Press it a second time to stop the session.")
            print("Alternatively, press [Enter] in this terminal to simulate a button press.")
            
            pebble_thread = threading.Thread(target=self._pebble.run_sync)
            pebble_thread.daemon = True
            pebble_thread.start()
        else:
            print("\nReady. Press [Enter] in this terminal to start a Gemini Live session.")
            print("Press [Enter] again to stop the session.")

        # Listen for keyboard input in the main thread
        while True:
            try:
                # If we are connected to a pebble, exit when its thread dies.
                # Otherwise, this loop runs until the user presses Ctrl+C.
                if pebble_thread and not pebble_thread.is_alive():
                    print("Pebble connection lost. Exiting.")
                    break
                
                input()
                print("\n>>> [Enter] key pressed! Simulating button press...")
                self._toggle_session()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

    def shutdown(self):
        """Cleans up resources gracefully."""
        print("\nShutting down...")
        if self._session_future and not self._session_future.done():
            self._loop.call_soon_threadsafe(self._session_future.cancel)
        
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._pebble and self._pebble.connected:
            self._pebble.disconnect()
            print("Pebble disconnected.")

        # PyAudio termination
        pya.terminate()
        print("PyAudio terminated.")

        if hasattr(self, '_picam2') and self._picam2.started:
            self._picam2.stop()
            print("Camera stopped.")

def main():
    trigger = PebbleLiveSession()
    try:
        trigger.connect()
        trigger.run()
    except Exception as e:
        # Generic error handling now that Pebble connection is handled internally.
        if "Invalid input device" in str(e) or "No Default Input Device Available" in str(e):
            print("\nError: Could not find a valid microphone.")
            print("Please ensure a microphone is connected and configured.")
        elif "PortAudio" in str(e) or "portaudio.h" in str(e):
             print("\nError: PortAudio library not found or could not be initialized.")
             print("Please install it and its development headers with 'sudo apt-get install libportaudio2 portaudio19-dev' and ensure a microphone is connected.")
        else:
            print(f"\nA critical error occurred: {e}")
            traceback.print_exc()
    finally:
        trigger.shutdown()


if __name__ == "__main__":
    main() 