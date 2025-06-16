"""
This script runs on a Raspberry Pi and starts a Gemini Live session, streaming
video from the Pi Camera and audio from a USB microphone.

Pressing 'Enter' in the terminal will start the session.

It checks for a connected Bluetooth audio device. If one is found, Gemini's
responses will be streamed as audio. Otherwise, they will be printed as text.

This script is a combination of the reference pebble_camera_trigger_button.py
and the Gemini Live API example provided.

--- REQUIREMENTS ---
You will need to install the following libraries:
pip install google-generativeai "google-generativeai<0.7" pillow
pip install picamera2 pyaudio

You may need to install portaudio for PyAudio:
sudo apt-get install portaudio19-dev

Ensure the GOOGLE_API_KEY environment variable is set.
export GOOGLE_API_KEY="your-api-key"
"""
import asyncio
import base64
import io
import os
import sys
import subprocess
import traceback

try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("Could not import picamera2. Run 'pip install picamera2'.")

try:
    import PIL.Image
except ImportError:
    sys.exit("Could not import Pillow. Run 'pip install pillow'.")

try:
    from google import genai
except ImportError:
    sys.exit("Could not import google.genai. Run 'pip install google-generativeai \"google-generativeai<0.7\"'.")

try:
    import pyaudio
except ImportError:
    sys.exit("Could not import pyaudio. Run 'pip install pyaudio'. You may also need 'sudo apt-get install portaudio19-dev'")

# --- Compatibility for older Python versions ---
if sys.version_info < (3, 11, 0):
    try:
        import taskgroup
        import exceptiongroup
        asyncio.TaskGroup = taskgroup.TaskGroup
        asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup
    except ImportError:
        sys.exit("For Python < 3.11, please install taskgroup and exceptiongroup: 'pip install taskgroup exceptiongroup'")

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")

# This model is specified in the user's example for Live API.
MODEL_ID = "models/gemini-2.0-flash-live-001"

# Audio settings from the example
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024


def check_for_bluetooth_audio():
    """Checks if a Bluetooth audio sink is available using pactl."""
    print("Checking for Bluetooth audio device...")
    try:
        output = subprocess.check_output(['pactl', 'list', 'sinks'], text=True)
        if 'bluez' in output:
            print("Bluetooth audio device found.")
            return True
        else:
            print("No Bluetooth audio device found. Defaulting to text output.")
            return False
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Warning: 'pactl' command failed: {e}. Assuming no Bluetooth audio.")
        print("Audio output will be disabled.")
        return False


class GeminiLiveSession:
    """
    Manages the connection to Gemini Live, capturing and streaming video and audio.
    """
    def __init__(self):
        if not API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        self.client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1beta"})
        self.picam2 = Picamera2()
        self.pya = pyaudio.PyAudio()

        self.has_bt_audio = check_for_bluetooth_audio()
        self.config = {"response_modalities": ["AUDIO"] if self.has_bt_audio else ["TEXT"]}

        self.session = None
        self.audio_in_queue = None  # For received audio
        self.realtime_out_queue = None # For sending audio/video

        self.audio_stream_in = None

    def _setup_camera(self):
        """Initializes and configures the PiCamera."""
        print("Initializing camera...")
        video_config = self.picam2.create_video_configuration(main={"size": (800, 600), "format": "RGB888"})
        self.picam2.configure(video_config)
        self.picam2.start()
        print("Camera ready.")

    def _get_jpeg_frame(self):
        """Captures a frame and encodes it as JPEG."""
        frame_array = self.picam2.capture_array()
        img = PIL.Image.fromarray(frame_array)
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        
        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def _stream_video_frames(self):
        """Continuously captures frames and puts them in the output queue."""
        while True:
            frame = await asyncio.to_thread(self._get_jpeg_frame)
            await self.realtime_out_queue.put(frame)
            await asyncio.sleep(1.0) # Send one frame per second

    async def _stream_audio_chunks(self):
        """Continuously captures audio and puts it in the output queue."""
        mic_info = self.pya.get_default_input_device_info()
        print(f"Using audio input device: {mic_info['name']}")
        
        self.audio_stream_in = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        while True:
            # The exception_on_overflow=False is important on slower devices
            # to avoid crashing if the buffer overflows.
            data = await asyncio.to_thread(self.audio_stream_in.read, CHUNK_SIZE, exception_on_overflow=False)
            await self.realtime_out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def _send_realtime_data(self):
        """Sends data from the output queue to the Gemini API."""
        while True:
            msg = await self.realtime_out_queue.get()
            await self.session.send(input=msg)
    
    async def _handle_user_text_input(self):
        """Handles text input from the user to send to the session."""
        while True:
            text = await asyncio.to_thread(input, "Enter text to send (or 'q' to quit): ")
            if text.lower() == "q":
                break
            await self.session.send(input=text, end_of_turn=True)
        # Signal shutdown
        raise asyncio.CancelledError("User requested exit")

    async def _receive_responses(self):
        """Receives responses from Gemini and processes them."""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    if self.has_bt_audio:
                        self.audio_in_queue.put_nowait(data)
                if text := response.text:
                    print(f"Gemini: {text}", end="", flush=True)

            # Empty the audio queue on turn completion to prevent stale audio
            # if the model was interrupted.
            if self.has_bt_audio:
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()
            else:
                 print() # for a clean newline after text output

    async def _play_audio_responses(self):
        """Plays back received audio chunks."""
        if not self.has_bt_audio:
            return # Don't run this task if we don't have audio output

        stream_out = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream_out.write, bytestream)

    async def _start_session(self):
        """Sets up and runs the concurrent tasks for the session."""
        try:
            async with self.client.aio.live.connect(model=MODEL_ID, config=self.config) as session, \
                       asyncio.TaskGroup() as tg:
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.realtime_out_queue = asyncio.Queue(maxsize=5)

                # Start all the background tasks
                tg.create_task(self._send_realtime_data())
                tg.create_task(self._stream_audio_chunks())
                tg.create_task(self._stream_video_frames())
                tg.create_task(self._receive_responses())
                tg.create_task(self._play_audio_responses())
                
                # The input handler runs in the foreground of the task group
                await self._handle_user_text_input()

        except asyncio.CancelledError:
            print("\nSession cancelled.")
        except asyncio.ExceptionGroup as eg:
            print("\nAn error occurred during the session:")
            traceback.print_exception(eg)

    def run(self):
        """Main entry point for the class."""
        print("\n--- Gemini Live on Raspberry Pi ---")
        input("Press Enter to start the live session...")
        
        self._setup_camera()
        
        try:
            asyncio.run(self._start_session())
        except KeyboardInterrupt:
            print("\nSession interrupted by user.")
        finally:
            self._shutdown()

    def _shutdown(self):
        """Cleans up all resources."""
        print("\nShutting down...")
        if self.audio_stream_in and self.audio_stream_in.is_active():
            self.audio_stream_in.stop_stream()
            self.audio_stream_in.close()
        
        if self.pya:
            self.pya.terminate()
            
        if self.picam2 and self.picam2.started:
            self.picam2.stop()
            print("Camera stopped.")
        
        print("Shutdown complete.")


def main():
    """Main function to run the script."""
    try:
        live_session = GeminiLiveSession()
        live_session.run()
    except Exception as e:
        print(f"A critical error occurred: {e}")

if __name__ == "__main__":
    main()



