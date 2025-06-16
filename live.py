#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai opencv-python pyaudio pillow mss
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

For Raspberry Pi, ensure the camera is enabled and configured to work with
OpenCV. You may need to enable the legacy camera support via `raspi-config`
for `cv2.VideoCapture(0)` to work.

Important: **Use headphones or a separate speaker**. This script uses the
system default audio input and output, which often won't include echo
cancellation. So to prevent the model from interrupting itself it is
important that you use headphones or a speaker that is not close to the
microphone.

## Run

To run the script:

```
python live.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera".
"""

import asyncio
import base64
import io
import os
import sys
import traceback
import subprocess
from contextlib import contextmanager

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai

@contextmanager
def suppress_stderr():
    """A context manager that redirects stderr to devnull"""
    with open(os.devnull, "w") as fnull:
        original_stderr = sys.stderr
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stderr = original_stderr

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-1.5-flash"

DEFAULT_MODE = "camera"

client = genai.Client(http_options={"api_version": "v1beta"})

CONFIG = {"response_modalities": ["AUDIO"]}

with suppress_stderr():
    pya = pyaudio.PyAudio()


def check_bluetooth_audio():
    """Checks if a Bluetooth audio device is connected."""
    print("Checking for Bluetooth audio device...")
    try:
        # Use pactl to list sinks and check for bluez (common for bluetooth devices)
        result = subprocess.run(
            "pactl list sinks", shell=True, capture_output=True, text=True
        )

        if result.returncode != 0:
            print("Could not run `pactl`. Is PulseAudio or PipeWire running?")
            print("Assuming no Bluetooth audio device is connected.")
            return False

        if "bluez" in result.stdout.lower():
            print("Bluetooth audio device found.")
            return True
        else:
            print("No Bluetooth audio sink found. Please connect a Bluetooth audio device.")
            return False
    except FileNotFoundError:
        print(
            "Could not check for Bluetooth audio devices. `pactl` command not found."
        )
        print(
            "Please ensure 'pactl' is available (from pulseaudio-utils) and a Bluetooth speaker is connected."
        )
        return False


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.output_modality = "AUDIO"

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.audio_stream = None

    async def send_text(self):
        print('Enter "1" for audio output (default), "2" for text output.')
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break

            if text == "1":
                if self.output_modality != "AUDIO":
                    if await asyncio.to_thread(check_bluetooth_audio):
                        print("Switched to AUDIO output.")
                        self.output_modality = "AUDIO"
                    else:
                        print(
                            "Failed to switch to AUDIO output. No Bluetooth audio device found."
                        )
                else:
                    print("Audio output is already enabled.")
                continue

            if text == "2":
                if self.output_modality != "TEXT":
                    print("Switched to TEXT only output.")
                    self.output_modality = "TEXT"
                else:
                    print("Text only output is already enabled.")
                continue

            config = {"response_modalities": [self.output_modality]}
            await self.session.send(input=text or ".", end_of_turn=True, config=config)

    def _get_frame(self, cap):
        # Read the frame
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        if not cap.isOpened():
            print("Error: Cannot open camera.", file=sys.stderr)
            print(
                "If you are on a Raspberry Pi, please ensure the camera is enabled "
                "and the legacy camera stack is activated (`sudo raspi-config`).",
                file=sys.stderr,
            )
            return

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        if self.output_modality == "AUDIO":
            if not check_bluetooth_audio():
                print(
                    "Bluetooth audio device not found. Defaulting to TEXT output.",
                    file=sys.stderr,
                )
                self.output_modality = "TEXT"
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            traceback.print_exception(EG)
        finally:
            if self.audio_stream:
                self.audio_stream.close()


if __name__ == "__main__":
    input("Press Enter to start the Gemini Live session...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
    finally:
        pya.terminate()
