
# import pydub_fix  # must be the first import

import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

from pydub import AudioSegment
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"


from generator import load_csm_1b
import torchaudio

generator = load_csm_1b(device="cpu")  # or "cuda" if you have GPU

user_input = input("Enter the text: ")
audio = generator.generate(
    text=user_input,
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
