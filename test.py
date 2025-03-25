
# import pydub_fix  # must be the first import
# Please install OpenAI SDK first: `pip3 install openai`
from openai import OpenAI

import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

from pydub import AudioSegment
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"

from generator import load_csm_1b
import torchaudio

# Initialize the OpenAI client
client = OpenAI(api_key="sk-87b6197ceca94a52b1d1ffd163a14876", base_url="https://api.deepseek.com")

user_input = input("Ask a question: ")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_input},
    ],
    stream=False
)


generator = load_csm_1b(device="cpu")  # or "cuda" if you have GPU

Video_Generation = response.choices[0].message.content
print(Video_Generation)

audio = generator.generate(
    text=Video_Generation,
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)


torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
