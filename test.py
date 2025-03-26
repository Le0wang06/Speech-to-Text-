
# import pydub_fix  # must be the first import
# Please install OpenAI SDK first: `pip3 install openai`
from openai import OpenAI

import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

from pydub import AudioSegment
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"

from generator import load_csm_1b
import torchaudio
import torch._dynamo

from pydub import AudioSegment


# 🚫 Fully disable TorchDynamo + Inductor + Triton
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"

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


generator = load_csm_1b(device="cuda")  # or "cuda" if you have GPU

Video_Generation = response.choices[0].message.content
print(Video_Generation)

audio = generator.generate(
    text=Video_Generation,  
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)

audio = AudioSegment.from_file("output.wav")

audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)

audio.export("fixed.wav", format="wav")

from playsound import playsound
playsound("fixed.wav")
