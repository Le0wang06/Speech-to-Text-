from generator import load_csm_1b
import torchaudio

# Load the generator onto your GPU
generator = load_csm_1b(device="cuda")

# Generate audio from text
audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

# Save it
torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
print("✅ Audio saved to output.wav")
