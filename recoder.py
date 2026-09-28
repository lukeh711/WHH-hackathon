import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

model_size = "tiny.en"
print("modle")
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("loaded")

fs = 44100
seconds = 5
sd.default.device = (4, None)

def record_audio(time):
    print("recorging")
    data = sd.rec(int(time * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write("audio/out.mp3", data, fs)

    segments, info = model.transcribe("audio/out.mp3", beam_size=5, language="en", condition_on_previous_text=False)
    print(segments)
    text = []
    for segment in segments:
        text.append(segment.text)
    return text

print(record_audio(seconds))