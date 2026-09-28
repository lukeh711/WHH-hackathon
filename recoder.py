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
    print("stopped")
    segments, info = model.transcribe("audio/out.mp3", beam_size=5, language="en", condition_on_previous_text=False)
    text = ""
    for segment in segments:
        text += segment.text
    return text.strip()

def wake_word(text):
    text = text.split(" ")
    if "wake" in text:
        return True
    else:
        return False
text = record_audio(seconds).lower()
print(text)
print(wake_word(text))
