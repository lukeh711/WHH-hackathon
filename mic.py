from faster_whisper import WhisperModel

model_size = "tiny.en"
print("modle")
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("loaded")
segments, info = model.transcribe("1.mp3", beam_size=5, language="en", condition_on_previous_text=False)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))