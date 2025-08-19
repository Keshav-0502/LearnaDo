import whisper

model = whisper.load_model("medium", device="cuda")
result = model.transcribe("data/uploads/hhh.mp3")
print(result["text"])
