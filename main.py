import os
import shutil
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI()

MODEL_PATH = "./models/latest" 
print(f"⚠️ Chuyển sang CPU (INT8 tối ưu)...")
model = WhisperModel(MODEL_PATH, device="cuda", compute_type="float16")
# model = WhisperModel("tiny", device="cuda", compute_type="float16")

@app.post("/transcribe")
async def transcribe_api(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        segments, _ = model.transcribe(temp_path, language="vi", beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        
        return {"filename": file.filename, "transcription": text}
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)