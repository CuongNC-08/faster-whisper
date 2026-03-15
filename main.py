import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel

app = FastAPI()

# Khởi tạo model (nên để bên ngoài để load 1 lần duy nhất)
MODEL_PATH = "./models/latest" 
model = WhisperModel(MODEL_PATH, device="cuda", compute_type="float16")

@app.post("/transcribe")
async def transcribe_api(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        segments, _ = model.transcribe(temp_path, language="vi", beam_size=5)
        
        text = " ".join([s.text for s in segments]).strip()
        
        return {
            "filename": file.filename, 
            "transcription": text
        }
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)