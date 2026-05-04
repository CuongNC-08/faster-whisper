import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import logging
app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")
MODEL_PATH = "./models/latest" 
model = WhisperModel(MODEL_PATH, device="cuda", compute_type="float16")

@app.post("/transcribe")
async def transcribe_api(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    logger.info(f"--- Bắt đầu xử lý file: {file.filename} ---")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        logger.info(f"Đang tạo file tạm cho {file.filename}")
        segments, _ = model.transcribe(temp_path, language="vi", beam_size=5)
        
        text = " ".join([s.text for s in segments]).strip()
        logger.info("Chuyển đổi âm thanh thành văn bản thành công.")
        return {
            "filename": file.filename, 
            "transcription": text
        }
    
    except Exception as e:
        logger.error(f"Lỗi xảy ra khi xử lý file {file.filename}: {str(e)}")
        return {"error": str(e)}
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

