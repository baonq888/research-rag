import os
from fastapi import APIRouter, UploadFile, Query
from src.api.services.qa_service import QAService

qa_service = QAService()
router = APIRouter()

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile):
    # Save file into /data/
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = qa_service.load_and_index_pdf(file_path)
    return {"message": "PDF processed", "stats": result}

@router.get("/query")
def query_pdf(q: str = Query(..., alias="question")):
    return qa_service.answer_query(q)