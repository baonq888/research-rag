from fastapi import APIRouter, UploadFile, Query
from services.qa_service import load_and_index_pdf, answer_query

router = APIRouter()

@router.post("/upload")
async def upload_pdf(file: UploadFile):
    file_path = f"./data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = load_and_index_pdf(file_path)
    return {"message": "PDF processed", "stats": result}

@router.get("/query")
def query_pdf(q: str = Query(..., alias="question")):
    return answer_query(q)