from fastapi import APIRouter,UploadFile,File
from llama_index.core import Document
from module.data_extractor.data_extractor import DataExtractor

UPLOAD_DIR = '../file-uploaded'

class DocumentModule:
    router = APIRouter(prefix='/document',tags=["document"])

    @staticmethod
    @router.post("/uploadfile/")
    async def upload_file(file: UploadFile = File(...)):
        dataExtractor = DataExtractor()
        contents = await file.read()
        text = contents.decode("utf-8")
        dataExtractor.extractData([Document(text=text)])
        return "add knowledge successful"

            
documentModule = DocumentModule()