import os
from openai import OpenAI
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaOpenAIEmbedding


from fastapi import HTTPException

MODEL = os.getenv("MODEL")

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_ENDPOINT = os.getenv("AZURE_OPENAI_API_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION_ID = os.getenv("OPENAI_ORGANIZATION_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

API_VERSION = os.getenv("API_VERSION")

class ModelClientConnService:
    
    def __init__(self) -> None:
        if not AZURE_API_KEY:
            raise HTTPException(status_code=500, detail="API key is missing")
        
        self.lamaModelClient = LlamaOpenAI(
            model=MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.0,
            api_version=API_VERSION
        )

        self.lamaModelClient = LlamaAzureOpenAI(
                model=MODEL,
                api_key=AZURE_API_KEY,
                api_version=API_VERSION,
                azure_endpoint=AZURE_OPENAI_API_ENDPOINT,
                engine=AZURE_OPENAI_DEPLOYMENT_NAME,
                temperature=0.0,
                reuse_client=True,
            )
            

        self.embedModel = LlamaOpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key=OPENAI_API_KEY,
            api_version=API_VERSION
        )

        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            organization=OPENAI_ORGANIZATION_ID,
            project=OPENAI_PROJECT_ID,
            timeout=None,
        )



modelClientConnService = ModelClientConnService()