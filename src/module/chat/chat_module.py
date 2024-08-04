import os
from fastapi import APIRouter,HTTPException
from service.retriever.retriever_service import kgRetriever,modelClientConnService


MODEL = os.getenv("MODEL")
retrieverEngine = kgRetriever()

class ChatModule:
    router = APIRouter(prefix="/chat",tags=["chat"])

    @router.post('/retriever')
    async def Retriever(prompt:str):

        retrieveText = retrieverEngine.kgRetrieverClient.custom_retrieve(prompt).split("Here are some facts extracted from the provided text:")
        query = "\n".join(retrieveText)
        query = "นี่คือข้อมูลเบื้องต้นจาก knowledge graph" + query

        print(query)
        
        try:
            return modelClientConnService.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": f"{query}\n\nจากข้อมูลใน knowledge graph\n\n{prompt}"}],
                stream=False,
            ).choices[0].message.content
            
            
                
        except Exception as e:
            print(f"Error : {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    


chatModule = ChatModule()