import os
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from module.document.document_module import documentModule
from module.chat.chat_module import chatModule
import nest_asyncio

nest_asyncio.apply()

load_dotenv('../.env')

class AppModule:
    def __init__(self) -> None:
        pass
        self.HOST=os.getenv("HOST")
        self.PORT=os.getenv("PORT")

        self.app = FastAPI()
        origins = ["*"]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app.include_router(documentModule.router)
        self.app.include_router(chatModule.router)

        if __name__ == "__main__":
            import uvicorn
            try:
                uvicorn.run(self.app, host=self.HOST, port=int(self.PORT), log_level="info")
            except Exception as e:
                print(f"app runtime error : {e}")
                raise SystemExit

AppModule()

