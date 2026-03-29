from fastapi import FastAPI

from app import routes


app = FastAPI(title="RAG AI Assistant API", version="0.1.0")

app.include_router(routes.router)
