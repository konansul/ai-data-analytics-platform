# backend/api/main.py
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database.db import engine
from backend.database.models import Base

from backend.api import datasets, profiling, policy, cleaning, auth, visualization
app = FastAPI(title="Multi-Agent Excel Analytics API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
app.include_router(auth.router, prefix="/v1", tags=["auth"])
app.include_router(datasets.router, prefix="/v1", tags=["datasets"])
app.include_router(profiling.router, prefix="/v1", tags=["profiling"])
app.include_router(policy.router, prefix="/v1", tags=["policy"])
app.include_router(cleaning.router, prefix="/v1", tags=["cleaning"])
app.include_router(visualization.router, prefix="/v1", tags=["visualization"])