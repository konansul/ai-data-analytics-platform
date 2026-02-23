# backend/api/main.py
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database.db import engine
from backend.database.models import Base

from backend.api import datasets, profiling, policy, cleaning, auth, visualization, forecasting

app = FastAPI(title="4CAST — AI-Assisted Data Cleaning, Visualization, and Forecasting Platform",
              version="1.0",
              description = "4CAST is an end-to-end data analytics platform for transforming raw Excel and CSV files into clean, analysis-ready datasets with automated visualization and time-series forecasting. The system provides user-scoped ingestion of multi-sheet spreadsheets, dataset profiling, rule-based and LLM-assisted cleaning pipelines, persistent execution history, and reproducible exports. On top of cleaned data, 4CAST generates intelligent visualization suggestions and executes forecasting workflows using a signal, planning, execution architecture. The backend is built with FastAPI and PostgreSQL for durable storage of datasets and runs, while the Streamlit frontend delivers an interactive workflow for ingestion, cleaning, exploration, visualization, and forecasting — all fully decoupled through REST APIs.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
app.include_router(auth.router, prefix="/v1", tags=["Authentication"])
app.include_router(datasets.router, prefix="/v1", tags=["Datasets"])
app.include_router(profiling.router, prefix="/v1", tags=["Profiling"])
app.include_router(policy.router, prefix="/v1", tags=["Policy"])
app.include_router(cleaning.router, prefix="/v1", tags=["Cleaning"])
app.include_router(visualization.router, prefix="/v1", tags=["Visualization"])
app.include_router(forecasting.router, prefix="/v1", tags=["Forecasting"])