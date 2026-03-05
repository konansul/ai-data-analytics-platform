# 4CAST — AI-Assisted Data Cleaning, Visualization, and Forecasting Platform

4CAST is an end-to-end data analytics platform for transforming raw Excel and CSV files into clean, analysis-ready datasets with automated visualization and time-series forecasting. The system provides user-scoped ingestion of multi-sheet spreadsheets, dataset profiling, rule-based and LLM-assisted cleaning pipelines, persistent execution history, and reproducible exports. On top of cleaned data, 4CAST generates intelligent visualization suggestions and executes forecasting workflows using a signal, planning, execution architecture. The backend is built with FastAPI and PostgreSQL for durable storage of datasets and runs, while the Streamlit frontend delivers an interactive workflow for ingestion, cleaning, exploration, visualization, and forecasting — all fully decoupled through REST APIs.

## Requirements
	•	Docker, Docker Compose
	•	Python 3.9+ 

Core backend dependencies include:

- fastapi
- uvicorn
- sqlalchemy
- psycopg2
- pandas
- pyarrow
- xlsxwriter
- pydantic
- scikit-learn 
- statsmodels 
- prophet or fbprophet

Frontend:
- streamlit
- requests

All Python dependencies are listed in requirements.txt.


## Setup
### 1. Clone the repository
```bash
git clone https://github.com/konansul/ai-data-analytics-platform
cd ai-data-analytics-platform
```

### 2. Start PostgreSQL through Docker.
PostgreSQL is started via Docker Compose, this will start at port 5433.

```bash
docker compose up -d
```

### 3. Create and configure environment variables
This project uses environment variables for database connection, authentication, storage, and LLM access. Create a .env file in the project root directory (the same level as README.md):

```bash
touch .env
```

Add the following variables to the .env file, GEMINI_API_KEY is required only if LLM-assisted cleaning is enabled.

```bash
GEMINI_API_KEY=your_gemini_api_key_here

DATABASE_URL=postgresql+psycopg2://excel:excel@localhost:5433/excel_analytics

JWT_SECRET_KEY=long-random-string-at-least-32chars
JWT_EXPIRE_MINUTES=60
JWT_ALGORITHM=HS256
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### 4. Create virtual environment and install requirements.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 5. Start the backend API server
The backend will be available at: http://127.0.0.1:8000, swagger documentation: http://127.0.0.1:8000/docs

```bash
 uvicorn backend.api.main:app --reload --reload-dir backend --port 8000
```

### 6. Start Streamlit frontend 
In a separate terminal, start the Streamlit frontend, which will be available at http://localhost:8501

```bash
streamlit run frontend/main_streamlit.py
```
## Database Schema
The backend uses a PostgreSQL database to persist users, datasets, profiling results, and cleaning executions in a fully reproducible way. Each registered user has isolated ownership over their uploaded datasets, generated profiles, and cleaning runs. Uploaded Excel and CSV files are represented as datasets, where each Excel sheet is stored and processed independently. Profiling results are stored separately and capture structural and statistical signals used to guide cleaning decisions. Every execution of the cleaning pipeline is recorded as a cleaning run, including its status, generated artifacts, and reports. Multiple cleaning runs can exist for the same dataset, allowing history tracking and safe re-execution with different policies. The schema is designed to preserve the full lineage from raw data to cleaned outputs, even across user logins. Database records reference large artifacts stored on disk, combining transactional metadata with efficient file storage. The overall schema is depicted in the figure below:

![F5938AB8-064D-411D-9EDF-B6F9A4F1F09F_1_201_a](https://github.com/user-attachments/assets/6ededfca-e5a1-4666-9371-4c131b88f7f4)


## Project structure

```bash
ai-data-analytics-platform/
│
├── backend/                             # FastAPI backend
│   ├── api/                            # HTTP layer (routers + request/response models)
│   │   ├── auth.py                     # Authentication endpoints (register/login/me)
│   │   ├── datasets.py                # Dataset upload, preview, download
│   │   ├── cleaning.py                # Cleaning run orchestration endpoints
│   │   ├── profiling.py               # Dataset profiling endpoints
│   │   ├── policy.py                  # Cleaning policy suggestion endpoints
│   │   ├── visualization.py           # Visualization suggest/explain endpoints
│   │   ├── forecasting.py             # Forecast signals / planning / execution endpoints
│   │   ├── models.py                  # Shared API schemas (Pydantic)
│   │   └── main.py                    # FastAPI application entrypoint (router orchestration)
│   │
│   ├── app/                            # Core business logic (framework-agnostic)
│   │
│   │   ├── ingestion/                 # Excel / CSV ingestion
│   │   │   └── dataset_loader.py      # Multi-sheet Excel + CSV loader
│   │   │
│   │   ├── cleaning/                  # Deterministic + LLM cleaning system
│   │   │   ├── cleaning_agent/        # Cleaning policy engine
│   │   │   │   ├── cleaning_policy_agent.py   # Public cleaning planner API
│   │   │   │   ├── cleaning_policy_llm.py     # LLM-based policy generation
│   │   │   │   ├── cleaning_policy_rule_based.py
│   │   │   │   ├── cleaning_policy_utils.py   # Validation / coercion / safety
│   │   │   │   ├── llm_client.py              # Gemini wrapper
│   │   │   │   └── schemas.py                 # CleaningPlan schemas
│   │   │   │
│   │   │   └── cleaning_steps/        # 10-stage deterministic cleaning pipeline
│   │   │       ├── _01_normalize.py
│   │   │       ├── _02_trim_strings.py
│   │   │       ├── _03_standardize_missing.py
│   │   │       ├── _04_cast_types.py
│   │   │       ├── _05_encode_booleans.py
│   │   │       ├── _06_drop_rules.py
│   │   │       ├── _07_datetime_inference.py
│   │   │       ├── _08_deduplicate.py
│   │   │       ├── _09_outliers.py
│   │   │       └── _10_impute_missing.py
│   │   │
│   │   ├── forecasting/              # Time-series forecasting subsystem
│   │   │   ├── signals.py            # Forecast feasibility + candidate detection
│   │   │   ├── planning_agent.py     # LLM planning agent (WHAT + HOW to forecast)
│   │   │   ├── execution.py          # Deterministic model execution (ARIMA/Prophet/RF)
│   │   │   └── schemas.py            # Forecast domain models
│   │   │
│   │   ├── visualization/            # Visualization agent
│   │   │   ├── agent.py              # LLM visualization reasoning
│   │   │   ├── service.py            # Chart suggestion/explanation logic
│   │   │   └── schemas.py
│   │   │
│   │   └── profiling/                # Dataset profiling logic
│   │       └── profiling.py
│   │
│   ├── database/                     # Persistence layer
│   │   ├── db.py                     # SQLAlchemy session
│   │   ├── models.py                 # ORM models (users, datasets, runs)
│   │   ├── security.py               # JWT helpers
│   │   └── storage.py                # Local blob storage abstraction
│   │
│   ├── test_data/                    # Sample datasets
│   └── test_scripts/                # Experiments and backend tests
│
├── frontend/                         # Streamlit frontend
│   ├── main_streamlit.py            # UI entrypoint + tab orchestration
│   └── ui/
│       ├── tab_00_authentication.py
│       ├── tab_01_excel_upload.py
│       ├── tab_02_cleaning.py
│       ├── tab_03_signals.py
│       ├── tab_04 visualization.py
│       ├── tab_05_forecasting.py
│       ├── tab_07_save_all_files.py
│       ├── components.py
│       └── data_access.py           # Frontend and backend REST client
│
├── storage/                         # Persistent local storage (user-scoped)
│   └── users/
│       └── usr_<user_id>/
│           ├── datasets/            # Uploaded datasets (per sheet)
│           │   └── ds_<dataset_id>/
│           │       ├── raw.bin      # Original uploaded file bytes
│           │       ├── raw.parquet  # Parsed raw dataframe
│           │       └── current.parquet # Latest cleaned version
│           │
│           └── runs/                # Cleaning run history
│               └── run_<run_id>/
│                   ├── cleaned.parquet  # Cleaned dataframe
│                   ├── cleaned.xlsx     # Exported Excel
│                   └── report.json      # Full cleaning report + signals
│
├── docker-compose.yml               # PostgreSQL container
├── requirements.txt                # Python dependencies
├── README.md
└── LICENSE
```

## API Endpoints

### 1. Authentication

```bash
POST   /v1/auth/register     Register new user
POST   /v1/auth/login        Login
GET    /v1/auth/me           Get current user
POST   /v1/auth/logout       Logout (client-side token removal)
```

### 2. Datasets

```bash
GET    /v1/datasets                         List all datasets for the current user
POST   /v1/datasets                         Upload a new dataset (Excel or CSV)=
GET    /v1/datasets/{dataset_id}            Get dataset metadata
GET    /v1/datasets/{dataset_id}/preview    Preview dataset rows
GET    /v1/datasets/{dataset_id}/download   Download dataset (raw or cleaned)
```

### 3. Profiling

```bash
POST   /v1/profiling                  Run profiling
GET    /v1/profiling/{profile_id}     Get profiling report
```

### 4. Policy

```bash
POST   /v1/policy/suggest             Suggest cleaning policy
```

### 5. Cleaning

```bash
GET    /v1/cleaning/runs                          List user cleaning runs
POST   /v1/cleaning/runs                          Run cleaning
GET    /v1/cleaning/runs/{run_id}                 Get run status
GET    /v1/cleaning/runs/{run_id}/report          Get run report
GET    /v1/cleaning/runs/{run_id}/artifacts/{name} Download artifact
DELETE /v1/cleaning/runs/{run_id}                 Delete run
```

### 6. Visualization

```bash
POST /v1/visualization/suggest   Suggest visualizations for a dataset
POST /v1/visualization/explain  Explain a specific chart configuration
```

### 7. Forecasting

```bash
POST /v1/forecast/signals   Generate forecasting signals (datetime, targets, feasibility)
POST /v1/forecast/plan      Create forecast plan (LLM or fallback planner)
POST /v1/forecast/run       Execute forecasting models and return predictions
```
Notes:  Multi-sheet Excel files are ingested as separate datasets; CSV files are treated as single-sheet inputs. Data cleaning can run in deterministic mode or with optional LLM assistance, and all runs are persisted per user. Visualization and time-series forecasting operate on cleaned datasets using generated signals and planning agents. Frontend (Streamlit) and backend (FastAPI) are fully decoupled and run as independent services.