from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from backend.app.profiling.profiling import profile_dataframe
from backend.app.cleaning.cleaning_agent.cleaning_policy_agent import build_cleaning_plan

df = pd.read_csv("/Users/konansul/Desktop/github/ml-projects/02 – Logistic Regression/data/framingham.csv")

pre = profile_dataframe(df)

plan = build_cleaning_plan(
    pre,
    use_llm=True,
    model="gemini-2.5-flash",
)

print(plan.to_dict())