from dotenv import load_dotenv
load_dotenv()

from backend.app.cleaning.cleaning_agent.llm_client import LLMClient

client = LLMClient.from_env()

response = client.complete(
    "Return ONLY JSON: {\"status\": \"ok\", \"source\": \"gemini\"}"
)

print(response)