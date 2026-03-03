class ExplainAgent:
    def __init__(self, llm_utils):
        self.llm = llm_utils

    def explain(self, plot_title: str, axis_info: str) -> str:
        prompt = f"""
You are a data analyst explaining a chart to a business user.

CHART TITLE: {plot_title}
AXIS INFO:   {axis_info}

Write a concise, plain-language explanation (2–4 sentences) of:
1. What this chart is showing.
2. What patterns or insights a viewer should look for.

Do not make up specific numbers. Focus on what the chart type and axes reveal analytically.
Return only the explanation text — no bullet points, no JSON, no preamble.
"""
        try:
            return self.llm.complete(prompt).strip()
        except Exception as e:
            return f"Could not generate explanation: {e}"