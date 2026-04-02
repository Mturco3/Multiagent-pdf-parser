import os
import requests

class SyntacticImproverAgent:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        # Default to GROQ, fallback to GOOGLE
        self.model = "llama3-70b-8192"  # Example Groq model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = api_keys.get("GROQ") or api_keys.get("GOOGLE")

    def improve(self, text):
        prompt = (
            "Improve the following text for sentence connectivity and syntactic flow. "
            "Do not change the meaning.\n\n" + text
        )
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096
        }
        response = requests.post(self.api_url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
