from dotenv import load_dotenv
import os
import requests
from typing import List


load_dotenv()
hf_token = os.getenv("HF_TOKEN")

class Embedding:
    def __init__(self, model_id: str, hf_token: str = hf_token) -> None:
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        self.headers = {"Authorization": f"Bearer {hf_token}"}
    
    def embed_query(self, query: List[str]) -> List[List[float]]:
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": query, "options":{"wait_for_model":True}})
        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}, {response.text}")
        return response.json()

