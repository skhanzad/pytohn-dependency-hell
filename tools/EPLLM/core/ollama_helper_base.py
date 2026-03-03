# Base file for Ollama helper (standalone EPLLM core)
from langchain_community.chat_models import ChatOllama
import os


class OllamaHelperBase:
    def __init__(self, base_url="http://localhost:11434", model='phi3:medium', temp=0.7, logging=False) -> None:
        self.logging = logging
        if 'gpt' in model:
            from dotenv import load_dotenv
            from langchain_openai import ChatOpenAI
            load_dotenv()
            OPENAI_KEY = os.getenv('OPENAI_KEY')
            self.model = ChatOpenAI(model=model, api_key=OPENAI_KEY, temperature=temp)
        else:
            # Low temperature (default 0.2) reduces hallucination for structured JSON output
            self.model = ChatOllama(base_url=base_url, model=model, format="json", temperature=temp)

    def read_python_file(self, file):
        with open(file, 'r') as f:
            data = f.read().replace('\n', '')
        return data
