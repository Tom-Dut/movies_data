import os
from langchain_ollama import ChatOllama
#from dotenv import load_dotenv

os.environ["HTTP_PROXY"] = "http://cache-etu.univ-artois.fr:3128"
os.environ["HTTPS_PROXY"] = "http://cache-etu.univ-artois.fr:3128"

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

PROJECT_PATH = "/home/tom/Documents/BUT INFO 3/R5.10/Projet"

IN_FOLDER = os.path.join(PROJECT_PATH, "movies_data")
OUT_FOLDER = os.path.join(PROJECT_PATH, "out")
LIB_FOLDER = os.path.join(PROJECT_PATH, "lib")

API_KEY = "sk-4ccbe02f4f7146ddb269c8e057a61b54"

#load_dotenv()
#os.environ["LANGCHAIN_TRACING_V2"]="true"
#os.environ["LANGCHAIN_API_KEY"]=API_KEY
os.environ["OLLAMA_API_KEY"]=API_KEY


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "nnnnnn4j"

OLLAMA_MODEL = "qwen3.5:9b"
#OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = "http://172.18.38.116/ollama" # Default Ollama URL
#OLLAMA_BASE_URL = "http://localhost:11434"

#OLLAMA_TEXT_SIMILARITY_MODEL = "all-minilm:l6-v2"
#OLLAMA_TEXT_SIMILARITY_MODEL = "qwen3.5:9b"
OLLAMA_TEXT_SIMILARITY_MODEL = "bge-m3:latest"

MOVIE_QA_MODEL = OLLAMA_MODEL
MOVIE_CYPHER_MODEL = OLLAMA_MODEL
#MOVIE_CYPHER_MODEL = "tomasonjo/llama3-text2cypher-demo:8b_4bit"
MOVIE_AGENT_MODEL = OLLAMA_MODEL


chat_model = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, client_kwargs={
                "headers": {
                    "Authorization": "Bearer "+API_KEY
                }
            })
