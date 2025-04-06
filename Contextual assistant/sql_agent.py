from typing import List, Any, Literal, TypedDict, Dict, Optional
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace
from transformers import pipeline
from dotenv import load_dotenv
import duckdb
import os
import json
from langsmith import traceable

# Load environment variables
load_dotenv("sql_agent.env")

# AgentState as a TypedDict
class AgentState(TypedDict):
    question: str
    sql_query: str
    query_result: str
    query_rows: list
    attempts: int
    relevance: str
    sql_error: bool
    harmful: bool  

class SQLAgent:
    """
    A SQL Agent that converts natural language questions to SQL queries
    and executes them on a DuckDB database.
    """
    
    def __init__(self, db_path: str = None, llm_provider: str = None):
        """Initialize the SQL Agent with database connection and LLM."""
        self.db_path = db_path or os.getenv("DB_PATH_sql", r"C:\Users\mathew\Desktop\Capstone\Datasets\Capstone_data_sql.duckdb")
        self.conn = duckdb.connect(self.db_path)
        # Placeholder for LLM and app - will be implemented by other team members
        self.llm = self.get_llm(llm_provider)
        self.app = None
        
    def get_llm(self, provider: str = None):
        """Initializes and returns an LLM instance based on the specified provider."""
        provider = provider or os.getenv("LLM_PROVIDER", "gemini").lower()

        if provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash"),
                google_api_key=os.getenv("GEMINI_KEY")
            )
        elif provider == "perplexity":
            return ChatOpenAI(
                model=os.getenv("PPLX_MODEL", "sonar-pro"),
                api_key=os.getenv("PERPLEXITY_API_KEY"),
                base_url=os.getenv("PPLX_BASE_URL", "https://api.perplexity.ai/chat/completions")
            )
        elif provider == "huggingface":
            return HuggingFaceHub(
                repo_id=os.getenv("HF_MODEL", "meta-llama/Llama-3-8b-chat-hf"),
                huggingfacehub_api_token=os.getenv("HF_TOKEN")
            )
        elif provider == "local-hf":
            pipe = pipeline(
                "text-generation",
                model=os.getenv("LOCAL_HF_MODEL", "meta-llama/Llama-3-8b-chat-hf"),
                device_map="auto"
            )
            return ChatHuggingFace(pipeline=pipe)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")