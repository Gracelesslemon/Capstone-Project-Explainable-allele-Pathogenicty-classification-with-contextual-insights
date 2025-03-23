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
        self.llm = None
        self.app = None