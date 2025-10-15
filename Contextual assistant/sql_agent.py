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
from IPython.display import Image

# Load environment variables
load_dotenv(".env")

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
        self.db_path = db_path or os.getenv("DB_PATH_sql")
        self.conn = duckdb.connect(self.db_path)
        self.llm = self.get_llm(llm_provider)
        self.app = self._build_workflow()

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

    def get_database_schema(self):
        """Fetches all tables and their column info from the DuckDB database."""
        schema = ""
        tables = self.conn.execute("SHOW TABLES").fetchall()
        for (table_name,) in tables:
            schema += f"Table: {table_name}\n"
            columns = self.conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            for col in columns:
                col_name, col_type, pk = col[1], col[2], col[5]
                col_type_str = f"{col_type}"
                if pk:
                    col_type_str += ", Primary Key"
                schema += f"- {col_name}: {col_type_str}\n"
            schema += "\n"
        return schema

    # Pydantic models for structured output
    class CheckRelevance(BaseModel):
        relevance: Literal["relevant", "not_relevant"]

    class ConvertToSQL(BaseModel):
        sql_query: str = Field(description="The SQL query.")

    class RewrittenQuestion(BaseModel):
        question: str

    @traceable
    def check_relevance(self, state: AgentState, config=None):
        """Checks if the user's question is relevant to the database schema."""
        question = state["question"]
        schema = self.get_database_schema()
        system = f"""You must decide if the question is related to the database schema.
        - Only respond 'relevant' if the question explicitly refers to columns, tables, or concepts present in the schema.
        - If a table or column name is slightly misspelled but strongly resembles one in the schema, still consider it relevant.
        Respond with only 'relevant' or 'not_relevant'.
        Schema:
        {schema}
        Extra information :  
        - The database only houses germline.
        - The database only houses SNP's
        - allele table has 2 columns that each house multiple id's. this a comphrehensive list of that : PhenotypeIDS: ['EFO', 'Gene', 'Human Phenotype Ontology', 'MONDO', 'MeSH', 'MedGen', 'OMIM', 'Orphanet']
        OtherIDs: ['BRCA1-HCI', 'BTK @ LOVD', 'Breast Cancer Information Core (BIC) (BRCA1)', 'Breast Cancer Information Core (BIC) (BRCA2)', 'British Heart Foundation', 'COL7A1 database', 'COSMIC', 'ClinGen', 'ClinVar', 'GUCY2C database', 'HBVAR', 'LDLR-LOVD', 'LOVD 3', 'Leiden Muscular Dystrophy (CAV3)', 'Leiden Muscular Dystrophy (CHRND)', 'Leiden Muscular Dystrophy (CHRNE)', 'Leiden Muscular Dystrophy (DAG1)', 'Leiden Muscular Dystrophy (DPM3)', 'Leiden Muscular Dystrophy (MYL2)', 'Leiden Muscular Dystrophy (MYL3)', 'Leiden Muscular Dystrophy (MYOZ2)', 'Leiden Muscular Dystrophy (MYPN)', 'Leiden Muscular Dystrophy (PDLIM3)', 'Leiden Muscular Dystrophy (TNNT1)', 'Leiden Muscular Dystrophy (TNNT3)', 'Leiden Muscular Dystrophy (TPM1)', 'MSeqDR', 'MYBPC3 homepage - Leiden Muscular Dystrophy pages', 'NAA10 @ LOVD', 'OMIM', 'PharmGKB Clinical Annotation', 'RettBASE (CDKL5)', 'Tuberous sclerosis database (TSC1)', 'Tuberous sclerosis database (TSC2)', 'UniProtKB', 'UniProtKB/Swiss-Prot', 'dbRBC', 'dbVar']
        - MC stands for Molecular consequence
        """

        human = f"Question: {question}"
        check_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", human),
        ])
        structured_llm = self.llm.with_structured_output(self.CheckRelevance)
        relevance_checker = check_prompt | structured_llm

        try:
            rel = relevance_checker.invoke({})
            state["relevance"] = rel.relevance
        except ValidationError:
            state["relevance"] = "not_relevant"
            print("Error at check relevance")
        return state

    @traceable
    def convert_nl_to_sql(self, state: AgentState, config=None):
        """Converts a natural language question into a valid SQL query."""
        question = state["question"]
        schema = self.get_database_schema()
        system = f"""
        You are an assistant that converts natural language questions into SQL queries for a DuckDB database.

        Schema:
        {schema}

        CRITICAL RULE:
        - All TEXT values MUST use single quotes ('value'). Never use double quotes ("value").
        - This applies regardless of user input formatting.
        - Double quotes are ONLY for identifiers (table or column names).

        Correct:
        SELECT * FROM allele WHERE Category = 'within single gene';

        Incorrect:
        SELECT * FROM allele WHERE Category = "within single gene";   -- NEVER DO THIS

        ======================
        General Rules:
        ======================
        - Use the correct SQL literal type based on the column datatype provided in the schema.
        * Integers/floats: no quotes.
        * TEXT columns: always use single quotes.
        * Double quotes are reserved only for identifiers.
        - Return only the SQL query, no explanations.

        ======================
        Allele Table Guidelines:
        ======================
        1. **ID Lookups**
        - For all ID lookups except AlleleID, GeneID, and RS# (dbSNP), prepend the appropriate key to the numeric value.
            Example:
            ```sql
            SELECT * FROM allele WHERE HGNCID = 'HGNC:28986';
            ```

        2. **ClinicalSignificance**
        - Values are separated by `;`.
        - Exact match when user requests strictly 'Pathogenic':
            ```sql
            SELECT * FROM allele WHERE ClinicalSignificance = 'Pathogenic';
            ```
        - Otherwise, partial match:
            ```sql
            SELECT * FROM allele WHERE ClinicalSignificance LIKE '%Pathogenic%';
            ```

        3. **TestedInGTR**
        - Stores 'Y' for yes and 'N' for no.
            Example:
            ```sql
            SELECT * FROM allele WHERE TestedInGTR = 'Y';
            ```

        4. **LastEvaluated**
        - Uses format 'Mon DD, YYYY' (e.g., 'Jun 29, 2015').
            Example:
            ```sql
            SELECT * FROM allele WHERE LastEvaluated = 'Jun 29, 2015';
            ```

        5. **RCVAccession**
        - Values are separated by `|`.
        - To search for a specific accession:
            ```sql
            SELECT * FROM allele WHERE RCVAccession LIKE '%RCV000123456%';
            ```

        6. **PhenotypeIDS**
        - `,` separates database IDs for the same phenotype.
        - `|` separates different phenotypes entirely.
        - Example:
            ```sql
            SELECT * FROM allele WHERE PhenotypeIDS LIKE '%OMIM:12345%';
            ```
        - Valid keys to use:
            ['EFO', 'Gene', 'Human Phenotype Ontology', 'MONDO', 'MeSH', 'MedGen', 'OMIM', 'Orphanet']

        7. **Origin**
        - Uses `;` as separator.
            Example:
            ```sql
            SELECT * FROM allele WHERE Origin LIKE '%germline%';
            ```

        8. **ReviewStatus**
        - Possible values:
            'no assertion criteria provided',
            'criteria provided, multiple submitters, no conflicts',
            'criteria provided, single submitter',
            'criteria provided, conflicting classifications',
            'reviewed by expert panel',
            'no classifications from unflagged records',
            'no classification provided',
            'practice guideline'
        - Example:
            ```sql
            SELECT * FROM allele WHERE ReviewStatus = 'reviewed by expert panel';
            ```

        9. **OtherIDs**
        - Values separated by `,`.
        - Example:
            ```sql
            SELECT * FROM allele WHERE OtherIDs LIKE '%ClinVarAccession123%';
            ```
        - Valid keys to use:
            ['BRCA1-HCI', 'BTK @ LOVD', 'Breast Cancer Information Core (BIC) (BRCA1)', 'Breast Cancer Information Core (BIC) (BRCA2)', 'British Heart Foundation', 'COL7A1 database', 'COSMIC', 'ClinGen', 'ClinVar', 'GUCY2C database', 'HBVAR', 'LDLR-LOVD', 'LOVD 3', 'Leiden Muscular Dystrophy (CAV3)', 'Leiden Muscular Dystrophy (CHRND)', 'Leiden Muscular Dystrophy (CHRNE)', 'Leiden Muscular Dystrophy (DAG1)', 'Leiden Muscular Dystrophy (DPM3)', 'Leiden Muscular Dystrophy (MYL2)', 'Leiden Muscular Dystrophy (MYL3)', 'Leiden Muscular Dystrophy (MYOZ2)', 'Leiden Muscular Dystrophy (MYPN)', 'Leiden Muscular Dystrophy (PDLIM3)', 'Leiden Muscular Dystrophy (TNNT1)', 'Leiden Muscular Dystrophy (TNNT3)', 'Leiden Muscular Dystrophy (TPM1)', 'MSeqDR', 'MYBPC3 homepage - Leiden Muscular Dystrophy pages', 'NAA10 @ LOVD', 'OMIM', 'PharmGKB Clinical Annotation', 'RettBASE (CDKL5)', 'Tuberous sclerosis database (TSC1)', 'Tuberous sclerosis database (TSC2)', 'UniProtKB', 'UniProtKB/Swiss-Prot', 'dbRBC', 'dbVar']

        10. **Category**
            - Possible values:
            'within single gene',
            'within multiple genes by overlap',
            'near gene, upstream',
            'asserted, but not computed',
            'near gene, downstream',
            'not identified'
            - Example:
            ```sql
            SELECT * FROM allele WHERE Category = 'within single gene';
            ```

        11. **MC (Molecular Consequence)**
            - Follows format `SO:ID|consequence_name`.
            - To query by ID:
            ```sql
            SELECT * FROM allele WHERE MC LIKE 'SO:0001583%';
            ```

        ======================
        Cross-table Information:
        ======================
        - When asked "when was an allele last updated", use `cross_references` table.

        - `var_citations` links:
        * `alleleID` ↔ organization ID
        * `variationID` ↔ organization ID
        * OrganizationID column can contain multiple comma-separated IDs.

        Example:
        To get all organization names for AlleleID = 15043:
        SELECT DISTINCT T1.OrganizationName
        FROM organization_summary AS T1
        JOIN (
            SELECT AlleleID, UNNEST(STRING_SPLIT(OrganizationID, ',')) AS OrgID
            FROM var_citations
        ) AS T2
        ON T1.OrganizationID = OrgID
        WHERE T2.AlleleID = 15043;

        ======================
        Database Scope:
        ======================
        - Only germline variants are stored.
        - Only SNPs are stored.
        """

        convert_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", f"Question: {question}"),
        ])
        structured_llm = self.llm.with_structured_output(self.ConvertToSQL)
        sql_generator = convert_prompt | structured_llm
        result = sql_generator.invoke({"question": question})
        sql = result.sql_query.strip()
        state["sql_query"] = sql
        return state

    @traceable
    def check_harmful_sql(self, state: AgentState):
        """Inspects the generated SQL query for destructive operations."""
        sql = state["sql_query"].lower()
        harmful_keywords = ["drop", "delete", "update", "insert", "alter"]

        state["sql_error"] = False
        state["harmful"] = False

        if any(x in sql for x in harmful_keywords):
            state["query_result"] = "Error: Destructive queries are not allowed. This system is read-only."
            state["harmful"] = True

        return state

    @traceable
    def execute_sql(self, state: AgentState):
        """Executes the generated SQL query on the DuckDB connection."""
        sql_query = state["sql_query"]
        try:
            result = self.conn.execute(sql_query)
            if sql_query.lower().startswith("select"):
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description]
                if rows:
                    state["query_rows"] = [dict(zip(columns, row)) for row in rows]
                    formatted = "\n".join([str(dict(zip(columns, row))) for row in rows])
                    state["query_result"] = formatted
                else:
                    state["query_rows"] = []
                    state["query_result"] = "No results found."
                state["sql_error"] = False
        except Exception as e:
            state["query_result"] = f"Error executing SQL query: {str(e)}"
            state["sql_error"] = True
        return state

    @traceable
    def generate_human_readable_answer(self, state: AgentState):
        """
        Placeholder for generating human-readable answers.
        Currently just formats the result. In the future, this will call an LLM
        to convert raw SQL results into natural language.
        
        Future implementation will be:
        result = state["query_result"]    
        state["query_result"] = f"Result:\n{call_llm(result)}"
        """
        result = state["query_result"]
        state["query_result"] = f"Result:\n{result}"
        return state

    @traceable
    def regenerate_query(self, state: AgentState):
        """Reformulates a failed question to make it easier to convert into SQL."""
        question = state["question"]
        system = "Reformulate the question to make it easier to convert into a correct SQL query without losing the meaning of the query."
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", f"Original Question: {question}"),
        ])
        structured_llm = self.llm.with_structured_output(self.RewrittenQuestion)
        rewriter = rewrite_prompt | structured_llm
        rewritten = rewriter.invoke({})
        state["question"] = rewritten.question
        state["attempts"] += 1
        return state

    @traceable
    def end_max_iterations(self, state: AgentState):
        """Handles cases where maximum retry attempts are reached."""
        state["query_result"] = "Please try again."
        return state

    @traceable
    def handle_not_relevant(self, state: AgentState):
        """Handles cases where the question is not relevant to the database."""
        state["query_result"] = "The query isn't relevant to the SQL tables."
        return state

    # Router functions
    def relevance_router(self, state: AgentState):
        return "convert_to_sql" if state["relevance"].lower() == "relevant" else "handle_not_relevant"

    def execute_sql_router(self, state: AgentState):
        return "generate_human_readable_answer" if not state["sql_error"] else "regenerate_query"

    def check_attempts_router(self, state: AgentState):
        return "convert_to_sql" if state["attempts"] < 3 else "end_max_iterations"

    def harmful_sql_router(self, state: AgentState):
        return "generate_human_readable_answer" if state["harmful"] else "execute_sql"

    def _build_workflow(self):
        """Build and compile the workflow graph."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("check_relevance", self.check_relevance)
        workflow.add_node("convert_to_sql", self.convert_nl_to_sql)
        workflow.add_node("check_harmful_sql", self.check_harmful_sql)
        workflow.add_node("execute_sql", self.execute_sql)
        workflow.add_node("generate_human_readable_answer", self.generate_human_readable_answer)
        workflow.add_node("regenerate_query", self.regenerate_query)
        workflow.add_node("handle_not_relevant", self.handle_not_relevant)
        workflow.add_node("end_max_iterations", self.end_max_iterations)

        # Add edges
        workflow.add_conditional_edges("check_relevance", self.relevance_router, {
            "convert_to_sql": "convert_to_sql",
            "handle_not_relevant": "handle_not_relevant",
        })

        workflow.add_edge("convert_to_sql", "check_harmful_sql")
        workflow.add_conditional_edges("execute_sql", self.execute_sql_router, {
            "generate_human_readable_answer": "generate_human_readable_answer",
            "regenerate_query": "regenerate_query",
        })

        workflow.add_conditional_edges("regenerate_query", self.check_attempts_router, {
            "convert_to_sql": "convert_to_sql",
            "end_max_iterations": "end_max_iterations",
        })

        workflow.add_conditional_edges("check_harmful_sql", self.harmful_sql_router, {
            "generate_human_readable_answer": "generate_human_readable_answer",
            "execute_sql": "execute_sql"
        })

        workflow.add_edge("generate_human_readable_answer", END)
        workflow.add_edge("handle_not_relevant", END)
        workflow.add_edge("end_max_iterations", END)

        workflow.set_entry_point("check_relevance")
        return workflow.compile()

    def run(self, question: str) -> dict:
        """
        Main entry point for the SQL Agent.
        
        Args:
            question (str): Natural language question to convert to SQL and execute
            
        Returns:
            dict: Final state containing query results and metadata
        """
        state = {
            "question": question,
            "sql_query": "",
            "query_result": "",
            "query_rows": [],
            "attempts": 0,
            "relevance": "",
            "sql_error": False,
            "harmful": False,
        }
        return self.app.invoke(state)

    def get_workflow_graph(self):
        """Get the workflow graph for visualization."""
        return self.app.get_graph(xray=True)

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

# Main execution and demo
if __name__ == "__main__":
    # Example usage
    agent = SQLAgent()
    
    # Test query
    result = agent.run("info about allele with 15043")
    print("Query Result:", result["query_result"])
    print("SQL Query:", result["sql_query"])
    
    # Optionally display workflow graph
    img_obj = Image(agent.get_workflow_graph().draw_mermaid_png())

    # Save the image data to a file
    output_path = "output_image.png"
    with open(output_path, "wb") as f:
        f.write(img_obj.data)
    
    # Clean up
    agent.close()


'''
USAGE :

from sql_agent import SQLAgent
agent = SQLAgent() or agent = SQLAgent(db_path="/custom/path/to/db.duckdb", llm_provider="gemini")
result = agent.run("Show me pathogenic variants")
agent.close()'''