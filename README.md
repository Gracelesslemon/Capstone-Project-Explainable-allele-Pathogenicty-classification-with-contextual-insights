# Allele Pathogenicity Classification with Contextual Insights

## About the Project
This project aims to provide pathogenicity classification for a given allele and supplement it with additional contextual information in an interpretable manner.

## Novelty
- The model used for classification is inherently interpretable, allowing us to understand how it reached a specific conclusion.
- Additional contextual insights for the given input are provided for a comprehensive understanding.

---

## What Has Been Done So Far

### SQL Agent
A natural language to SQL query system that processes user questions and returns database query results.

**Features:**
- **Relevance Checker**: Determines if the natural language request is relevant to the database schema
- **Natural Language to SQL Conversion**: Converts natural language questions into valid SQL queries
- **Harmful Query Prevention**: Basic security check to prevent destructive SQL operations.
- **Query Regeneration**: When SQL generation fails, the system reformulates the original question and retries up to 3 times
- **Automatic Query Execution**: Executes generated queries and returns formatted results
- **Multi-LLM Support**: Supports multiple language model providers (Gemini, Perplexity, HuggingFace)

**Usage:**
```python
from sql_agent import SQLAgent

# Initialize with default settings
agent = SQLAgent()

# Or initialize with custom parameters
agent = SQLAgent(db_path="/custom/path/to/db.duckdb", llm_provider="gemini")

# Run a query
result = agent.run("Show me allele id 15043")
print("Query Result:", result["query_result"])
print("Generated SQL:", result["sql_query"])

# Clean up
agent.close()
```

**Limitations:**
- Primarily focused on straightforward SQL queries or simple joins
- Complex queries with multiple joins should be handled by the graph RAG system

**Database Scope:**
- Contains only germline variants
- Focuses exclusively on SNPs (Single Nucleotide Polymorphisms)

---
