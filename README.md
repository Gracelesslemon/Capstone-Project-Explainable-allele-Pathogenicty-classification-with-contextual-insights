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
> [!NOTE]
> Primarily focused on straightforward SQL queries or simple joins
> Complex queries with multiple joins should be handled by the graph RAG system

**Database Scope:**
- Contains only germline variants
- Focuses exclusively on SNPs (Single Nucleotide Polymorphisms)
---
### Input Encoder Pipeline
An encoding pipeline that processes user input and converts it into multi-hot and single-hot encoded features suitable for machine learning models.

**Features:**
- **Input Types**: Supports both single variant inputs and batch processing from files
- **Input Validation**: Provides predefined options for categorical fields and validates user selections
- **Clinical Significance Integration**: Automatically retrieves and includes clinical significance data when available in the database
- **Error Logging**: Provides detailed feedback when validation fails or incorrect information is provided
- **Feature Encoding**: Generates 66-length feature vectors using appropriate encoding methods

**Usage:**

Single Variant Processing:
```python
result = encode_variant_endpoint(
    input_data={
        'AlleleID': 15044,
        'GeneID': 55572,
        'Origin': 'germline',
        'Chromosome': '11',
        'ReferenceAlleleVCF': 'G',
        'AlternateAlleleVCF': 'T',
        'VariantGeneRelation': 'within single gene',
        'MC': 'nonsense,non-coding_transcript_variant',
        'GenomicLocationData': 'g'
    },
    input_type="single"
)

# Output Schema
# result = {
#     "allele_id": str | int | None,        # AlleleID if provided, else None
#     "gene_id": str | int | None,          # GeneID if provided, else None
#     "clinical_significance": str | None,  # Retrieved from database if found
#     "encoded_features": List[float] | None,  # 66-length feature array if encoding succeeds
#     "validation_issues": List[str]        # Any warnings/errors/unknown categories
# }
```

Batch Processing:
```python
batch_result = encode_variant_endpoint(
    file_path=r"C:\path\to\batch\file",
    input_type="batch"
)

# Output Schema
# batch_result = {
#     "total_variants": int,               # Total rows processed from file
#     "successful_encodings": int,         # Number of successfully encoded variants
#     "failed_encodings": int,             # Number of failed encodings
#     "results": [                         # List of results, one per variant (same schema as single result)
#         # ... individual variant results
#     ]
# }
```
