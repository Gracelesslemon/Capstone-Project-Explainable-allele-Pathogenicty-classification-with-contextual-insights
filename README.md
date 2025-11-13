# Explainable Allele Pathogenicity Classification with Contextual Insights

## About the Project
This project aims to provide pathogenicity classification for a given allele and supplement it with additional contextual information in an interpretable manner.

## Novelty
- The model used for classification is inherently interpretable, allowing us to understand how it reached a specific conclusion.
- Additional contextual insights for the given input are provided for a comprehensive understanding.

---

## Features:

### SQL Agent
A natural language to SQL query system that processes user questions and returns database query results.

**Features:**
- **Relevance Checker**: Determines if the natural language request is relevant to the database schema
- **Fuzzy matcher**: To catch any misspelling 
- **Natural Language to SQL Conversion**: Converts natural language questions into valid SQL queries
- **Harmful Query Prevention**: Basic security check to prevent destructive SQL operations.
- **Query Regeneration**: When SQL generation fails, the system reformulates the original question and retries up to 3 times
- **Automatic Query Execution**: Executes generated queries and returns formatted results
- **Multi-LLM Support**: Supports multiple language model providers (Gemini, Perplexity, HuggingFace)

**Database Scope:**
- Contains only germline variants
- Focuses exclusively on SNPs (Single Nucleotide Polymorphisms)
---
### Input Encoder Pipeline
An encoding pipeline that processes user input and converts it into multi-hot and single-hot encoded features suitable for machine learning models.

**Features:**
- **Input Types**: Supports both single variant inputs and batch processing from files (TSV/CSV)
- **Multi-Hot and One-Hot Encoding**: Uses multi-hot encoding for MC and Origin categories, one-hot encoding for chromosomes, alleles, and variant-gene relations
- **Input Validation**: Validates against predefined categories extracted from ClinVar data and provides detailed validation feedback
- **Clinical Significance Integration**: Automatically retrieves and includes clinical significance data from the database when AlleleID is provided
- **File Format Detection**: Auto-detects TSV and CSV file formats for batch processing
- **Comprehensive Error Reporting**: Provides detailed logging for unknown categories, missing data, and processing failures at both individual and batch levels
- **Feature Template System**: Uses a consistent 66-feature template ensuring uniform output across all inputs

> [!IMPORTANT]
> Allele id and gene id are not encoded and are returned as is.


### Self-Explainable Neural Network (SENN)
An interpretable machine learning model for allele pathogenicity classification that provides inherent explainability through its architecture.

**Model Architecture:**
- **Identity Conceptizer**: Uses raw encoded features directly as concepts (no additional transformation needed due to extensive preprocessing in the input pipeline)
- **Linear Parameterizer**: Multi-layer neural network that generates relevance scores for each concept-class pair
  - Hidden layers: 128 → 64 → 32 neurons
  - Dropout rate: 0.3 for regularization
- **Sum Aggregator**: Combines concepts and relevances using weighted summation with log-softmax activation

**Training Configuration:**
- **Data Balancing**: Majority undersampling applied to balance benign and pathogenic classes
- **Data Split**: 70% training, 15% validation, 15% testing with stratified sampling
- **Feature Selection**: Removed chromosome and allele-specific features during training
- **Batch Size**: 64
- **Loss Function**: Negative Log Likelihood (NLL)
- **Optimizer**: Adam with weight decay (1e-5)
- **Learning Rate Scheduler**: StepLR (step size: 30 epochs, gamma: 0.5)
- **Early Stopping**: Patience of 10 epochs based on validation accuracy

**Interpretability Features:**
- **Individual Feature Importance**: Ranked importance scores for each feature across global, benign, and pathogenic classifications
- **Concept Grouping**: Features organized into meaningful biological concepts:
  - Genomic Location (chromosomes, mitochondrial markers)
  - Sequence Changes (reference and alternate alleles)
  - Gene Context (variant-gene relationships)
  - Molecular Consequences (functional impact categories)
  - Data Source (origin information)
- **Class-Specific Analysis**: Separate importance rankings for benign vs pathogenic predictions
- **Model Transparency**: Direct access to concept relevances for each prediction

### A gradio ui:
**three tabs**
- query the sql agent for information
- Classify single variant and continue with context retrieval
- Batch classification

--- 
## Project Tasks

### Global
- [X] **Chat UI Interface**: Unified interface connecting all system components

### Model Side
- [x] **Input Encoder Pipeline**: Converts input values into model-ready features
- [x] **Classification SENN Model**: Self-explainable neural network achieving 93% peak accuracy
- [X] **Natural Language Explanation Generator**: Convert feature importance scores into interpretable explanations for chat interface

### Contextual Assistant Side
- [x] **SQL Agent**: Handles straightforward natural language to SQL queries with basic joins
    - [ ] Potential addons would be to flush out the prompts through testing
    - [ ] Need to add more examples to make it more robust at handling a wide variety of queries.

# Changes to be made : 
- FUZZY MATCH FOR DISEASE IS ; SEPARATED CHANGE THAT . IMPORTANT 

[FUZZY MATCH] Original: any conflicts where  disease is galloway ?
[FUZZY MATCH] Extracted terms: ['galloway']
[FUZZY MATCH] Modified: any conflicts where  disease is Galloway-Mowat syndrome 8;Nephrotic syndrome, type 18 ?
[FUZZY MATCH] Replacements: [('galloway', 'Galloway-Mowat syndrome 8;Nephrotic syndrome, type 18')]
