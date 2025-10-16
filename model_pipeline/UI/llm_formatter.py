"""
LLM Formatter Module
Handles formatting of classification results and SQL query results using LLM with streaming support
"""

import os
import json
from typing import Dict, Any, List, Optional, Generator
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
# from langchain_huggingface import HuggingFaceHub, ChatHuggingFace
from transformers import pipeline
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()


class LLMFormatter:
    """
    Formats classification and SQL results using LLM for natural language output
    Supports streaming for real-time responses in Gradio
    """
    
    def __init__(self, llm_provider: Optional[str] = None):
        """
        Initialize LLM formatter
        
        Args:
            llm_provider: "gemini", "perplexity", "huggingface", "local-hf" (optional, uses env var)
        """
        self.provider = llm_provider or os.getenv("LLM_PROVIDER", "gemini").lower()
        self.llm = self.get_llm(self.provider)
        
        # Context storage for active variant
        self.current_variant_context = None
    
    
    def get_llm(self, provider: str = None):
        """Initializes and returns an LLM instance based on the specified provider."""
        provider = provider or os.getenv("LLM_PROVIDER", "gemini").lower()
        
        if provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash"),
                google_api_key=os.getenv("GEMINI_KEY"),
            )
        # elif provider == "perplexity":
        #     return ChatOpenAI(
        #         model=os.getenv("PPLX_MODEL", "sonar-pro"),
        #         api_key=os.getenv("PERPLEXITY_API_KEY"),
        #         base_url=os.getenv("PPLX_BASE_URL", "https://api.perplexity.ai/chat/completions"),
        #     )
        # elif provider == "huggingface":
        #     return HuggingFaceHub(
        #         repo_id=os.getenv("HF_MODEL", "meta-llama/Llama-3-8b-chat-hf"),
        #         huggingfacehub_api_token=os.getenv("HF_TOKEN")
        #     )
        # elif provider == "local-hf":
        #     pipe = pipeline(
        #         "text-generation",
        #         model=os.getenv("LOCAL_HF_MODEL", "meta-llama/Llama-3-8b-chat-hf"),
        #         device_map="auto"
        #     )
            # return ChatHuggingFace(pipeline=pipe)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    
    def set_variant_context(self, classification_result: Dict[str, Any]):
        """
        Store the current variant context for follow-up queries
        
        Args:
            classification_result: Output from classifier.classify_single()
        """
        self.current_variant_context = {
            'allele_id': classification_result['input_metadata']['allele_id'],
            'gene_id': classification_result['input_metadata']['gene_id'],
            'prediction': classification_result['prediction'],
            'confidence': classification_result['confidence'],
            'full_result': classification_result
        }
    
    
    def clear_variant_context(self):
        """Clear stored variant context (call when user classifies new variant)"""
        self.current_variant_context = None
    
    
    def get_variant_context(self) -> Optional[Dict[str, Any]]:
        """Get current variant context"""
        return self.current_variant_context
    
    
    # ===== METHOD 1: Format SQL Query Results (Streaming) =====
    def format_sql_result_stream(self, 
                                 user_query: str,
                                 sql_query: str,
                                 query_result: Any) -> Generator[str, None, None]:
        """
        Format SQL agent results into readable markdown with streaming
        
        Args:
            user_query: Original user question
            sql_query: The SQL query executed
            query_result: Result from SQL agent (dict, list, or string)
        
        Yields:
            Chunks of formatted markdown
        """
        # Convert result to JSON string
        if isinstance(query_result, (dict, list)):
            result_str = json.dumps(query_result, indent=2)
        else:
            result_str = str(query_result)
        
        system_prompt = """You are a clinical database assistant formatting ClinVar query results.

**Rules**:
1. State findings directly in one sentence
2. Use qualifying language inline: "suggests", "indicates", "based on current records"
3. No separate explanatory sentences (no "This suggests that...")
4. Format data in tables where appropriate

**Examples**:
- Bad: "Based on the query, no records were found. This suggests there are no conflicts."
- Good: "No conflicts found in current records."
- Bad: "The data shows 45 pathogenic variants."
- Good: "45 pathogenic variants recorded (based on submissions)."

**Style**: Direct statements with inline qualifiers. One sentence maximum for simple queries."""



        user_prompt = f"""**User Question**: {user_query}
                        **Database Query Result**:{result_str},
                        **SQL Query Used**: {sql_query}
                        Format this information for the user now:"""
        messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
        # Stream response
        for chunk in self.llm.stream(messages):
            if hasattr(chunk, 'content'):
                yield chunk.content
    
    
    # ===== METHOD 2: Format Classification Results (Streaming) =====
    def format_classification_result_stream(self, 
                                           classification_result: Dict[str, Any]) -> Generator[str, None, None]:
        """
        Format SENN classification results with streaming
        
        Args:
            classification_result: Output from classifier.classify_single()
        
        Yields:
            Chunks of formatted markdown
        """
        # Store context for follow-up queries
        self.set_variant_context(classification_result)
        
        # Extract key information
        prediction = classification_result['prediction']
        confidence = classification_result['confidence']
        allele_id = classification_result['input_metadata']['allele_id']
        gene_id = classification_result['input_metadata']['gene_id']
        
        # Get rankings
        top_global = classification_result['feature_rankings']['global_ranking'][:5]
        top_pathogenic = classification_result['feature_rankings']['pathogenic_ranking'][:5]
        top_benign = classification_result['feature_rankings']['benign_ranking'][:5]
        
        # Get concept importance
        concepts = classification_result['detailed_concept_analysis']['concept_scores']
        
        system_prompt = """You are explaining SENN variant predictions to clinicians assessing model trustability.

**Guidelines**:
1. **Prediction**: "🧬 **Prediction**: [X] ([Y]% confidence)"
2. **Evidence**: Top 5 features in ranked table
3. **Model Reasoning**: "Model weighted [feature] heavily (suggests [pattern])"
4. **Confidence**: "[X]% confidence (indicates [interpretation])"
5. **Rankings**: Global, pathogenic, benign tables
6. **Concepts**: Grouped importance scores
7. **Limitations**: Brief caveats (e.g., "High reliance on origin data")

**Style**: 
- Concise statements with inline qualifiers
- Use parentheses for interpretations: "(suggests X)", "(indicates Y)"
- Facts direct, interpretations qualified
- No standalone explanatory sentences

**Constraint**: Qualify interpretations inline but avoid bloat. "Feature X weighted heavily (suggests importance)" not "Feature X was weighted heavily. This suggests importance."  """



        user_prompt = f"""**Variant Classification Results**
                    **Allele ID**: {allele_id}
                    **Gene ID**: {gene_id}
                    **Prediction**: {prediction}
                    **Confidence**: {confidence:.1%}

                    **Top 5 Global Features**:
                    {json.dumps(top_global, indent=2)}

                    **Top 5 Pathogenic Features**:
                    {json.dumps(top_pathogenic, indent=2)}

                    **Top 5 Benign Features**:
                    {json.dumps(top_benign, indent=2)}

                    **Concept-Level Importance**:
                    {json.dumps(concepts, indent=2)}

                    Format this classification result in a comprehensive, easy-to-understand format:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Stream response
        for chunk in self.llm.stream(messages):
            if hasattr(chunk, 'content'):
                yield chunk.content
    
    
    # ===== METHOD 3: Format Weight Adjustment (Streaming) =====
    def format_weight_adjustment_stream(self,
                                       adjustment_result: Dict[str, Any]) -> Generator[str, None, None]:
        """
        Format weight adjustment comparison with streaming
        
        Args:
            adjustment_result: Output from classifier.classify_with_weight_adjustment()
        
        Yields:
            Chunks of formatted markdown
        """
        original_pred = adjustment_result['original_prediction']
        adjusted_pred = adjustment_result['adjusted_prediction']
        changed = adjustment_result['prediction_changed']
        confidence_change = adjustment_result['confidence_change']
        weights_applied = adjustment_result['weight_adjustments_applied']
        
        system_prompt = """You are explaining weight sensitivity analysis to clinicians.

**Guidelines**:
1. **Impact**: "⚠️ Prediction FLIPPED" or "✅ Prediction STABLE"
2. **Comparison Table**: Before/after with confidence
3. **Adjustments**: Which features changed (table)
4. **Behavior**: "Prediction remained stable (suggests low sensitivity to [feature])"
5. **Confidence Delta**: "Confidence changed by [X]%"
6. **Takeaway**: "Pattern suggests [interpretation]"

**Style**: 
- Concise with inline qualifiers
- Use parentheses: "(suggests robustness)", "(indicates sensitivity)"
- No absolute claims like "proves" or "definitely shows"
- No separate explanatory sentences

**Example**:
- Bad: "The prediction remained stable. This is strong evidence of robustness."
- Good: "Prediction stable despite adjustments (suggests low feature sensitivity)." """



        user_prompt = f"""**Weight Adjustment Results**
                        **Original Prediction**: {original_pred} ({adjustment_result['original_confidence']:.1%})
                        **Adjusted Prediction**: {adjusted_pred} ({adjustment_result['adjusted_confidence']:.1%})
                        **Prediction Changed**: {"Yes ⚠️" if changed else "No ✅"}
                        **Confidence Change**: {confidence_change:+.1%}

                        **Weight Adjustments Applied**:
                        {json.dumps(weights_applied, indent=2)}
                        Format this adjustment comparison clearly:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Stream response
        for chunk in self.llm.stream(messages):
            if hasattr(chunk, 'content'):
                yield chunk.content
    
    
    # ===== METHOD 4: Answer Follow-up Questions (Streaming) =====
    def answer_followup_stream(self,
                              user_question: str,
                              sql_result: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """
        Answer follow-up questions about the classified variant with streaming
        
        Args:
            user_question: User's follow-up question
            sql_result: Optional SQL query result if question needs DB lookup
        
        Yields:
            Chunks of answer text
        """
        if not self.current_variant_context:
            yield "⚠️ No variant context found. Please classify a variant first."
            return
        
        context = self.current_variant_context
        
        context_summary = f"""**Current Variant Context**:
- Allele ID: {context['allele_id']}
- Gene ID: {context['gene_id']}
- Prediction: {context['prediction']} ({context['confidence']:.1%})
"""
        
        sql_context = ""
        if sql_result:
            if isinstance(sql_result, (dict, list)):
                sql_context = f"\n**Additional Database Information**:\n``````"
            else:
                sql_context = f"\n**Additional Database Information**:\n{sql_result}"
        
        system_prompt = """You are a clinical genetics assistant answering variant questions.

**Rules**:
1. One sentence for simple queries
2. Qualify when interpreting: "(suggests X)", "(based on Y)"
3. State facts directly: "96.5% confidence" not "appears to be 96.5%"
4. No filler like "This suggests..." as separate sentences
5. Use tables for multi-row data only

**Examples**:
- Q: "Any conflicts?" → A: "No conflicts in current records."
- Q: "Confidence?" → A: "96.5%"
- Q: "Why pathogenic?" → A: "Nonsense mutation with de novo origin (suggests loss-of-function impact)."

**Style**: Direct facts, qualified interpretations (inline with parentheses)."""



        user_prompt = f"""{context_summary} , {sql_context}
        **User Question**: {user_question}

        Answer the question using the context provided:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Stream response
        for chunk in self.llm.stream(messages):
            if hasattr(chunk, 'content'):
                yield chunk.content

# ===== CONVENIENCE FUNCTION =====
def create_formatter(llm_provider: Optional[str] = None) -> LLMFormatter:
    """
    Factory function to create LLM formatter instance
    
    Args:
        llm_provider: "gemini", "perplexity", "huggingface", "local-hf" (optional)
    
    Returns:
        LLMFormatter instance
    """
    return LLMFormatter(llm_provider=llm_provider)


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from model_utils import create_classifier
    from input_pipeline import encode_single_variant
    
    # Initialize
    formatter = create_formatter()
    classifier = create_classifier()
    
    print("="*80)
    print("LLM FORMATTER - STREAMING DEMO")
    print("="*80)
    
    # Example 1: Classification formatting
    print("\n=== CLASSIFICATION FORMATTING ===\n")
    
    encoder_output = encode_single_variant({
        'AlleleID': 15040,
        'GeneID': 55572,
        'Origin': 'germline',
        'VariantGeneRelation': 'within single gene',
        'MC': 'nonsense,non-coding_transcript_variant',
        'GenomicLocationData': 'g'
    })
    
    classification_result = classifier.classify_single(encoder_output)
    
    # Stream output
    for chunk in formatter.format_classification_result_stream(classification_result):
        print(chunk, end='', flush=True)
    
    print("\n\n" + "="*80)
    
    # Example 2: Follow-up question
    print("\n=== FOLLOW-UP QUESTION ===\n")
    
    for chunk in formatter.answer_followup_stream("Why is this variant pathogenic?"):
        print(chunk, end='', flush=True)
    
    print("\n\n" + "="*80)