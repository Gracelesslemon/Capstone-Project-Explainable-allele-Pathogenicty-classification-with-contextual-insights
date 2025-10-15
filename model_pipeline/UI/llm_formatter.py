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
        
        system_prompt = """You are a clinical database assistant formatting ClinVar query results for clinicians.

            **Formatting Guidelines**:
            1. Use structured markdown tables for gene/variant statistics
            2. Highlight clinically relevant metrics (pathogenic counts, conflicts, uncertain variants)
            3. Present submission data clearly (total submissions, reporting labs)
            4. Flag important findings (high conflict rates, uncertain significance prevalence)
            5. Be concise - clinicians need quick data interpretation
            6. No need to explain genetics terminology

            **Style**: Professional, data-focused, use tables for structured data.
            
            **Critical Constraint**: 
Never overstate model capabilities. Use cautious language: "suggests", "indicates", "may suggest", "appears to". Avoid absolute claims like "highly robust", "strong evidence", "definitively shows". Acknowledge this is a predictive model with inherent limitations. Always qualify confidence statements.
"""


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
        
        system_prompt = """You are a clinical decision support AI explaining Self-Explaining Neural Network (SENN) predictions for variant pathogenicity classification.

        **Target Audience**: Clinicians and genetic counselors who need to assess model trustability.

        **Formatting Guidelines**:
        1. **Prediction Summary**: Start with "🧬 **Prediction**: [X] ([Y]% confidence)"
        2. **Evidence Analysis**: Present the top 5 features that drove this prediction in a ranked table with their relevance scores
        3. **Model Reasoning**: Explain which biological patterns the model detected (e.g., "Model weighted nonsense mutations heavily due to loss-of-function impact")
        4. **Confidence Assessment**: Interpret the confidence score - what makes this prediction strong/weak?
        5. **Feature Ranking Tables**: Show global, pathogenic-favoring, and benign-favoring features separately
        6. **Concept-Level Summary**: Group by molecular consequence, origin, gene relation, and location with aggregate scores
        7. **Clinical Implications**: Brief note on what this classification means for clinical interpretation
        8. **Potential Limitations**: Any caveats based on the feature distribution (e.g., "High reliance on origin data")

        **Style**:
        - Assume clinical genetics knowledge - no need to define basic terms
        - Focus on interpretability and trustability
        - Use tables extensively for feature rankings
        - Be concise but thorough
        - Use minimal emojis (✅ ⚠️ 📊)
        
        **Critical Constraint**: 
Never overstate model capabilities. Use cautious language: "suggests", "indicates", "may suggest", "appears to". Avoid absolute claims like "highly robust", "strong evidence", "definitively shows". Acknowledge this is a predictive model with inherent limitations. Always qualify confidence statements.
"""


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
        
        system_prompt = """You are explaining feature weight sensitivity analysis for a SENN variant classifier.

        **Target Audience**: Clinicians evaluating model robustness.

        **Formatting Guidelines**:
        1. **Impact Summary**: Start with whether prediction changed: "⚠️ Prediction FLIPPED" or "✅ Prediction STABLE"
        2. **Before/After Comparison Table**: Show original vs adjusted prediction with confidence scores
        3. **Sensitivity Analysis**: Which features were adjusted and by how much (table format)
        4. **Model Robustness**: Interpret what this tells us about the model:
        - If prediction flipped: "Model is sensitive to [feature X] - this suggests [interpretation]"
        - If stable: "Prediction robust despite weight adjustments - strong evidence convergence"
        5. **Confidence Delta**: Explain the change in confidence percentage
        6. **Clinical Takeaway**: What does this sensitivity tell us about trusting this classification?

        **Style**: 
        - Focus on model interpretability and trustability
        - Use comparison tables
        - Emphasize clinical decision-making implications
        - Be direct about model strengths/weaknesses
        
        **Critical Constraint**: 
Never overstate model capabilities. Use cautious language: "suggests", "indicates", "may suggest", "appears to". Avoid absolute claims like "highly robust", "strong evidence", "definitively shows". Acknowledge this is a predictive model with inherent limitations. Always qualify confidence statements.
"""


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
        
        system_prompt = """You are a clinical genetics assistant answering questions about classified variants.

        **Guidelines**:
        1. Answer directly and concisely
        2. Use classification context and database data when available
        3. Format structured data in tables
        4. Assume clinical genetics knowledge
        5. Focus on actionable information
        6. Reference specific features/scores when relevant

        **Style**: Professional, concise, clinically focused.
        
        **Critical Constraint**: 
Never overstate model capabilities. Use cautious language: "suggests", "indicates", "may suggest", "appears to". Avoid absolute claims like "highly robust", "strong evidence", "definitively shows". Acknowledge this is a predictive model with inherent limitations. Always qualify confidence statements.
"""


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