import sys
import os
import pandas as pd  
# Add model_pipeline directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr

# Import from model_pipeline root
from model_utils import create_classifier
from input_pipeline import encode_single_variant, encode_batch_variants

# Import from contextual_assistant subdirectory
from contextual_assistant.sql_agent import SQLAgent

# Import from same directory (UI/)
from llm_formatter import create_formatter

# Initialize once at startup
classifier = create_classifier()
sql_agent = SQLAgent()
llm_formatter = create_formatter()

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)


# ===== TAB 1: SQL QUERY HANDLER (STREAMING) =====
def sql_query_handler(message, history):
    """Handler for SQL queries with streaming"""
    # Run SQL agent
    sql_result = sql_agent.run(message)
    
    # Extract query and result based on sql_agent output format
    if isinstance(sql_result, dict):
        sql_query = sql_result.get('sql_query', sql_result.get('SQL Query', ''))
        query_result = sql_result.get('result', sql_result.get('Query Result', sql_result))
    else:
        sql_query = ""
        query_result = sql_result
    
    # Add user message to history
    history = history + [{"role": "user", "content": message}]
    
    # Stream LLM response
    full_response = ""
    for chunk in llm_formatter.format_sql_result_stream(message, sql_query, query_result):
        full_response += chunk
        # Update history with accumulated response
        history_with_response = history + [{"role": "assistant", "content": full_response}]
        yield history_with_response


# ===== TAB 2: CLASSIFICATION HANDLER (STREAMING) =====
def classify_handler(allele_id, gene_id, mc_list, origin_list, var_gene_rel, genomic_loc):
    """Handler for single allele classification with streaming"""
    
    # Validation
    if not allele_id or not gene_id:
        error_msg = "⚠️ Please provide both Allele ID and Gene ID"
        return error_msg, gr.update(visible=False), None
    
    if not mc_list or not origin_list or not var_gene_rel or not genomic_loc:
        error_msg = "⚠️ Please fill in all dropdown fields"
        return error_msg, gr.update(visible=False), None
    
    # Show "Processing..." message
    yield "🔄 **Processing...** Encoding variant features...", gr.update(visible=False), None
    
    # Clear previous variant context
    llm_formatter.clear_variant_context()
    
    # Build input for encoder
    input_data = {
        'AlleleID': allele_id,
        'GeneID': gene_id,
        'MC': ','.join(mc_list),
        'Origin': ';'.join(origin_list),
        'VariantGeneRelation': var_gene_rel,
        'GenomicLocationData': 'g' if genomic_loc == 'is genomic' else 'm'
    }
    
    try:
        # Encode
        encoder_output = encode_single_variant(input_data)
        
        # Check for encoding errors
        if encoder_output.get('validation_issues'):
            error_msg = f"⚠️ Encoding Issues:\n" + "\n".join(encoder_output['validation_issues'])
            return error_msg, gr.update(visible=False), None
        
        yield "🧬 **Classifying...** Running SENN model...", gr.update(visible=False), None
        
        # Classify
        classification_result = classifier.classify_single(encoder_output)
        
        # Store encoded_features in context
        classification_result['encoded_features'] = encoder_output['encoded_features']
        
        yield "✨ **Formatting...** Generating explanation...", gr.update(visible=False), classification_result
        
        # Stream formatted result
        full_response = ""
        for chunk in llm_formatter.format_classification_result_stream(classification_result):
            full_response += chunk
            # Update display as chunks arrive
            yield full_response, gr.update(visible=False), classification_result
        
        # Final update - show sliders with proper labels
        yield full_response, gr.update(visible=True), classification_result
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, gr.update(visible=False), None

# ===== TAB 2: EXPORT HANDLER =====
def export_relevance_handler(allele_context):
    """Generates a CSV file for feature relevance with Global, Pathogenic, and Benign scores"""
    if not allele_context:
        return None
    
    try:
        # Create output directory
        os.makedirs("outputs", exist_ok=True)
        
        # Safe filename generation
        allele_id = allele_context.get('input_metadata', {}).get('allele_id', 'variant')
        file_path = f"outputs/feature_relevance_{allele_id}.csv"
        
        # 1. Get the dictionary containing the scores
        # Based on your logs, the data is in allele_context['feature_importance']
        feature_importance_data = allele_context.get('feature_importance', {})
        
        data = []
        
        # 2. Iterate through the dictionary to build rows
        for feature_name, scores in feature_importance_data.items():
            # Clean up feature name for readability (optional, remove if you want raw names)
            # readable_name = feature_name.replace('has_MC_', '').replace('has_Origin_', '').replace('is_', '')
            
            row = {
                "Feature Name": feature_name,
                "Global Importance": scores.get('global', 0.0),
                "Pathogenic Score": scores.get('Pathogenic', 0.0),
                "Benign Score": scores.get('Benign', 0.0)
            }
            data.append(row)

        # 3. Create DataFrame
        df = pd.DataFrame(data)
        
        # 4. Sort by Global Importance (Descending) to show most important first
        if not df.empty:
            df = df.sort_values(by='Global Importance', ascending=False)
            
        # 5. Save to CSV
        df.to_csv(file_path, index=False)
        print(f"✅ Exported relevance to {file_path}")
        
        return gr.update(value=file_path, visible=True)
        
    except Exception as e:
        print(f"❌ Export Error: {str(e)}")
        return None

# ===== TAB 2: WEIGHT ADJUSTMENT HANDLER (STREAMING) =====
def reclassify_handler(allele_context, *slider_values):
    """Handler for reclassification with adjusted weights"""
    
    if not allele_context:
        return "⚠️ No classification context found. Please classify a variant first."
    
    try:
        # Get all 33 feature names in order
        all_features = classifier.get_feature_names()
        
        # Map slider values to feature names
        weight_adjustments = {}
        for feature_name, slider_value in zip(all_features, slider_values):
            if slider_value != 1.0:  # Only include adjusted weights
                weight_adjustments[feature_name] = slider_value
        
        if not weight_adjustments:
            return "ℹ️ No weights adjusted. All sliders at default (1.0)."
        
        # Prepare encoder output
        encoder_output = {
            'allele_id': allele_context['input_metadata']['allele_id'],
            'gene_id': allele_context['input_metadata']['gene_id'],
            'encoded_features': allele_context['encoded_features'],
            'clinical_significance': allele_context['input_metadata']['clinical_significance'],
            'validation_issues': allele_context['input_metadata']['validation_issues']
        }
        
        # Reclassify
        adjustment_result = classifier.classify_with_weight_adjustment(
            encoder_output, weight_adjustments
        )
        
        # Stream result
        full_response = ""
        for chunk in llm_formatter.format_weight_adjustment_stream(adjustment_result):
            full_response += chunk
            yield full_response
    
    except Exception as e:
        yield f"❌ Error: {str(e)}"


def reset_sliders_handler():
    """Reset all sliders to 1.0"""
    all_features = classifier.get_feature_names()
    return [1.0] * len(all_features)  # Return list of 1.0 values


# # ===== TAB 2: FOLLOW-UP QUERY HANDLER (STREAMING) =====
# def followup_query_handler(query, allele_context):
#     """Handler for follow-up queries with streaming"""
    
#     if not query.strip():
#         yield "⚠️ Please enter a question"
#         return
    
#     if not allele_context:
#         yield "⚠️ No variant context. Please classify a variant first."
#         return
    
#     try:
#         # Check if query needs SQL lookup
#         sql_result = None
#         keywords = ['gene', 'disease', 'submission', 'allele', 'variant', 'chromosome']
        
#         if any(keyword in query.lower() for keyword in keywords):
#             allele_id = allele_context['input_metadata']['allele_id']
#             gene_id = allele_context['input_metadata']['gene_id']
            
#             # Enhance query with context
#             enhanced_query = f"{query} (AlleleID: {allele_id}, GeneID: {gene_id})"
#             sql_result = sql_agent.run(enhanced_query)
        
#         # Stream answer
#         full_response = ""
#         for chunk in llm_formatter.answer_followup_stream(query, sql_result):
#             full_response += chunk
#             yield full_response
    
#     except Exception as e:
#         yield f"❌ Error: {str(e)}"


# ===== TAB 3: BATCH HANDLER =====
def batch_classify_handler(file, progress=gr.Progress()):
    """Handler for batch classification"""
    if file is None:
        return None
    
    try:
        # Get file path
        progress(0, desc="📂 Reading uploaded file...")
        input_path = file.name
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"outputs/{base_name}_predictions.csv"
        
        # Encode batch
        progress(0.2, desc="📝 Encoding variants...")
        batch_encoded = encode_batch_variants(input_path)
        
        # Check for encoding errors
        if batch_encoded['successful_encodings'] == 0:
            print(f"⚠️ No variants successfully encoded. Issues: {batch_encoded['global_issues']}")
            return None
        
        total_variants = batch_encoded['total_variants']
        successful_encodings = batch_encoded['successful_encodings']
        
        # Classify batch
        progress(0.5, desc=f"🧬 Classifying {successful_encodings} variants...")
        result = classifier.classify_batch(
            batch_encoder_output=batch_encoded,
            original_csv_path=input_path,
            output_csv_path=output_path,
            include_confidence=True
        )
        
        progress(0.9, desc="💾 Saving results to CSV...")
        
        progress(1.0, desc=f"✅ Complete! {result['successful_predictions']}/{total_variants} classified")
        
        print(f"✅ Batch processed: {result['successful_predictions']}/{result['total_variants']} successful")
        return output_path
    
    except Exception as e:
        print(f"❌ Batch classification error: {str(e)}")
        return None


# ===== BUILD UI =====
with gr.Blocks(title="Variant Classification System", theme=gr.themes.Glass()) as demo:
    
    gr.Markdown("# 🧬 Variant Classification & Query System")
    
    with gr.Tabs():
        
        # ===== TAB 1: DATABASE QUERY =====
        with gr.Tab("Database Query"):
            gr.Markdown("## Ask questions about variants in the database")
            
            chatbot = gr.Chatbot(type="messages", height=400)
            
            msg = gr.Textbox(
                label="Your Question",
                placeholder="e.g., Tell me about BRCA1"
            )
            
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear Chat")
            
            # Event: Submit message
            submit_btn.click(
                fn=sql_query_handler,
                inputs=[msg, chatbot],
                outputs=[chatbot]
            ).then(
                lambda: "",  # Clear input after submit
                outputs=[msg]
            )
            
            # Event: Enter key submits
            msg.submit(
                fn=sql_query_handler,
                inputs=[msg, chatbot],
                outputs=[chatbot]
            ).then(
                lambda: "",
                outputs=[msg]
            )
            
            # Event: Clear chat
            clear_btn.click(lambda: [], outputs=[chatbot])
        
        # ===== TAB 2: SINGLE CLASSIFICATION =====
        with gr.Tab("Single Classification"):
            gr.Markdown("### Classify a single variant")
            
            # Input Section
            with gr.Row(): 
                allele_ID_input = gr.Textbox( 
                    label="Allele ID", 
                    placeholder="e.g., 15040"
                ) 
                gene_ID_input = gr.Textbox( 
                    label="Gene ID", 
                    placeholder="e.g., 55572"
                )
            
            with gr.Row():
                mc_dd = gr.Dropdown(
                    label="Molecular Consequence",
                    choices=[
                        "nonsense",
                        "non-coding_transcript_variant",
                        "missense_variant",
                        "intron_variant",
                        "5_prime_UTR_variant",
                        "splice_donor_variant",
                        "synonymous_variant",
                        "splice_acceptor_variant",
                        "initiator_codon_variant",
                        "3_prime_UTR_variant",
                        "no_sequence_alteration",
                        "stop_lost",
                        "genic_upstream_transcript_variant",
                        "genic_downstream_transcript_variant"
                    ],
                    multiselect=True,
                    interactive=True
                )
                
                origin_dd = gr.Dropdown(
                    label="Origin",
                    choices=[
                        "germline",
                        "biparental",
                        "unknown",
                        "maternal",
                        "paternal",
                        "inherited",
                        "de novo",
                        "not applicable",
                        "tested-inconclusive",
                        "uniparental",
                        "not-reported"
                    ],
                    multiselect=True,
                    interactive=True
                )
            
            with gr.Row():
                var_gene_reln_dd = gr.Dropdown(
                    label="Variant Gene Relation",
                    choices=[
                        "within single gene",
                        "within multiple genes by overlap",
                        "asserted, but not computed",
                        "near gene, upstream",
                        "near gene, downstream",
                        "not identified"
                    ],
                    interactive=True
                )
                
                gen_loc_data = gr.Dropdown(
                    label="Genomic Location",
                    choices=[
                        "is genomic",
                        "is mitochondrial"
                    ],
                    interactive=True
                )
            
            classify_btn = gr.Button("🔬 Classify Variant", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column(scale=4):
                    pass # Empty space to push button to the right
                with gr.Column(scale=1, min_width=200):
                    export_btn = gr.Button("📥 Download Relevance (.xlsx)", size="sm", variant="secondary")
                    # Hidden file component to facilitate the download
                    export_file = gr.File(label="Download Completed", visible=False, file_count="single")

            # Result Display
            result_display = gr.Markdown(label="Classification Result")
            
            # Weight Adjustment Section (initially hidden)
            with gr.Column(visible=False) as slider_section:
                gr.Markdown("### ⚖️ Adjust Feature Weights")
                gr.Markdown("*Modify feature importance to see how predictions change. 1.0 = original weight*")
                
                # Get all 33 feature names from classifier
                all_features = classifier.get_feature_names()
                
                # Group features by concept
                mc_features = [f for f in all_features if f.startswith('has_MC_')]
                origin_features = [f for f in all_features if f.startswith('has_Origin_')]
                vgr_features = [f for f in all_features if f.startswith('has_VariantGeneRelation_')]
                location_features = [f for f in all_features if f.startswith('is_')]
                
                # Store sliders in list (order matters!)
                sliders = []
                
                # Molecular Consequence (14 features)
                with gr.Accordion("🧬 Molecular Consequence (14 features)", open=False):
                    for feature in mc_features:
                        readable = feature.replace('has_MC_', '').replace('_', ' ').title()
                        slider = gr.Slider(0.1, 3.0, value=1.0, step=0.1, 
                                          label=readable, info=feature)
                        sliders.append(slider)
                
                # Origin (11 features)
                with gr.Accordion("🔬 Origin (11 features)", open=False):
                    for feature in origin_features:
                        readable = feature.replace('has_Origin_', '').replace('_', ' ').title()
                        slider = gr.Slider(0.1, 3.0, value=1.0, step=0.1,
                                          label=readable, info=feature)
                        sliders.append(slider)
                
                # Variant-Gene Relation (6 features)
                with gr.Accordion("🧪 Variant-Gene Relation (6 features)", open=False):
                    for feature in vgr_features:
                        readable = feature.replace('has_VariantGeneRelation_', '').replace('_', ' ').title()
                        slider = gr.Slider(0.1, 3.0, value=1.0, step=0.1,
                                          label=readable, info=feature)
                        sliders.append(slider)
                
                # Genomic Location (2 features)
                with gr.Accordion("📍 Genomic Location (2 features)", open=True):
                    for feature in location_features:
                        readable = feature.replace('is_', '').replace('_', ' ').title()
                        slider = gr.Slider(0.1, 3.0, value=1.0, step=0.1,
                                          label=readable, info=feature)
                        sliders.append(slider)
                
                # Buttons
                with gr.Row():
                    reset_btn = gr.Button("↺ Reset All", variant="secondary")
                    reclassify_btn = gr.Button("🔄 Reclassify", variant="primary")
            
            # Follow-up Query Section
            # gr.Markdown("---")
            # gr.Markdown("### 💬 Ask About This Variant")
            
            # followup_input = gr.Textbox(
            #     label="Your Question",
            #     placeholder="e.g., Why is this variant pathogenic? What gene is this in?"
            # )
            # followup_btn = gr.Button("Ask", variant="primary")
            # followup_output = gr.Markdown(label="Answer")
            
            # Hidden State (stores current allele context)
            allele_context = gr.State(None)
            
            # ===== ALL EVENT BINDINGS =====
            
            # Event: Classify button
            classify_btn.click(
            fn=classify_handler,
            inputs=[allele_ID_input, gene_ID_input, mc_dd, origin_dd, var_gene_reln_dd, gen_loc_data], # 
            outputs=[result_display, slider_section, allele_context],
            show_progress="full"  
            )

            export_btn.click(
                fn=export_relevance_handler,
                inputs=[allele_context],
                outputs=[export_file]
            )

            # Event: Reset sliders
            reset_btn.click(
                fn=reset_sliders_handler,
                outputs=sliders  # Update all sliders
            )
            
            # Event: Reclassify with adjusted weights
            reclassify_btn.click(
                fn=reclassify_handler,
                inputs=[allele_context] + sliders,  # Pass context + all slider values
                outputs=[result_display]
            )
            
            # # Event: Follow-up query (button)
            # followup_btn.click(
            #     fn=followup_query_handler,
            #     inputs=[followup_input, allele_context],
            #     outputs=[followup_output]
            # )
            
            # # Event: Follow-up query (enter key)
            # followup_input.submit(
            #     fn=followup_query_handler,
            #     inputs=[followup_input, allele_context],
            #     outputs=[followup_output]
            # )
        
        # ===== TAB 3: BATCH CLASSIFICATION =====
        with gr.Tab("Batch Classification"):
            gr.Markdown("### 📊 Upload CSV for batch classification")
            gr.Markdown("*Upload a CSV file with variant data. Results will include predictions and confidence scores.*")
            
            file_input = gr.File(
                label="Upload CSV File",
                file_types=[".csv"],
                type="filepath"
            )
            
            process_btn = gr.Button("⚙️ Process Batch", variant="primary", size="lg")
            
            gr.Markdown("---")
            
            file_output = gr.File(label="📥 Download Results")
            
            # Event: Process batch
            process_btn.click(
                fn=batch_classify_handler,
                inputs=[file_input],
                outputs=[file_output],
                show_progress="full"
            )


# Launch the app
if __name__ == "__main__":
    demo.launch(debug=True, share=False, inbrowser=True)
