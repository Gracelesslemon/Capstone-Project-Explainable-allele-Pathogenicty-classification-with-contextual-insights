import gradio as gr

# ===== TAB 1: DATABASE QUERY =====
# def sql_query_handler(message, history):
#     """Handler for SQL queries"""
#     bot_response = f"You asked: {message}"
#     return bot_response

def sql_query_handler(message, history):
    history = history + [{"role": "user", "content": message}]
    response = "Example answer"
    history = history + [{"role": "assistant", "content": response}]
    return history


# ===== TAB 2: SINGLE CLASSIFICATION =====
def classify_handler(allele_input):
    """Handler for single allele classification"""
    return f"Classification result for: {allele_input}"

def reclassify_handler(allele_context, *slider_values):
    """Handler for reclassification with adjusted weights"""
    return "Reclassified with adjusted weights"

def followup_query_handler(query, allele_context):
    """Handler for follow-up queries about classified variant"""
    return f"Follow-up answer: {query}"

# ===== TAB 3: BATCH CLASSIFICATION =====
def batch_classify_handler(file):
    """Handler for batch classification"""
    return "output_placeholder.csv"

# ===== BUILD UI =====
with gr.Blocks(title="Variant Classification System") as demo:
    
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
            submit_btn = gr.Button("Submit")
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
            clear_btn.click(lambda: None, outputs=[chatbot])
        
        # ===== TAB 2: SINGLE CLASSIFICATION =====
        with gr.Tab("Single Classification"):
            gr.Markdown("### Classify a single variant")
            
            # Input Section
            with gr.Row(): 
                allele_ID_input = gr.Textbox( 
                    label="Allele ID ", 
                    placeholder="e.g., 15044" ) 
                gene_ID_input = gr.Textbox( 
                    label="gene ID ", 
                    placeholder="e.g., 15044" )
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
                            ]
                            ,
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
                            ]
                            ,
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
                            ]
                            ,
                    interactive=True
                )
                gen_loc_data = gr.Dropdown(
                    label="GenomicLocationData",
                    choices=[
                            "is genomic",
                            "is mitochondrial",
                            ]
                            ,
                    interactive=True
                )
            with gr.Row():

                classify_btn = gr.Button("Classify", variant="primary")
            
            # Result Display
            result_display = gr.Markdown(label="Classification Result")
            
            # Weight Adjustment Section (initially hidden)
            with gr.Column(visible=False) as slider_section:
                gr.Markdown("### Adjust Feature Weights")
                
                with gr.Accordion("Top 10 Features", open=True):
                    slider1 = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Feature 1")
                    slider2 = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Feature 2")
                    slider3 = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Feature 3")
                    slider4 = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Feature 4")
                    slider5 = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Feature 5")
                
                with gr.Accordion("All Features", open=False):
                    slider6 = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Feature 6")
                    slider7 = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Feature 7")
                    slider8 = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Feature 8")
                    slider9 = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Feature 9")
                    slider10 = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Feature 10")
                
                reclassify_btn = gr.Button("Reclassify with Adjusted Weights")
            
            # Follow-up Query Section
            gr.Markdown("---")
            gr.Markdown("### Ask About This Variant")
            
            with gr.Row():
                followup_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., Give me more info about this variant"
                )
            with gr.Row():
                followup_btn = gr.Button("Ask")
            
            followup_output = gr.Markdown(label="Answer")
            
            # Hidden State (stores current allele context)
            allele_context = gr.State(None)
            
            # Event: Classify button
            classify_btn.click(
                fn=classify_handler,
                inputs=[allele_ID_input,gene_ID_input,mc_dd,origin_dd,var_gene_reln_dd,gen_loc_data],
                outputs=[result_display, slider_section, allele_context]
            )
            
            # Event: Reclassify with adjusted weights
            reclassify_btn.click(
                fn=reclassify_handler,
                inputs=[allele_context, slider1, slider2, slider3, slider4, slider5,
                       slider6, slider7, slider8, slider9, slider10],
                outputs=[result_display]
            )
            
            # Event: Follow-up query
            followup_btn.click(
                fn=followup_query_handler,
                inputs=[followup_input, allele_context],
                outputs=[followup_output]
            )
        
        # ===== TAB 3: BATCH CLASSIFICATION =====
        with gr.Tab("Batch Classification"):
            gr.Markdown("### Upload CSV for batch classification")
            
            with gr.Column():
                file_input = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"]
                )
                
                process_btn = gr.Button("Process Batch", variant="primary")
                
                gr.Markdown("---")
                
                file_output = gr.File(label="Download Results")
            
            # Event: Process batch
            process_btn.click(
                fn=batch_classify_handler,
                inputs=[file_input],
                outputs=[file_output]
            )

# Launch the app
if __name__ == "__main__":
    demo.launch(debug=True, inbrowser=True)








# Check : Cluade code to merge to ui
# # In your Gradio handler file
# from senn_classifier import create_classifier
# from input_encoder import encode_single_variant

# # Initialize once at startup
# classifier = create_classifier()

# # Handler 1: Single variant classification
# def classify_variant_handler(variant_input):
#     # Encode
#     encoder_output = encode_single_variant(variant_input)
    
#     # Classify
#     result = classifier.classify_single(encoder_output)
    
#     # Return for display/LLM
#     return result

# # Handler 2: Weight adjustment
# def weight_adjustment_handler(variant_input, slider_values):
#     # Encode
#     encoder_output = encode_single_variant(variant_input)
    
#     # Build weight adjustments from sliders
#     weight_adjustments = {
#         "has_MC_nonsense": slider_values['slider1'],
#         "has_Origin_germline": slider_values['slider2'],
#         # ... map all your sliders to feature names
#     }
    
#     # Classify with adjustments
#     result = classifier.classify_with_weight_adjustment(
#         encoder_output, weight_adjustments
#     )
    
#     return result

# # Handler 3: Batch classification
# def batch_classify_handler(uploaded_file_path):
#     # Encode batch
#     from input_encoder import encode_batch_variants
#     batch_encoded = encode_batch_variants(uploaded_file_path)
    
#     # Classify batch
#     output_path = "outputs/batch_predictions.csv"
#     result = classifier.classify_batch(
#         batch_encoder_output=batch_encoded,
#         original_csv_path=uploaded_file_path,
#         output_csv_path=output_path,
#         include_confidence=True
#     )
    
#     return result
