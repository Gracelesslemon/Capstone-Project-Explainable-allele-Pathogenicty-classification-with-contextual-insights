import gradio as gr

# ===== TAB 1: DATABASE QUERY =====
def sql_query_handler(message, history):
    """Handler for SQL queries"""
    bot_response = f"You asked: {message}"
    return bot_response

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
            
            chatbot = gr.Chatbot(height=400)
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
                allele_dropdown = gr.Dropdown(
                    label="Allele ID",
                    choices=["15044", "15045", "15046"],
                    value="15044",
                    interactive=True
                )

            with gr.Row():
                gene_dropdown = gr.Dropdown(
                    label="Gene",
                    choices=["BRCA1", "TP53", "EGFR", "KRAS"],
                    interactive=True
                )

            with gr.Row():
                effect_dropdown = gr.Dropdown(
                    label="Variant Effect",
                    choices=["Missense", "Nonsense", "Frameshift", "Silent"],
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
                    placeholder="e.g., Show other pathogenic variants in this gene"
                )
                followup_btn = gr.Button("Ask")
            
            followup_output = gr.Markdown(label="Answer")
            
            # Hidden State (stores current allele context)
            allele_context = gr.State(None)
            
            # Event: Classify button
            classify_btn.click(
                fn=classify_handler,
                inputs=[allele_dropdown,gene_dropdown,effect_dropdown ],
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
