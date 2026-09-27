import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gradio as gr
from src.modeling.predict import PhishingPredictor

try:
    predictor = PhishingPredictor()
except Exception as e:
    predictor = None
    print(f"⚠️ Warning: Could not load PhishingPredictor ({e}). Ensure model is trained first.")

def analyze_message_handler(text: str):
    if not text or not text.strip():
        return "*Please enter a message on the left to generate a diagnostic report.*", {}

    if predictor is None:
        return "⚠️ **Error:** Serialized model artifact missing. Please run `python src/modeling/train.py` first.", {}

    results = predictor.predict_message(text)

    if "error" in results:
        return f"⚠️ **Error:** {results['error']}", {}

    verdict = results["verdict"]
    phishing_score = results["phishing_risk_score"]
    raw_prob_map = results["confidence_distribution"]
    triggers = results["trigger_signals"]

    # --- CONSOLIDATE & CLEAN CONFIDENCE MAP FOR GRADIO CHART ---
    clean_prob_map = {}
    for key, val in raw_prob_map.items():
        clean_key = str(key).strip().title()
        if clean_key.lower() in ['benign', 'ham', 'legitimate', 'safe']:
            clean_key = 'Safe'
        elif clean_key.lower() in ['phishing', 'spam']:
            clean_key = 'Phishing'
            
        clean_prob_map[clean_key] = clean_prob_map.get(clean_key, 0.0) + float(val)

    keywords_str = ", ".join(f"`{kw}`" for kw in triggers) if triggers else "No high-risk vocabulary signals detected."

    status_icon = "🔴" if verdict.upper() in ["PHISHING", "SPAM"] else "🟢"
    report = f"## {status_icon} Verdict: **{verdict.upper()}**\n\n"
    report += f"**Phishing Risk Score:** `{phishing_score:.1f}%`\n\n"
    report += "### Key Token Signals Extracted\n"
    report += f"{keywords_str}\n\n"
    report += "### SOC Threat Assessment\n"
    if phishing_score >= 85:
        report += "**CRITICAL RISK:** High likelihood of malicious intent. Immediate containment recommended."
    elif phishing_score >= 50:
        report += "**MEDIUM RISK:** Suspicious indicator triggers detected. Analyst review required."
    else:
        report += "**LOW RISK:** Message text aligns with benign communication patterns."

    return report, clean_prob_map

def quarantine_action():
    return "*Status: 🟡 Message successfully moved to quarantine queue.*"

def block_action():
    return "*Status: 🔴 Sender address/domain flagged and blocked in firewall policy.*"

def purge_action():
    return "*Status: ⚫ Message permanently purged from inbox store.*"

def create_dashboard():
    with gr.Blocks(theme=gr.themes.Ocean(), title="Social Sentinel SOC Dashboard") as dashboard:
        gr.Markdown("# Social Sentinel: Phishing Text Predictor")
        gr.Markdown("Interactive message analysis dashboard backed by cross-validated machine learning pipelines.")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    lines=8,
                    max_lines=12,
                    placeholder="Paste email or message text here (e.g., 'URGENT: Click here to verify your bank password immediately')...",
                    label="1. Input Message Text"
                )
                analyze_btn = gr.Button("Analyze Message", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_report = gr.Markdown(
                    value="*Enter a message on the left and click **Analyze Message** to generate a diagnostic report.*",
                    label="2. Message Analytics & Diagnosis"
                )
                output_chart = gr.Label(label="3. Prediction Confidence Distribution")

        gr.Markdown("---")
        gr.Markdown("### Incident Response Actions")
        with gr.Row():
            quarantine_btn = gr.Button("🟡 Quarantine Message", variant="secondary")
            block_btn = gr.Button("🟠 Block Sender", variant="stop")
            purge_btn = gr.Button("⚫ Purge Message")
        
        action_status = gr.Markdown("*Status: Waiting for action...*")

        analyze_btn.click(
            fn=analyze_message_handler,
            inputs=[input_text],
            outputs=[output_report, output_chart]
        )

        quarantine_btn.click(fn=quarantine_action, inputs=[], outputs=[action_status])
        block_btn.click(fn=block_action, inputs=[], outputs=[action_status])
        purge_btn.click(fn=purge_action, inputs=[], outputs=[action_status])

    return dashboard

if __name__ == "__main__":
    app = create_dashboard()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
