import os
import joblib
import numpy as np
import yaml

class PhishingPredictor:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        model_path = self.config["model"]["artifact_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact missing at {model_path}. Run training first.")
            
        self.pipeline = joblib.load(model_path)
        
    def predict_message(self, text: str) -> dict:
        if not text or not text.strip():
            return {"error": "Empty text provided."}

        # Inference
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        prob_map = {classes[i]: float(probs[i]) for i in range(len(classes))}
        
        phishing_score = prob_map.get("phishing", 0.0)
        is_phishing = phishing_score >= self.config["model"]["phishing_threshold"]

        # Token Signal Extraction
        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        
        feature_names = np.array(tfidf.get_feature_names_out())
        transformed = tfidf.transform([text])
        nonzero_idx = transformed.nonzero()[1]

        triggers = []
        if nonzero_idx.size > 0 and hasattr(clf, "coef_"):
            phish_idx = np.where(clf.classes_ == "phishing")[0][0]
            coefs = clf.coef_[phish_idx]
            word_scores = [(feature_names[i], float(coefs[i])) for i in nonzero_idx if coefs[i] > 0]
            word_scores.sort(key=lambda x: x[1], reverse=True)
            triggers = [w[0] for w in word_scores[:5]]

        return {
            "verdict": "PHISHING" if is_phishing else "BENIGN",
            "phishing_risk_score": round(phishing_score * 100, 2),
            "confidence_distribution": prob_map,
            "trigger_signals": triggers
        }

if __name__ == "__main__":
    predictor = PhishingPredictor()
    sample = "URGENT: Your account password has expired. Click here to verify immediately."
    print(predictor.predict_message(sample))
