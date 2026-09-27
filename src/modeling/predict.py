import os
import joblib
import numpy as np
import yaml

class PhishingPredictor:
    def __init__(self, config_path="config/config.yaml"):
        # Resolve config path relative to repository root if needed
        if not os.path.exists(config_path):
            if os.path.exists("../../config/config.yaml"):
                config_path = "../../config/config.yaml"
            elif os.path.exists("../config/config.yaml"):
                config_path = "../config/config.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        model_path = self.config["model"]["artifact_path"]
        if not os.path.exists(model_path):
            if os.path.exists("../../" + model_path):
                model_path = "../../" + model_path
            elif os.path.exists("../" + model_path):
                model_path = "../" + model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact missing at {model_path}. Run training first.")
            
        self.pipeline = joblib.load(model_path)
        
    def predict_message(self, text: str) -> dict:
        if not text or not text.strip():
            return {"error": "Empty text provided."}

        # Inference
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        prob_map = {str(classes[i]).strip().title(): float(probs[i]) for i in range(len(classes))}
        
        phishing_score = prob_map.get("Phishing", prob_map.get("Spam", 0.0))
        is_phishing = phishing_score >= self.config["model"]["phishing_threshold"]

        # Robust Token Signal Extraction with Bounds Checking
        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        
        feature_names = np.array(tfidf.get_feature_names_out())
        transformed = tfidf.transform([text])
        nonzero_idx = transformed.nonzero()[1]

        triggers = []
        if nonzero_idx.size > 0 and hasattr(clf, "coef_"):
            try:
                coefs = clf.coef_
                if hasattr(coefs, "toarray"):
                    coefs = coefs.toarray()
                
                # Normalize coefficient dimensions across models (SVM, LogisticRegression, MNB)
                if coefs.ndim == 2:
                    if len(clf.classes_) == 2:
                        coefs = coefs[0]
                    else:
                        phish_classes = [i for i, c in enumerate(clf.classes_) if str(c).lower() in ["phishing", "spam"]]
                        phish_idx = phish_classes[0] if phish_classes else 0
                        coefs = coefs[phish_idx]

                word_scores = []
                for idx in nonzero_idx:
                    # Strict bounds check against both feature names and coefficient array lengths
                    if idx < len(feature_names) and idx < len(coefs):
                        score = float(coefs[idx])
                        if score > 0:
                            word_scores.append((feature_names[idx], score))
                
                word_scores.sort(key=lambda x: x[1], reverse=True)
                triggers = [w[0] for w in word_scores[:5]]
            except Exception as e:
                print(f"Feature extraction note: {e}")
                triggers = []

        return {
            "verdict": "PHISHING" if is_phishing else "SAFE",
            "phishing_risk_score": round(phishing_score * 100, 2),
            "confidence_distribution": prob_map,
            "trigger_signals": triggers
        }

if __name__ == "__main__":
    predictor = PhishingPredictor()
    sample = "URGENT: Your corporate email will be suspended in 24 hours. Click here to verify credentials."
    print(predictor.predict_message(sample))
