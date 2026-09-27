import os
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def load_config(config_path="config/config.yaml"):
    """Load hyperparameters and paths from config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_classifier(model_name):
    """Factory to retrieve the appropriate classifier initialized with balanced weights."""
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    elif model_name == "SVM (Linear)":
        return SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    elif model_name == "Multinomial Naive Bayes":
        return MultinomialNB()
    else:
        # Default to Logistic Regression
        return LogisticRegression(class_weight='balanced', random_state=42)

def train_model():
    # 1. Load Configurations
    config = load_config()
    data_path = config["data"]["raw_path"]
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place your CSV file there.")
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X = df['text'].astype(str)
    y = df['label']
    
    # 2. Stratified Train-Test Split
    test_size = config["data"].get("test_size", 0.2)
    random_state = config["data"].get("random_state", 42)
    
    print(f"Splitting dataset (Test Size: {test_size * 100}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 3. Pipeline Construction
    ngram_range = tuple(config["features"].get("ngram_range", [1, 2]))
    max_features = config["features"].get("max_features", 5000)
    model_name = config["model"].get("default_classifier", "Logistic Regression")
    
    print(f"Building NLP pipeline using {model_name}...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)),
        ('clf', get_classifier(model_name))
    ])
    
    # 4. Training
    print("Training model (this may take a moment)...")
    pipeline.fit(X_train, y_train)
    
    # 5. Model Evaluation
    print("Evaluating model against unseen test data...")
    y_pred = pipeline.predict(X_test)
    
    print("\n" + "="*30)
    print("--- Validation Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4%}")
    print("-" * 30)
    print(classification_report(y_test, y_pred))
    print("="*30 + "\n")
    
    # 6. Artifact Persistence
    artifact_path = config["model"]["artifact_path"]
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    
    print(f"Saving serialized pipeline artifact to {artifact_path}...")
    joblib.dump(pipeline, artifact_path)
    print("Training complete! Model successfully saved and ready for production.")

if __name__ == "__main__":
    train_model()
