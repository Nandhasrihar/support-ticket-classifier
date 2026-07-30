"""
Support Ticket Classifier - Training Pipeline
================================================
Model: TF-IDF + Multinomial Naive Bayes
Features:
- Text preprocessing (lowercase, punctuation removal, stopword filtering)
- TF-IDF vectorization
- Multinomial Naive Bayes classification
- Comprehensive evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
- Model serialization using joblib
"""

import pandas as pd
import numpy as np
import string
import nltk
import os
import json
from datetime import datetime

# NLP preprocessing
from nltk.corpus import stopwords

# ML pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Model serialization
import joblib

# Download required NLTK data
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "tfidf_max_features": 1000,
    "tfidf_ngram_range": (1, 2),
    "confidence_threshold": 0.60,  # Threshold for "Needs Human Review"
}

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(filepath="data/tickets.csv"):
    """Load and display basic dataset statistics."""
    print("📂 Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"✅ Dataset loaded: {len(df)} tickets")
    print(f"\n📊 Category distribution:")
    print(df["category"].value_counts())
    
    return df


def preprocess_text(text):
    """
    Preprocess text:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove stopwords
    """
    stop_words = set(stopwords.words("english"))
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words and w.strip()]
    
    return " ".join(words)


def prepare_data(df):
    """Combine subject and body, then preprocess."""
    print("\n🔄 Preprocessing text...")
    
    # Combine subject and body for richer text representation
    df["combined_text"] = df["subject"] + " " + df["body"]
    
    # Apply preprocessing
    df["clean_text"] = df["combined_text"].apply(preprocess_text)
    
    print(f"✅ Text preprocessing complete")
    print(f"\n📝 Sample preprocessed text:")
    print(f"Original: {df['combined_text'].iloc[0]}")
    print(f"Cleaned:  {df['clean_text'].iloc[0]}")
    
    return df


# ============================================================================
# VECTORIZATION & TRAIN/TEST SPLIT
# ============================================================================

def vectorize_text(df, fit=True):
    """
    Apply TF-IDF vectorization.
    
    TF-IDF: Term Frequency - Inverse Document Frequency
    - Gives higher weight to important terms
    - Reduces impact of common words
    """
    print("\n🔢 Vectorizing text with TF-IDF...")
    
    vectorizer = TfidfVectorizer(
        max_features=CONFIG["tfidf_max_features"],
        ngram_range=CONFIG["tfidf_ngram_range"],
        lowercase=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(df["clean_text"])
    
    print(f"✅ Vectorization complete")
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"   - Samples: {X.shape[0]}")
    print(f"   - Features (TF-IDF): {X.shape[1]}")
    
    return X, vectorizer


def split_data(X, y):
    """Split into train/test sets."""
    print(f"\n✂️  Splitting data (80/20)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y  # Maintain category distribution
    )
    
    print(f"✅ Split complete")
    print(f"   - Training: {X_train.shape[0]} samples")
    print(f"   - Testing: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train):
    """Train Multinomial Naive Bayes classifier."""
    print("\n🧠 Training Multinomial Naive Bayes...")
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    print(f"✅ Model training complete")
    print(f"   - Classes: {model.classes_}")
    print(f"   - Number of features: {model.feature_log_prob_.shape[1]}")
    
    return model


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation:
    - Accuracy
    - Precision, Recall, F1-score (per class)
    - Confusion Matrix
    """
    print("\n📊 Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probabilities for confidence scoring
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    # Classification report
    print(f"\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔀 Confusion Matrix:")
    categories = model.classes_
    print(f"   {'Predicted →':>12} " + " ".join(f"{cat:>10}" for cat in categories))
    for i, cat in enumerate(categories):
        print(f"{cat:>12} " + " ".join(f"{cm[i][j]:>10}" for j in range(len(categories))))
    
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }


def calculate_confidence_stats(y_proba):
    """Analyze confidence distribution."""
    max_proba = y_proba.max(axis=1)
    
    print(f"\n📉 Confidence Statistics:")
    print(f"   - Mean:   {max_proba.mean():.4f}")
    print(f"   - Median: {np.median(max_proba):.4f}")
    print(f"   - Min:    {max_proba.min():.4f}")
    print(f"   - Max:    {max_proba.max():.4f}")
    print(f"   - Std:    {max_proba.std():.4f}")
    
    # Count predictions needing human review
    needs_review = (max_proba < CONFIG["confidence_threshold"]).sum()
    pct_review = (needs_review / len(max_proba)) * 100
    print(f"\n👤 Human Review Threshold ({CONFIG['confidence_threshold']*100:.0f}%):")
    print(f"   - Needs review: {needs_review} ({pct_review:.2f}%)")
    print(f"   - Auto-assigned: {len(max_proba) - needs_review} ({100-pct_review:.2f}%)")


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(model, vectorizer):
    """Save trained model and vectorizer."""
    print("\n💾 Saving model and vectorizer...")
    
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    
    # Save config
    with open("models/config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    
    print(f"✅ Model saved to models/model.pkl")
    print(f"✅ Vectorizer saved to models/vectorizer.pkl")
    print(f"✅ Config saved to models/config.json")


def save_training_metadata(model, results):
    """Save training metadata for reference."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "MultinomialNB",
        "vectorizer_type": "TfidfVectorizer",
        "accuracy": float(results["accuracy"]),
        "config": CONFIG,
        "classes": list(model.classes_)
    }
    
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to models/metadata.json")


# ============================================================================
# INFERENCE / PREDICTION
# ============================================================================

def predict_ticket(ticket_text, model, vectorizer):
    """
    Predict ticket category with confidence score.
    
    Returns:
    - category: Predicted category
    - confidence: Confidence probability
    - status: "Auto Assigned" or "Needs Human Review"
    """
    # Preprocess
    clean_text = preprocess_text(ticket_text)
    
    # Vectorize
    X_vec = vectorizer.transform([clean_text])
    
    # Predict
    category = model.predict(X_vec)[0]
    confidence = model.predict_proba(X_vec).max()
    
    # Determine status
    status = "Auto Assigned" if confidence >= CONFIG["confidence_threshold"] else "Needs Human Review"
    
    return {
        "category": category,
        "confidence": confidence,
        "status": status
    }


def test_predictions(model, vectorizer):
    """Test predictions on sample tickets."""
    print("\n🧪 Testing predictions on sample tickets...")
    print(f"{'='*70}")
    
    test_tickets = [
        "I was charged twice for my subscription this month.",
        "The application crashes when I try to upload files.",
        "Please send my salary slip for this month.",
        "What are your office working hours?",
        "Help help help"  # Low confidence example
    ]
    
    for ticket in test_tickets:
        result = predict_ticket(ticket, model, vectorizer)
        print(f"\n📄 Ticket: {ticket[:50]}...")
        print(f"   🏷️  Category: {result['category']}")
        print(f"   📊 Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"   ✅ Status: {result['status']}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute complete training pipeline."""
    print("\n" + "="*70)
    print("🚀 SUPPORT TICKET CLASSIFIER - TRAINING PIPELINE")
    print("="*70)
    
    # 1. Load data
    df = load_data()
    
    # 2. Preprocess
    df = prepare_data(df)
    
    # 3. Vectorize
    X, vectorizer = vectorize_text(df)
    y = df["category"]
    
    # 4. Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 5. Train
    model = train_model(X_train, y_train)
    
    # 6. Evaluate
    results = evaluate_model(model, X_test, y_test)
    calculate_confidence_stats(results["y_proba"])
    
    # 7. Save
    save_model(model, vectorizer)
    save_training_metadata(model, results)
    
    # 8. Test
    test_predictions(model, vectorizer)
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("="*70)
    print("\n📁 Project structure:")
    print("   data/tickets.csv          → Training data")
    print("   models/model.pkl          → Trained model")
    print("   models/vectorizer.pkl     → TF-IDF vectorizer")
    print("   models/config.json        → Configuration")
    print("   models/metadata.json      → Training metadata")
    print("\n🚀 Next: Run 'streamlit run app.py' to start the web interface")


if __name__ == "__main__":
    main()
