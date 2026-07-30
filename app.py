"""
Support Ticket Classifier - Streamlit Web App
==============================================
Interactive interface for classifying support tickets
- Real-time predictions
- Confidence scoring
- Human review flagging
- Batch processing
"""

import streamlit as st
import joblib
import json
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd

# Download NLTK data
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 20px;
    }
    .stSuccess {
        background-color: #d4edda;
    }
    .stWarning {
        background-color: #fff3cd;
    }
    .stError {
        background-color: #f8d7da;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL & VECTORIZER
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load model, vectorizer, and config from disk."""
    try:
        model = joblib.load("models/model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        
        # Load config
        with open("models/config.json", "r") as f:
            config = json.load(f)
        
        # Load metadata
        with open("models/metadata.json", "r") as f:
            metadata = json.load(f)
        
        return model, vectorizer, config, metadata
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run train.py first.")
        st.stop()


model, vectorizer, config, metadata = load_model_artifacts()

# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_text(text):
    """Preprocess text (same as training)."""
    stop_words = set(stopwords.words("english"))
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words and w.strip()]
    
    return " ".join(words)


# ============================================================================
# PREDICTION
# ============================================================================

def predict_ticket(ticket_text):
    """Predict ticket category with confidence."""
    if not ticket_text.strip():
        return None
    
    # Preprocess
    clean_text = preprocess_text(ticket_text)
    
    if not clean_text.strip():
        return None
    
    # Vectorize
    X_vec = vectorizer.transform([clean_text])
    
    # Predict
    category = model.predict(X_vec)[0]
    probabilities = model.predict_proba(X_vec)[0]
    confidence = probabilities.max()
    
    # Confidence scores for all classes
    class_scores = {
        model.classes_[i]: float(probabilities[i])
        for i in range(len(model.classes_))
    }
    
    # Status
    threshold = config["confidence_threshold"]
    status = "Auto Assigned" if confidence >= threshold else "Needs Human Review"
    
    return {
        "category": category,
        "confidence": confidence,
        "status": status,
        "class_scores": class_scores
    }


# ============================================================================
# UI - HEADER
# ============================================================================

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("🎫 Support Ticket Classifier")
    st.markdown("**Intelligent ticket classification with confidence scoring**")

with col2:
    st.markdown(f"""
    **Model Info:**
    - Type: Multinomial Naive Bayes + TF-IDF
    - Accuracy: {metadata['accuracy']:.2%}
    - Classes: {len(metadata['classes'])}
    """)

st.divider()

# ============================================================================
# SIDEBAR - INFO & SETTINGS
# ============================================================================

with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This classifier uses:
    - **TF-IDF Vectorization** for text representation
    - **Multinomial Naive Bayes** for classification
    - **Confidence Threshold** for quality control
    """)
    
    st.divider()
    
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=config["confidence_threshold"],
        step=0.05,
        help="Predictions below this threshold → 'Needs Human Review'"
    )
    
    st.divider()
    
    st.header("📊 Model Details")
    st.json({
        "classes": metadata["classes"],
        "timestamp": metadata["timestamp"][:10],
        "config": config
    })

# ============================================================================
# MAIN INTERFACE - TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📋 Batch Processing", "📈 Statistics"])

# TAB 1: Single Prediction
with tab1:
    st.subheader("Classify a Single Ticket")
    
    col_subject, col_body = st.columns(2)
    
    with col_subject:
        subject = st.text_input(
            "Ticket Subject",
            placeholder="e.g., Payment failed",
            help="Brief title of the ticket"
        )
    
    with col_body:
        body = st.text_area(
            "Ticket Body",
            placeholder="e.g., I was charged twice for my subscription...",
            height=120,
            help="Detailed description of the issue"
        )
    
    if st.button("🔍 Classify", use_container_width=True, type="primary"):
        if not subject or not body:
            st.warning("⚠️ Please enter both subject and body")
        else:
            combined_text = f"{subject} {body}"
            result = predict_ticket(combined_text)
            
            if result:
                # Display results
                col_pred, col_conf, col_status = st.columns(3)
                
                with col_pred:
                    st.metric("📂 Category", result["category"])
                
                with col_conf:
                    conf_pct = result["confidence"] * 100
                    st.metric("📊 Confidence", f"{conf_pct:.1f}%")
                
                with col_status:
                    if result["status"] == "Auto Assigned":
                        st.metric("✅ Status", result["status"])
                    else:
                        st.metric("👤 Status", result["status"])
                
                st.divider()
                
                # Show scores for all categories
                st.subheader("📈 Confidence by Category")
                
                # Create DataFrame for visualization
                scores_df = pd.DataFrame(
                    list(result["class_scores"].items()),
                    columns=["Category", "Confidence"]
                ).sort_values("Confidence", ascending=False)
                
                # Bar chart
                st.bar_chart(scores_df.set_index("Category"))
                
                # Table
                st.dataframe(
                    scores_df.assign(Confidence=lambda x: x["Confidence"].apply(lambda y: f"{y:.2%}")),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Recommendation
                st.divider()
                if result["status"] == "Needs Human Review":
                    st.warning(
                        f"⚠️ **Low confidence ({result['confidence']:.1%})**\n\n"
                        f"This ticket should be reviewed by a human agent before auto-assignment.\n\n"
                        f"Top match: **{result['category']}** ({result['confidence']:.1%})"
                    )
                else:
                    st.success(
                        f"✅ **High confidence ({result['confidence']:.1%})**\n\n"
                        f"Recommended assignment: **{result['category']}**"
                    )
            else:
                st.error("❌ Could not process ticket. Ensure text contains meaningful words.")

# TAB 2: Batch Processing
with tab2:
    st.subheader("Classify Multiple Tickets")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with 'subject' and 'body' columns",
        type=["csv"]
    )
    
    if uploaded_file:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ["subject", "body"]
            if not all(col in df.columns for col in required_cols):
                st.error(f"❌ CSV must contain columns: {required_cols}")
            else:
                st.success(f"✅ Loaded {len(df)} tickets")
                
                # Process
                if st.button("🚀 Process All", use_container_width=True, type="primary"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for idx, row in df.iterrows():
                        combined = f"{row['subject']} {row['body']}"
                        result = predict_ticket(combined)
                        
                        if result:
                            results.append({
                                "Subject": row["subject"][:50],
                                "Category": result["category"],
                                "Confidence": f"{result['confidence']:.2%}",
                                "Status": result["status"]
                            })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    # Display results
                    st.divider()
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="classified_tickets.csv",
                        mime="text/csv"
                    )
                    
                    # Statistics
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        auto = (results_df["Status"] == "Auto Assigned").sum()
                        st.metric("Auto Assigned", auto, f"{(auto/len(results_df)*100):.1f}%")
                    
                    with col2:
                        review = (results_df["Status"] == "Needs Human Review").sum()
                        st.metric("Needs Review", review, f"{(review/len(results_df)*100):.1f}%")
                    
                    with col3:
                        st.metric("Total Processed", len(results_df))
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# TAB 3: Statistics
with tab3:
    st.subheader("Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
    
    with col2:
        st.metric("Categories", len(metadata["classes"]))
    
    with col3:
        st.metric("Threshold", f"{config['confidence_threshold']:.0%}")
    
    with col4:
        st.metric("Max Features", f"{config['tfidf_max_features']:,}")
    
    st.divider()
    
    st.subheader("📋 Categories")
    categories_df = pd.DataFrame({
        "Category": metadata["classes"]
    })
    st.dataframe(categories_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("⚙️ Configuration")
    st.json(config)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption(
    "🤖 Built with TF-IDF + Multinomial Naive Bayes | "
    f"Trained: {metadata['timestamp'][:10]}"
)
