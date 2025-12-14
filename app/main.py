import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (Embedding, SpatialDropout1D, Bidirectional,
                                     LSTM, Conv1D, GlobalMaxPooling1D,
                                     Dense, Dropout, Input, Attention)

# ==========================================
# 0. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Emotion & Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. CLEANING FUNCTION
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==========================================
# 2. LOAD DATASET (Cached)
# ==========================================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    train_path = os.path.join(project_root, 'data', 'train.csv')
    
    df_train = None

    # Load Train Data for EDA
    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
        df_train.dropna(subset=['tweets', 'class'], inplace=True)
        df_train = df_train[df_train['class'] != 'figurative']
        df_train['clean_text'] = df_train['tweets'].apply(clean_text)
        
    return df_train

# ==========================================
# 3. LOAD MODEL & ASSETS (Cached)
# ==========================================
@st.cache_resource
def load_model_and_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    tokenizer_path = os.path.join(project_root, 'assets', 'tokenizer.json')
    label_encoder_path = os.path.join(project_root, 'assets', 'label_encoder.pkl')
    model_path = os.path.join(project_root, 'assets', 'best_bilstm_attention.h5')

    # LOAD TOKENIZER
    try:
        with open(tokenizer_path, 'r') as f:
            public_tokenizer_data = f.read()
            tokenizer = tokenizer_from_json(public_tokenizer_data)
    except FileNotFoundError:
        st.error("❌ Tokenizer not found.")
        return None  # FIXED: Return single None on error

    # LOAD LABEL ENCODER
    try:
        with open(label_encoder_path, 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Label Encoder not found.")
        return None  # FIXED: Return single None on error
    
    label2id = {label: idx for idx, label in enumerate(le.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}

    # DEFINE MODEL ARCHITECTURE
    vocab_size = 30000 
    embedding_dim = 200    
    max_len = 80            
    num_classes = len(le.classes_) 

    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, trainable=False)(inputs)
    x = SpatialDropout1D(0.25)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    attn = Attention()([x, x])
    x = GlobalMaxPooling1D()(attn)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # LOAD WEIGHTS
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            return None  # FIXED: Return single None on error
        model.load_weights(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model weights: {e}")
        return None  # FIXED: Return single None on error
    
    return model, tokenizer, id2label, max_len, le # Success returns 5 values

# ==========================================
# 4. PAGE: EDA
# ==========================================
def show_eda_page(df_train):
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    if df_train is None:
        st.warning("Training Data not found.")
        return

    st.write(f"**Total Training Data (Cleaned):** {len(df_train)} rows")
    st.write("Data filtered: 'figurative' class removed & text cleaned.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df_train, x='class', palette='viridis', ax=ax)
        plt.title("Count per Class")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Text Length Distribution")
        df_train['word_count'] = df_train['clean_text'].apply(lambda x: len(str(x).split()))
        fig, ax = plt.subplots()
        sns.histplot(df_train['word_count'], bins=30, kde=True, color='purple', ax=ax)
        plt.title("Word Count Distribution")
        st.pyplot(fig)

    st.subheader("Data Samples")
    st.dataframe(df_train[['tweets', 'clean_text', 'class']].head(5))

# ==========================================
# 5. PAGE: EVALUATION RESULTS (IMAGES)
# ==========================================
def show_evaluation_page():
    st.header("📈 Model Evaluation Results")
    st.info("The following results are obtained from the test dataset evaluation.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Path gambar
    cls_report_path = os.path.join(project_root, 'assets', 'classification_report.png')
    cm_path = os.path.join(project_root, 'assets', 'confusion_matrix.png')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Classification Report")
        if os.path.exists(cls_report_path):
            st.image(cls_report_path, caption="Precision, Recall, F1-Score per Class", use_container_width=True)
        else:
            st.error(f"Image not found: {cls_report_path}")
            st.warning("Please save your classification report screenshot as 'classification_report.png' in the 'assets' folder.")

    with col2:
        st.subheader("2. Confusion Matrix")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix Visualization", use_container_width=True)
        else:
            st.error(f"Image not found: {cm_path}")
            st.warning("Please save your confusion matrix screenshot as 'confusion_matrix.png' in the 'assets' folder.")

# ==========================================
# 6. PAGE: INPUT & PREDICTION
# ==========================================
def show_prediction_page(model, tokenizer, id2label, max_len):
    st.header("🤖 Model Playground (Prediction)")
    st.caption("Architecture: BiLSTM + Attention Mechanism")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("Enter text:", height=150, placeholder="Example: I love waiting 2 hours for my food. #sarcasm")
        
        if st.button("Predict", type="primary"):
            if user_input.strip() != "":
                cleaned = clean_text(user_input)
                seq = tokenizer.texts_to_sequences([cleaned])
                pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
                
                probs = model.predict(pad)[0]
                pred_id = probs.argmax()
                pred_label = id2label[pred_id]
                confidence = probs.max() * 100
                
                st.success(f"Prediction: **{str(pred_label).upper()}**")
                st.metric("Confidence Score", f"{confidence:.2f}%")
                
                with st.expander("View Preprocessing Result"):
                    st.code(cleaned)
            else:
                st.warning("Please enter some text first.")

    with col2:
        if user_input.strip() != "" and 'probs' in locals():
            st.markdown("##### Class Probabilities")
            prob_df = pd.DataFrame({
                "Class": [str(x).capitalize() for x in id2label.values()],
                "Probability": probs
            })
            st.bar_chart(prob_df.set_index("Class"))

# ==========================================
# MAIN APP LOGIC
# ==========================================
def main():
    # Load Assets
    load_result = load_model_and_assets()
    
    # Check if load failed (result is None)
    if load_result is None:
        st.stop() # Stop execution here if assets missing
        
    # Unpack safely (because we know it's not None and has 5 elements)
    model, tokenizer, id2label, max_len, le = load_result

    # Load Data (Train only)
    df_train = load_data()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page:", [
        "1. EDA (Exploratory Data Analysis)", 
        "2. Model Evaluation Results", 
        "3. Prediction & Demo"
    ])

    st.sidebar.divider()
    st.sidebar.info("Project: Deep Learning Emotion Detection\nModel: BiLSTM Attention")

    # Page Routing
    if page == "1. EDA (Exploratory Data Analysis)":
        show_eda_page(df_train)
    elif page == "2. Model Evaluation Results":
        show_evaluation_page()
    elif page == "3. Prediction & Demo":
        show_prediction_page(model, tokenizer, id2label, max_len)

if __name__ == '__main__':
    main()
