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
    layout="wide", # Wide layout for better visualization
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. CLEANING FUNCTION (Matches main3.ipynb)
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)   # Completely remove hashtags
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
    test_path = os.path.join(project_root, 'data', 'test.csv')
    
    df_train = None
    df_test = None

    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
        # Filter according to main3.ipynb logic
        df_train.dropna(subset=['tweets', 'class'], inplace=True)
        df_train = df_train[df_train['class'] != 'figurative']
        df_train['clean_text'] = df_train['tweets'].apply(clean_text)
    
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        df_test.dropna(subset=['tweets', 'class'], inplace=True) # Ensure class exists for evaluation
        df_test = df_test[df_test['class'] != 'figurative']
        df_test['clean_text'] = df_test['tweets'].apply(clean_text)
        
    return df_train, df_test

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

    # Load Tokenizer
    try:
        with open(tokenizer_path, 'r') as f:
            public_tokenizer_data = f.read()
            tokenizer = tokenizer_from_json(public_tokenizer_data)
    except FileNotFoundError:
        st.error("❌ Tokenizer not found.")
        return None, None, None, None

    # Load Label Encoder
    try:
        with open(label_encoder_path, 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Label Encoder not found.")
        return None, None, None, None
    
    label2id = {label: idx for idx, label in enumerate(le.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}

    # Model Architecture (Attention BiLSTM)
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

    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            return None, None, None, None
        model.load_weights(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model weights: {e}")
        return None, None, None, None
    
    return model, tokenizer, id2label, max_len, le

# ==========================================
# 4. PAGE: EDA
# ==========================================
def show_eda_page(df_train):
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    if df_train is None:
        st.warning("Training Data not found. Ensure 'data/train.csv' exists.")
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
        # Calculate word count
        df_train['word_count'] = df_train['clean_text'].apply(lambda x: len(str(x).split()))
        fig, ax = plt.subplots()
        sns.histplot(df_train['word_count'], bins=30, kde=True, color='purple', ax=ax)
        plt.title("Word Count Distribution per Tweet")
        st.pyplot(fig)

    st.subheader("Data Samples (Raw vs Cleaned)")
    st.dataframe(df_train[['tweets', 'clean_text', 'class']].head(10))

# ==========================================
# 5. PAGE: EVALUATION RESULTS
# ==========================================
def show_evaluation_page(model, tokenizer, max_len, le, df_test):
    st.header("📈 Model Evaluation Results")

    # 1. Training vs Validation Loss
    st.subheader("1. Training vs Validation Loss")
    st.info("⚠️ Historical Loss charts are only available in the Notebook. Here we display real-time evaluation on Test Data.")
    
    if df_test is None:
        st.warning("Test Data not found for evaluation.")
        return

    # Run Bulk Prediction on Test Data
    if st.button("Run Evaluation on Test Data (May take a moment)"):
        with st.spinner("Processing predictions..."):
            # Preprocessing
            texts = df_test['clean_text'].astype(str).tolist()
            seqs = tokenizer.texts_to_sequences(texts)
            pads = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
            
            # Predict
            y_pred_probs = model.predict(pads, verbose=0)
            y_pred = y_pred_probs.argmax(axis=1)
            
            # True Labels
            y_true = le.transform(df_test['class'])
            target_names = [str(cls) for cls in le.classes_]

            # 2. Classification Report
            st.subheader("2. Classification Report")
            report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
            df_report = pd.DataFrame(report_dict).transpose()
            st.dataframe(df_report.style.highlight_max(axis=0))

            # 3. Confusion Matrix
            st.subheader("3. Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)

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
                
                # Expanders for details
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
    if load_result is None:
        st.stop()
    model, tokenizer, id2label, max_len, le = load_result

    # Load Data
    df_train, df_test = load_data()

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
        show_evaluation_page(model, tokenizer, max_len, le, df_test)
    elif page == "3. Prediction & Demo":
        show_prediction_page(model, tokenizer, id2label, max_len)

if __name__ == '__main__':
    main()
