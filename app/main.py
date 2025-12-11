import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import pickle
import json
import os
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (Embedding, SpatialDropout1D, Bidirectional,
                                     LSTM, Conv1D, GlobalMaxPooling1D,
                                     Dense, Dropout, Input)

# ==========================================
# 1. TEXT CLEANING FUNCTION
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)       # Remove URLs
    text = re.sub(r'@\w+', ' ', text)                   # Remove mentions
    text = re.sub(r'#', ' ', text)                      # Remove hashtags symbol
    text = re.sub(r'\s+', ' ', text)                    # Remove double spaces
    text = text.strip()
    return text

# ==========================================
# 2. LOAD MODEL & ASSETS
# ==========================================
@st.cache_resource
def load_model_and_assets():
    # --- PATH CONFIGURATION ---
    # Get current directory (.../app)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to project root (.../deep-learning-final-project)
    project_root = os.path.dirname(current_dir)
    
    # Define paths to assets
    tokenizer_path = os.path.join(project_root, 'assets', 'tokenizer.json')
    label_encoder_path = os.path.join(project_root, 'assets', 'label_encoder.pkl')
    model_path = os.path.join(project_root, 'assets', 'best_cnn_bilstm.h5')

    # A. LOAD TOKENIZER
    try:
        with open(tokenizer_path, 'r') as f:
            public_tokenizer_data = f.read()
            tokenizer = tokenizer_from_json(public_tokenizer_data)
    except FileNotFoundError:
        st.error(f"❌ File not found: {tokenizer_path}")
        return None, None, None, None

    # B. LOAD LABEL ENCODER
    try:
        with open(label_encoder_path, 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ File not found: {label_encoder_path}")
        return None, None, None, None
    
    label2id = {label: idx for idx, label in enumerate(le.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}

    # C. DEFINE MODEL ARCHITECTURE
    # (Must match the saved .h5 file structure)
    vocab_size = 30000     
    embedding_dim = 200    
    max_len = 40           
    num_classes = len(le.classes_) 

    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    x = SpatialDropout1D(0.4)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x) # 32 units
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x) # 64 filters
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x) # 64 units
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # D. LOAD WEIGHTS
    try:
        model.load_weights(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model weights: {e}")
        return None, None, None, None
    
    return model, tokenizer, id2label, max_len

# ==========================================
# 3. STREAMLIT UI (ENGLISH VERSION)
# ==========================================
def main_streamlit():
    st.set_page_config(page_title="Deep Learning Emotion Detection", layout="centered")
    
    st.title("🔮 Sentiment & Emotion Analysis")
    st.caption("Architecture: Embedding + BiLSTM (32) + CNN (64) + Dense (64)")

    # Load Model
    model, tokenizer, id2label, max_len = load_model_and_assets()

    if model is None:
        st.warning("Application cannot start because asset files are missing.")
        st.stop()

    # --- FEATURE 1: MANUAL PREDICTION ---
    st.subheader("Try it out")
    user_input = st.text_area("Enter text to analyze:", height=100, placeholder="Example: I am so happy today because the weather is nice!")

    if st.button("Predict"):
        if user_input.strip() != "":
            # Preprocessing & Prediction
            cleaned = clean_text(user_input)
            seq = tokenizer.texts_to_sequences([cleaned])
            pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
            
            probs = model.predict(pad)[0]
            pred_id = probs.argmax()
            pred_label = id2label[pred_id]
            
            # Display Results
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Prediction")
                st.success(f"**{pred_label.upper()}**")
                confidence = probs.max() * 100
                st.metric("Confidence Score", f"{confidence:.2f}%")
            
            with col2:
                st.markdown("### Probability Distribution")
                prob_df = pd.DataFrame({
                    "Class": list(id2label.values()),
                    "Probability": probs
                })
                # Capitalize labels for better display
                prob_df["Class"] = prob_df["Class"].str.capitalize()
                st.bar_chart(prob_df.set_index("Class"))
        else:
            st.warning("Please enter some text first.")

    # --- FEATURE 2: BATCH ANALYSIS (TEST.CSV) ---
    st.divider()
    st.subheader("📊 Test Data Analysis (Batch Prediction)")
    
    if st.checkbox("Show Top Confidence Examples from Test Data"):
        try:
            # --- PATH LOGIC FOR DATA ---
            current_dir = os.path.dirname(os.path.abspath(__file__)) 
            project_root = os.path.dirname(current_dir)              
            
            test_path = os.path.join(project_root, "data", "test.csv")
            
            if not os.path.exists(test_path):
                st.error(f"File test.csv not found at: {test_path}")
            else:
                df_test = pd.read_csv(test_path)
                st.info(f"Processing {len(df_test)} samples from test.csv...")
                
                # Batch Preprocessing
                progress_bar = st.progress(0)
                
                texts = df_test['tweets'].astype(str).tolist()
                cleaned_texts = [clean_text(t) for t in texts]
                
                seqs = tokenizer.texts_to_sequences(cleaned_texts)
                pads = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
                
                progress_bar.progress(50)
                
                # Batch Prediction
                predictions = model.predict(pads, verbose=0)
                progress_bar.progress(100)
                
                # Process Results
                pred_indices = np.argmax(predictions, axis=1)
                confidences = np.max(predictions, axis=1)
                pred_labels = [id2label[i] for i in pred_indices]
                
                df_results = df_test.copy()
                df_results['pred_label'] = pred_labels
                df_results['confidence'] = confidences
                
                # Display Top 3 per Category
                st.write("### 🔥 Top 3 Highest Confidence Examples per Category")
                
                unique_labels = sorted(list(set(pred_labels)))
                
                for label in unique_labels:
                    st.markdown(f"#### Category: **{label.upper()}**")
                    
                    # Filter & Sort
                    top_df = df_results[df_results['pred_label'] == label].sort_values(by='confidence', ascending=False).head(3)
                    
                    for _, row in top_df.iterrows():
                        with st.expander(f"🎯 {row['confidence']*100:.1f}% Confidence - {row['tweets'][:60]}..."):
                            st.write(f"**Original Text:** {row['tweets']}")
                            st.write(f"**True Label:** {row.get('class', 'N/A')}")
                            st.write(f"**Predicted:** {row['pred_label']}")
                            st.progress(float(row['confidence']))
                            
        except Exception as e:
            st.error(f"An error occurred while processing data: {e}")

if __name__ == '__main__':
    try:
        main_streamlit()
    except Exception as e:
        st.error(f"Runtime Error: {e}")
