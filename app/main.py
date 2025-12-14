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
                                     Dense, Dropout, Input, Attention) # Tambah Attention

# ==========================================
# 1. TEXT CLEANING FUNCTION
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    # PENTING: Sesuai main3.ipynb, ini menghapus '#sarcasm' sepenuhnya, bukan cuma '#'
    text = re.sub(r"#\w+", " ", text)   
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==========================================
# 2. LOAD MODEL & ASSETS
# ==========================================
@st.cache_resource
def load_model_and_assets():
    # --- PATH CONFIGURATION ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Paths
    tokenizer_path = os.path.join(project_root, 'assets', 'tokenizer.json')
    label_encoder_path = os.path.join(project_root, 'assets', 'label_encoder.pkl')
    model_path = os.path.join(project_root, 'assets', 'best_bilstm_attention.h5')

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

    # C. DEFINE MODEL ARCHITECTURE (Sesuai main3.ipynb)
    # ---------------------------------------------------------
    # Parameter dari notebook
    vocab_size = 30000 
    embedding_dim = 200    
    max_len = 80            # Diupdate dari 40 ke 80
    num_classes = len(le.classes_) 

    # Functional API Definition
    inputs = Input(shape=(max_len,))
    
    # Layer 1: Embedding
    # Note: Kita tidak perlu load GloVe txt manual disini, 
    # karena bobot GloVe sudah tersimpan di dalam file .h5
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, trainable=False)(inputs)
    
    # Layer 2: SpatialDropout
    x = SpatialDropout1D(0.25)(x)
    
    # Layer 3: Bi-LSTM
    # return_sequences=True diperlukan untuk Attention layer
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    
    # Layer 4: Self-Attention
    # Attention layer menerima [query, value] yang sama (self-attention)
    attn = Attention()([x, x])
    
    # Layer 5: Pooling
    x = GlobalMaxPooling1D()(attn)
    
    # Layer 6: Dense
    x = Dense(128, activation='relu')(x)
    
    # Layer 7: Dropout
    x = Dropout(0.4)(x)
    
    # Layer 8: Output
    outputs = Dense(num_classes, activation='softmax')(x)
    # ---------------------------------------------------------

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # D. LOAD WEIGHTS
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            return None, None, None, None
            
        model.load_weights(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model weights: {e}")
        st.warning("Ensure you have uploaded the NEW 'best_bilstm_attention.h5' and 'label_encoder.pkl' generated from main3.ipynb.")
        return None, None, None, None
    
    return model, tokenizer, id2label, max_len

# ==========================================
# 3. STREAMLIT UI
# ==========================================
def main_streamlit():
    st.set_page_config(page_title="Deep Learning Emotion Detection", layout="centered")
    
    st.title("🔮 Sentiment & Emotion Analysis")
    st.caption("Model Architecture: BiLSTM (64) + Attention + Dense (128)")
    
    

    # Load Model
    model, tokenizer, id2label, max_len = load_model_and_assets()

    if model is None:
        st.warning("Application cannot start because asset files are missing.")
        st.stop()

    # --- FEATURE 1: MANUAL PREDICTION ---
    st.subheader("Try it out")
    user_input = st.text_area("Enter text to analyze:", height=100, placeholder="Example: I am so happy today!")

    if st.button("Predict"):
        if user_input.strip() != "":
            # Preprocessing
            cleaned = clean_text(user_input)
            
            # Tokenizing & Padding
            seq = tokenizer.texts_to_sequences([cleaned])
            pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
            
            # Prediction
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
                prob_df["Class"] = prob_df["Class"].str.capitalize()
                st.bar_chart(prob_df.set_index("Class"))
        else:
            st.warning("Please enter some text first.")

    # --- FEATURE 2: BATCH ANALYSIS ---
    st.divider()
    st.subheader("📊 Test Data Analysis (Batch Prediction)")
    
    if st.checkbox("Show Top Confidence Examples from Test Data"):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__)) 
            project_root = os.path.dirname(current_dir)              
            test_path = os.path.join(project_root, "data", "test.csv")
            
            if not os.path.exists(test_path):
                st.error(f"File test.csv not found at: {test_path}")
            else:
                df_test = pd.read_csv(test_path)
                st.info(f"Processing {len(df_test)} samples from test.csv...")
                
                # Filter data to match training logic (remove figurative if present in test)
                # Note: In main3.ipynb, test set was also filtered.
                df_test_filtered = df_test[df_test['class'] != 'figurative'].copy()
                
                if df_test_filtered.empty:
                    st.warning("No data left after filtering 'figurative' class.")
                else:
                    progress_bar = st.progress(0)
                    
                    texts = df_test_filtered['tweets'].astype(str).tolist()
                    cleaned_texts = [clean_text(t) for t in texts]
                    
                    seqs = tokenizer.texts_to_sequences(cleaned_texts)
                    pads = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
                    
                    progress_bar.progress(50)
                    
                    predictions = model.predict(pads, verbose=0)
                    progress_bar.progress(100)
                    
                    pred_indices = np.argmax(predictions, axis=1)
                    confidences = np.max(predictions, axis=1)
                    pred_labels = [id2label[i] for i in pred_indices]
                    
                    df_results = df_test_filtered.copy()
                    df_results['pred_label'] = pred_labels
                    df_results['confidence'] = confidences
                    
                    st.write("### 🔥 Top 3 Highest Confidence Examples per Category")
                    unique_labels = sorted(list(set(pred_labels)))
                    
                    for label in unique_labels:
                        st.markdown(f"#### Category: **{label.upper()}**")
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
