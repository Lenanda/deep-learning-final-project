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
# 1. FUNGSI CLEANING
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)       # URL
    text = re.sub(r'@\w+', ' ', text)                   # mention
    text = re.sub(r'#', ' ', text)                      # buang simbol #
    text = re.sub(r'\s+', ' ', text)                    # spasi dobel
    text = text.strip()
    return text

# ==========================================
# 2. LOAD MODEL & ASET
# ==========================================
@st.cache_resource
def load_model_and_assets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(script_dir, 'assets', 'tokenizer.json')
    label_encoder_path = os.path.join(script_dir, 'assets', 'label_encoder.pkl')
    model_path = os.path.join(script_dir, 'assets', 'best_cnn_bilstm.h5')

    # A. LOAD TOKENIZER
    try:
        with open(tokenizer_path, 'r') as f:
            public_tokenizer_data = f.read()
            tokenizer = tokenizer_from_json(public_tokenizer_data)
    except FileNotFoundError:
        st.error("❌ File 'tokenizer.json' tidak ditemukan.")
        return None, None, None, None

    # B. LOAD LABEL ENCODER
    try:
        with open(label_encoder_path, 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ File 'label_encoder.pkl' tidak ditemukan.")
        return None, None, None, None
    
    label2id = {label: idx for idx, label in enumerate(le.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}

    # C. DEFINISI ARSITEKTUR MODEL (Fixed according to H5)
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
        st.error(f"❌ Gagal memuat bobot model: {e}")
        return None, None, None, None
    
    return model, tokenizer, id2label, max_len

# ==========================================
# 3. UI STREAMLIT
# ==========================================
def main_streamlit():
    st.set_page_config(page_title="Deep Learning Emotion Detection", layout="centered")
    
    st.title("🔮 Analisis Sentimen & Emosi")
    st.caption("Model: BiLSTM (32) + CNN (64) + Dense (64)")

    # Load Model (Wajib di awal)
    model, tokenizer, id2label, max_len = load_model_and_assets()

    if model is None:
        st.warning("Aplikasi tidak dapat berjalan karena file aset bermasalah.")
        st.stop()

    # --- FITUR 1: PREDIKSI MANUAL ---
    st.subheader("Coba Prediksi")
    user_input = st.text_area("Masukkan teks:", height=100, placeholder="Contoh: Saya sangat senang hari ini!")

    if st.button("Prediksi"):
        if user_input.strip() != "":
            # Preprocessing & Prediksi
            cleaned = clean_text(user_input)
            seq = tokenizer.texts_to_sequences([cleaned])
            pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
            
            probs = model.predict(pad)[0]
            pred_id = probs.argmax()
            pred_label = id2label[pred_id]
            
            # Tampilan Hasil
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Hasil")
                st.success(f"**{pred_label.upper()}**")
                confidence = probs.max() * 100
                st.metric("Confidence", f"{confidence:.2f}%")
            
            with col2:
                st.markdown("### Probabilitas")
                prob_df = pd.DataFrame({
                    "Label": list(id2label.values()),
                    "Probability": probs
                })
                st.bar_chart(prob_df.set_index("Label"))
        else:
            st.warning("Mohon masukkan teks.")

    # --- FITUR 2: ANALISIS TEST.CSV (BATCH) ---
    st.divider()
    st.subheader("📊 Analisis Data Test (Batch Prediction)")
    
    if st.checkbox("Tampilkan 12 Contoh dengan Confidence Terbaik"):
        try:
            # Cari file test.csv di folder yang sama dengan script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            test_path = os.path.join(script_dir, "data", "test.csv")
            
            if not os.path.exists(test_path):
                st.error("File test.csv tidak ditemukan di folder aplikasi.")
            else:
                df_test = pd.read_csv(test_path)
                st.info(f"Memproses {len(df_test)} data dari test.csv...")
                
                # Preprocessing Massal
                progress_bar = st.progress(0)
                
                texts = df_test['tweets'].astype(str).tolist()
                cleaned_texts = [clean_text(t) for t in texts]
                
                seqs = tokenizer.texts_to_sequences(cleaned_texts)
                pads = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
                
                progress_bar.progress(50)
                
                # Prediksi Massal
                predictions = model.predict(pads, verbose=0)
                progress_bar.progress(100)
                
                # Olah Hasil
                pred_indices = np.argmax(predictions, axis=1)
                confidences = np.max(predictions, axis=1)
                pred_labels = [id2label[i] for i in pred_indices]
                
                df_results = df_test.copy()
                df_results['pred_label'] = pred_labels
                df_results['confidence'] = confidences
                
                # Tampilkan Top 3 per Kategori
                st.write("### 🔥 Top 3 Teks Paling Yakin (Highest Confidence) per Kategori")
                
                unique_labels = sorted(list(set(pred_labels)))
                
                for label in unique_labels:
                    st.markdown(f"#### Kategori: **{label.upper()}**")
                    
                    # Filter & Sort
                    top_df = df_results[df_results['pred_label'] == label].sort_values(by='confidence', ascending=False).head(3)
                    
                    for _, row in top_df.iterrows():
                        # Warna-warni expander berdasarkan label agar cantik
                        with st.expander(f"🎯 {row['confidence']*100:.1f}% Confidence - {row['tweets'][:60]}..."):
                            st.write(f"**Teks Asli:** {row['tweets']}")
                            st.write(f"**Label Asli:** {row.get('class', 'N/A')}")
                            st.write(f"**Prediksi:** {row['pred_label']}")
                            st.progress(float(row['confidence']))
                            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")

if __name__ == '__main__':
    try:
        main_streamlit()
    except Exception as e:
        st.error(f"Runtime Error: {e}")