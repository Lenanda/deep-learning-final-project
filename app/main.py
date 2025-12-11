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
# 0. KONFIGURASI HALAMAN & DEBUGGING
# ==========================================
st.set_page_config(page_title="Deep Learning Emotion Detection", layout="centered")

# Fungsi cerdas untuk mencari folder assets secara otomatis
def find_project_root():
    # 1. Cek folder di mana main.py berada
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(current_dir, 'assets')):
        return current_dir
    
    # 2. Cek satu folder di atasnya (parent/root)
    parent_dir = os.path.dirname(current_dir)
    if os.path.exists(os.path.join(parent_dir, 'assets')):
        return parent_dir
    
    return None

# ==========================================
# 1. FUNGSI CLEANING
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)       
    text = re.sub(r'@\w+', ' ', text)                   
    text = re.sub(r'#', ' ', text)                      
    text = re.sub(r'\s+', ' ', text)                    
    text = text.strip()
    return text

# ==========================================
# 2. LOAD MODEL & ASET
# ==========================================
@st.cache_resource
def load_model_and_assets():
    # Cari root project
    project_root = find_project_root()
    
    if project_root is None:
        st.error("🚨 CRITICAL ERROR: Folder 'assets' tidak ditemukan!")
        st.code(f"Posisi file main.py saat ini: {os.path.dirname(os.path.abspath(__file__))}")
        st.warning("Pastikan struktur folder Anda: deep-learning-final-project/assets")
        return None, None, None, None

    # Set path lengkap
    tokenizer_path = os.path.join(project_root, 'assets', 'tokenizer.json')
    label_encoder_path = os.path.join(project_root, 'assets', 'label_encoder.pkl')
    model_path = os.path.join(project_root, 'assets', 'best_cnn_bilstm.h5')

    # Debugging Info (Akan hilang jika sukses)
    # st.write(f"Mencoba memuat aset dari: {project_root}")

    # A. LOAD TOKENIZER
    try:
        with open(tokenizer_path, 'r') as f:
            public_tokenizer_data = f.read()
            tokenizer = tokenizer_from_json(public_tokenizer_data)
    except FileNotFoundError:
        st.error(f"❌ Tidak ditemukan: {tokenizer_path}")
        return None, None, None, None

    # B. LOAD LABEL ENCODER
    try:
        with open(label_encoder_path, 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Tidak ditemukan: {label_encoder_path}")
        return None, None, None, None
    
    label2id = {label: idx for idx, label in enumerate(le.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}

    # C. DEFINISI ARSITEKTUR MODEL
    # Sesuaikan parameter ini dengan hasil diagnosa h5 Anda sebelumnya
    vocab_size = 30000     
    embedding_dim = 200    
    max_len = 40           
    num_classes = len(le.classes_) 

    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    x = SpatialDropout1D(0.4)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x) 
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x) 
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x) 
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # D. LOAD WEIGHTS
    try:
        model.load_weights(model_path)
    except Exception as e:
        st.error(f"❌ Gagal memuat bobot model: {e}")
        st.warning("Tips: Cek apakah versi TensorFlow di requirements.txt sudah 'tensorflow==2.15.0'")
        return None, None, None, None
    
    return model, tokenizer, id2label, max_len

# ==========================================
# 3. UI STREAMLIT
# ==========================================
def main_streamlit():
    st.title("🔮 Analisis Sentimen & Emosi")
    st.caption("Model: BiLSTM (32) + CNN (64) + Dense (64)")

    # Load Model
    model, tokenizer, id2label, max_len = load_model_and_assets()

    if model is None:
        st.error("Aplikasi berhenti karena gagal memuat aset. Periksa pesan error di atas.")
        st.stop()

    # --- FITUR 1: PREDIKSI MANUAL ---
    st.subheader("Coba Prediksi")
    user_input = st.text_area("Masukkan teks:", height=100, placeholder="Contoh: Saya sangat senang hari ini!")

    if st.button("Prediksi"):
        if user_input.strip() != "":
            cleaned = clean_text(user_input)
            seq = tokenizer.texts_to_sequences([cleaned])
            pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
            
            probs = model.predict(pad)[0]
            pred_id = probs.argmax()
            pred_label = id2label[pred_id]
            
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
            # Gunakan logika pencari root yang sama untuk folder 'data'
            project_root = find_project_root()
            if project_root:
                test_path = os.path.join(project_root, "data", "test.csv")
            else:
                test_path = "data/test.csv" # Fallback

            if not os.path.exists(test_path):
                st.error(f"File test.csv tidak ditemukan di: {test_path}")
            else:
                df_test = pd.read_csv(test_path)
                st.info(f"Memproses {len(df_test)} data dari test.csv...")
                
                progress_bar = st.progress(0)
                
                texts = df_test['tweets'].astype(str).tolist()
                cleaned_texts = [clean_text(t) for t in texts]
                
                seqs = tokenizer.texts_to_sequences(cleaned_texts)
                pads = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
                
                progress_bar.progress(50)
                
                predictions = model.predict(pads, verbose=0)
                progress_bar.progress(100)
                
                pred_indices = np.argmax(predictions, axis=1)
                confidences = np.max(predictions, axis=1)
                pred_labels = [id2label[i] for i in pred_indices]
                
                df_results = df_test.copy()
                df_results['pred_label'] = pred_labels
                df_results['confidence'] = confidences
                
                st.write("### 🔥 Top 3 Teks Paling Yakin (Highest Confidence) per Kategori")
                unique_labels = sorted(list(set(pred_labels)))
                
                for label in unique_labels:
                    st.markdown(f"#### Kategori: **{label.upper()}**")
                    top_df = df_results[df_results['pred_label'] == label].sort_values(by='confidence', ascending=False).head(3)
                    
                    for _, row in top_df.iterrows():
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
