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
# 0. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Analisis Emosi & Sentimen",
    layout="wide", # Layout lebar agar visualisasi lebih jelas
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. FUNGSI CLEANING (Sesuai main3.ipynb)
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)   # Menghapus hashtag sepenuhnya
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
        # Filter sesuai main3.ipynb
        df_train.dropna(subset=['tweets', 'class'], inplace=True)
        df_train = df_train[df_train['class'] != 'figurative']
        df_train['clean_text'] = df_train['tweets'].apply(clean_text)
    
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        df_test.dropna(subset=['tweets', 'class'], inplace=True) # Pastikan ada class untuk evaluasi
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
        st.error("❌ Tokenizer tidak ditemukan.")
        return None, None, None, None

    # Load Label Encoder
    try:
        with open(label_encoder_path, 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Label Encoder tidak ditemukan.")
        return None, None, None, None
    
    label2id = {label: idx for idx, label in enumerate(le.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}

    # Arsitektur Model (Attention BiLSTM)
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
            st.error(f"❌ Model file tidak ada di: {model_path}")
            return None, None, None, None
        model.load_weights(model_path)
    except Exception as e:
        st.error(f"❌ Gagal memuat bobot model: {e}")
        return None, None, None, None
    
    return model, tokenizer, id2label, max_len, le

# ==========================================
# 4. HALAMAN: EDA
# ==========================================
def show_eda_page(df_train):
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    if df_train is None:
        st.warning("Data Training tidak ditemukan. Pastikan 'data/train.csv' tersedia.")
        return

    st.write(f"**Total Data Training (Cleaned):** {len(df_train)} baris")
    st.write("Data telah difilter: Kelas 'figurative' dihapus & teks dibersihkan.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Kelas")
        fig, ax = plt.subplots()
        sns.countplot(data=df_train, x='class', palette='viridis', ax=ax)
        plt.title("Jumlah Data per Kelas")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Distribusi Panjang Teks")
        # Hitung panjang kata
        df_train['word_count'] = df_train['clean_text'].apply(lambda x: len(str(x).split()))
        fig, ax = plt.subplots()
        sns.histplot(df_train['word_count'], bins=30, kde=True, color='purple', ax=ax)
        plt.title("Distribusi Jumlah Kata per Tweet")
        st.pyplot(fig)

    st.subheader("Contoh Data (Raw vs Cleaned)")
    st.dataframe(df_train[['tweets', 'clean_text', 'class']].head(10))

# ==========================================
# 5. HALAMAN: HASIL TRAINING
# ==========================================
def show_evaluation_page(model, tokenizer, max_len, le, df_test):
    st.header("📈 Hasil Evaluasi Model")

    # 1. Training vs Validation Loss
    st.subheader("1. Training vs Validation Loss")
    st.info("⚠️ Grafik Loss historis hanya tersedia di Notebook. Di sini kita menampilkan evaluasi langsung terhadap Data Test.")
    
    if df_test is None:
        st.warning("Data Test tidak ditemukan untuk evaluasi.")
        return

    # Lakukan Prediksi Massal di Data Test untuk mendapatkan Matrix
    if st.button("Jalankan Evaluasi pada Data Test (Mungkin butuh waktu)"):
        with st.spinner("Sedang memproses prediksi..."):
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
# 6. HALAMAN: INPUT & PREDIKSI
# ==========================================
def show_prediction_page(model, tokenizer, id2label, max_len):
    st.header("🤖 Uji Coba Model (Prediksi)")
    st.caption("Arsitektur: BiLSTM + Attention Mechanism")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("Masukkan teks:", height=150, placeholder="Contoh: I love waiting 2 hours for my food. #sarcasm")
        
        if st.button("Prediksi", type="primary"):
            if user_input.strip() != "":
                cleaned = clean_text(user_input)
                seq = tokenizer.texts_to_sequences([cleaned])
                pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
                
                probs = model.predict(pad)[0]
                pred_id = probs.argmax()
                pred_label = id2label[pred_id]
                confidence = probs.max() * 100
                
                st.success(f"Prediksi: **{str(pred_label).upper()}**")
                st.metric("Tingkat Keyakinan (Confidence)", f"{confidence:.2f}%")
                
                # Expanders for details
                with st.expander("Lihat Hasil Preprocessing"):
                    st.code(cleaned)
            else:
                st.warning("Mohon masukkan teks terlebih dahulu.")

    with col2:
        if user_input.strip() != "" and 'probs' in locals():
            st.markdown("##### Probabilitas Kelas")
            prob_df = pd.DataFrame({
                "Kelas": [str(x).capitalize() for x in id2label.values()],
                "Probabilitas": probs
            })
            st.bar_chart(prob_df.set_index("Kelas"))

# ==========================================
# MAIN APP LOGIC
# ==========================================
def main():
    # Load Aset
    load_result = load_model_and_assets()
    if load_result is None:
        st.stop()
    model, tokenizer, id2label, max_len, le = load_result

    # Load Data
    df_train, df_test = load_data()

    # Sidebar Navigation
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman:", ["1. EDA (Eksplorasi Data)", "2. Hasil Evaluasi Model", "3. Prediksi & Demo"])

    st.sidebar.divider()
    st.sidebar.info("Project: Deep Learning Emotion Detection\nModel: BiLSTM Attention")

    # Page Routing
    if page == "1. EDA (Eksplorasi Data)":
        show_eda_page(df_train)
    elif page == "2. Hasil Evaluasi Model":
        show_evaluation_page(model, tokenizer, max_len, le, df_test)
    elif page == "3. Prediksi & Demo":
        show_prediction_page(model, tokenizer, id2label, max_len)

if __name__ == '__main__':
    main()
