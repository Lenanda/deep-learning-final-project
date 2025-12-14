import streamlit as st
import h5py
import os

st.title("🕵️ Diagnosa File .h5")

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
model_path = os.path.join(project_root, 'assets', 'best_bilstm_attention.h5')

st.write(f"Mengecek file di: `{model_path}`")

if os.path.exists(model_path):
    st.success("File ditemukan! Membaca struktur...")
    
    try:
        with h5py.File(model_path, 'r') as f:
            # Fungsi rekursif untuk print semua layer
            def print_attrs(name, obj):
                # Cari dataset yang menyimpan bobot (biasanya kernel atau weights)
                if isinstance(obj, h5py.Dataset):
                    st.text(f"{name}  -->  Shape: {obj.shape}")

            f.visititems(print_attrs)
            
        st.info("Silakan copy daftar layer di atas dan kirimkan kepada saya.")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
else:
    st.error("File tidak ditemukan. Pastikan nama file dan folder sudah benar.")
