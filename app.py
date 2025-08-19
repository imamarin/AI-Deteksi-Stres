import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import deepface_patch
from deepface import DeepFace
import cv2
import tempfile

# -------------------------
# TRAINING MODEL DATA NUMERIK
# -------------------------
np.random.seed(42)
data = {
    "jam_tidur": np.random.normal(6, 1.5, 200),
    "detak_jantung": np.random.normal(80, 10, 200),
    "jam_kerja": np.random.normal(8, 2, 200),
    "tingkat_kecemasan": np.random.randint(1, 11, 200)
}
df = pd.DataFrame(data)
df["stres"] = ((df["jam_tidur"] < 6) & 
               (df["jam_kerja"] > 8) & 
               (df["tingkat_kecemasan"] > 6)).astype(int)

X = df[["jam_tidur", "detak_jantung", "jam_kerja", "tingkat_kecemasan"]]
y = df["stres"]

model_numerik = RandomForestClassifier()
model_numerik.fit(X, y)

# -------------------------
# STREAMLIT APP
# -------------------------
st.set_page_config(page_title="Kenan AI", page_icon="🧠")

# CSS untuk menyembunyikan logo GitHub (ikon kanan atas)
hide_github_icon = """
    <style>
       .st-emotion-cache-1p1m4ay{
            visibility: hidden;
        }
    </style>
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)

st.title("🧠 KENAN AI - Deteksi Stres")
st.write("Menggabungkan data kesehatan & aktivitas harian dengan analisis ekspresi wajah.")

# --- INPUT DATA NUMERIK ---
st.subheader("📋 Data Kesehatan & Aktivitas")
jam_tidur = st.slider("Jam tidur per hari", 0.0, 12.0, 7.0)
detak_jantung = st.slider("Detak jantung (BPM)", 40, 150, 80)
jam_kerja = st.slider("Jam kerja per hari", 0.0, 16.0, 8.0)
tingkat_kecemasan = st.slider("Tingkat kecemasan (1=tenang, 10=sangat cemas)", 1, 10, 5)

# --- UPLOAD ATAU AMBIL FOTO ---
st.subheader("📷 Foto Wajah")
camera_image = st.camera_input("Ambil foto dari kamera")
uploaded_file = st.file_uploader("Atau pilih foto dari file", type=["jpg", "jpeg", "png"])

img_path = None
if camera_image:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tfile.write(camera_image.getvalue())
    img_path = tfile.name
elif uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    img_path = tfile.name

if st.button("Prediksi Stres"):
    # Prediksi dari data numerik
    pred_num = model_numerik.predict_proba([[jam_tidur, detak_jantung, jam_kerja, tingkat_kecemasan]])[0][1]

    # Prediksi dari foto
    score_foto = 0.0
    if img_path:
        img = cv2.imread(img_path)
        try:
            result = DeepFace.analyze(img_path=img_path, actions=['emotion'])
            emotion = result[0]['dominant_emotion']

            # Mapping emosi ke skor stres sederhana
            stress_emotions = ["sad", "angry", "fear", "disgust", "neutral"]
            score_foto = 0.8 if emotion.lower() in stress_emotions else 0.2

            st.write(f"Ekspresi terdeteksi: **{emotion}**")
        except Exception as e:
            st.error(f"Gagal mendeteksi wajah: {e}")
        
        final_score = (pred_num * 0.5) + (score_foto * 0.5)
    else:
        st.warning("Tidak ada foto, prediksi hanya dari data numerik.")
        final_score = pred_num 

    # Gabungkan skor (60% numerik, 40% foto)
    

    # Output akhir
    if final_score >= 0.5:
        st.error(f"⚠️ Risiko stres tinggi! (Skor: {final_score:.2f})")
    else:
        st.success(f"✅ Risiko stres rendah. (Skor: {final_score:.2f})")





