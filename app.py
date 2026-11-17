import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Segmentasi Citra - Thresholding", layout="wide")
st.title("Segmentasi Citra Digital: Teknik Thresholding")
st.caption("Tugas Pengolahan Citra Digital")


def buat_citra_contoh():
    img = np.full((320, 480, 3), 236, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "LAPORAN CITRA DIGITAL", (30, 50), font, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    baris = [
        "Segmentasi memisahkan objek dari",
        "latar belakang menggunakan nilai ambang.",
        "Pencahayaan tidak merata membuat",
        "ambang global gagal pada area gelap.",
        "Metode adaptif menghitung ambang lokal",
        "sehingga teks tetap terbaca jelas.",
    ]
    for i, teks in enumerate(baris):
        cv2.putText(img, teks, (30, 90 + i * 26), font, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    gradasi = np.linspace(1.0, 0.4, 480, dtype=np.float32)[None, :, None]
    return (img * gradasi).astype(np.uint8)


def segment_document(gray, block_size, c):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c
    )
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


# ---------------- Sidebar ----------------
st.sidebar.header("Pengaturan")
file = st.sidebar.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
T = st.sidebar.slider("Global threshold (T)", 0, 255, 127)
block_size = st.sidebar.slider("blockSize (ganjil)", 3, 61, 11, step=2)
C = st.sidebar.slider("Konstanta C", -10, 20, 2)

# ---------------- Baca gambar ----------------
if file is not None:
    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
else:
    image = buat_citra_contoh()
    st.info("Belum ada gambar diunggah, memakai citra contoh. Unggah gambar lewat sidebar.")

if image is None:
    st.error("Gambar tidak bisa dibaca. Coba file lain.")
    st.stop()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------- Proses ----------------
_, thresh_global = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
otsu_value, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_mean = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C
)
thresh_gaussian = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
)
hasil_segmentasi = segment_document(gray, block_size, C)

# ---------------- Tampilan ----------------
st.subheader("Perbandingan Metode")
st.write(f"Nilai threshold optimal (Otsu): **{int(otsu_value)}**")

lebar = st.sidebar.slider("Lebar tiap gambar (px)", 150, 500, 220, step=10)
jumlah_kolom = st.sidebar.radio("Kolom per baris", [2, 3, 6], index=1, horizontal=True)

hasil = [
    ("Grayscale (asli)", gray),
    (f"Global (T={T})", thresh_global),
    ("Otsu", thresh_otsu),
    ("Adaptive Mean", thresh_mean),
    ("Adaptive Gaussian", thresh_gaussian),
    ("Segmentasi Dokumen Lengkap", hasil_segmentasi),
]

for baris in range(0, len(hasil), jumlah_kolom):
    kolom = st.columns(jumlah_kolom)
    for kol, (nama, gambar) in zip(kolom, hasil[baris:baris + jumlah_kolom]):
        kol.image(gambar, caption=nama, width=lebar)

st.subheader("Panduan Singkat")
st.markdown(
    """
- **Global / Otsu**: cocok untuk pencahayaan merata dan kontras tinggi.
- **Adaptive Mean / Gaussian**: cocok untuk dokumen dengan bayangan atau pencahayaan tidak rata.
- **blockSize** kecil: detail tinggi tetapi rawan noise. **blockSize** besar: halus tetapi detail hilang.
- Naikkan **C** untuk menghilangkan bintik pada latar.
"""
)