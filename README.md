# Aplikasi Klasifikasi Wajah Selebriti

Aplikasi ini menggunakan model deep learning untuk mengklasifikasikan wajah selebriti dari gambar yang diunggah.

## Cara Penggunaan

1. Pastikan Anda telah menginstal semua dependensi:
   ```bash
   pip install -r requirements.txt
   ```

2. Pastikan model-model berikut sudah ada di folder `models/`:
   - `celebrity_model_baseline.keras`
   - `celebrity_model_vgg16.keras`
   - `celebrity_model_inceptionv3.keras`
   - `celebrity_model_mobilenetv2.keras`

3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

4. Buka browser dan akses aplikasi di `http://localhost:8501`

5. Unggah gambar wajah selebriti yang ingin diklasifikasikan

6. Pilih model yang ingin digunakan untuk klasifikasi

7. Lihat hasil prediksi yang ditampilkan

## Struktur Folder
```
.
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── celebrity_model_baseline.keras
    ├── celebrity_model_vgg16.keras
    ├── celebrity_model_inceptionv3.keras
    └── celebrity_model_mobilenetv2.keras
```

## Catatan
- Pastikan gambar yang diunggah memiliki wajah yang jelas
- Ukuran gambar akan otomatis diubah menjadi 224x224 pixel
- Model yang tersedia:
  - Baseline CNN
  - VGG16
  - InceptionV3
  - MobileNetV2 