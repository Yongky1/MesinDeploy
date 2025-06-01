import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import json

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Wajah Selebriti",
    page_icon="🎭",
    layout="centered"
)

# Path model
MODEL_PATHS = {
    "Baseline CNN": "models/celebrity_model_baseline.keras",
    "VGG16": "models/celebrity_model_vgg16.keras",
    "InceptionV3": "models/celebrity_model_inceptionv3.keras",
    "MobileNetV2": "models/celebrity_model_mobilenetv2.keras"
}

# Daftar kelas selebriti
CELEBRITY_CLASSES = [
    "Angelina Jolie", "Brad Pitt", "Denzel Washington", "Hugh Jackman",
    "Jennifer Lawrence", "Johnny Depp", "Kate Winslet", "Leonardo DiCaprio",
    "Megan Fox", "Natalie Portman", "Nicole Kidman", "Robert Downey Jr",
    "Sandra Bullock", "Scarlett Johansson", "Tom Cruise", "Tom Hanks", "Will Smith"
]

@st.cache_resource
def load_model(model_path):
    try:
        # Coba muat model dengan keras
        try:
            # Muat model dengan keras
            model = keras.models.load_model(model_path, compile=False)
            
            # Verifikasi model
            if model is not None:
                # Cek input shape
                input_shape = model.input_shape
                if input_shape[1:] != (224, 224, 3):
                    st.warning(f"Model mengharapkan input shape {input_shape[1:]}, tapi akan diresize ke (224, 224, 3)")
                
                # Cek output shape
                output_shape = model.output_shape
                if output_shape[-1] != len(CELEBRITY_CLASSES):
                    st.error(f"Jumlah kelas model ({output_shape[-1]}) tidak sesuai dengan jumlah selebriti ({len(CELEBRITY_CLASSES)})")
                    return None
                
                # Kompilasi ulang model
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                return model
            
        except Exception as e:
            st.error(f"Error memuat model: {str(e)}")
            st.info("Mencoba memuat model dengan metode alternatif...")
            
            try:
                # Coba muat model dengan tf.keras
                model = tf.keras.models.load_model(model_path, compile=False)
                
                # Rekonstruksi model jika diperlukan
                if isinstance(model.layers[0], tf.keras.layers.InputLayer):
                    input_shape = model.layers[0].input_shape[1:]
                    model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=input_shape),
                        *model.layers[1:]
                    ])
                
                # Kompilasi ulang model
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                return model
                
            except Exception as e2:
                st.error(f"Semua metode pemuatan model gagal. Error terakhir: {str(e2)}")
                return None
                
    except Exception as e:
        st.error(f"Error tidak terduga: {str(e)}")
        return None

def preprocess_image(image):
    try:
        # Konversi ke RGB jika gambar dalam format lain
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize gambar ke 224x224
        image = image.resize((224, 224))
        
        # Konversi ke array numpy dan normalisasi
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Tambahkan batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error saat memproses gambar: {str(e)}")
        return None

def predict_celebrity(model, image):
    try:
        # Preprocess gambar
        img_array = preprocess_image(image)
        if img_array is None:
            return None, None, None
        
        # Prediksi
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return CELEBRITY_CLASSES[predicted_class], confidence, predictions[0]
    except Exception as e:
        st.error(f"Error saat melakukan prediksi: {str(e)}")
        return None, None, None

# Judul aplikasi
st.title("🎭 Klasifikasi Wajah Selebriti")
st.markdown("Upload foto wajah selebriti untuk mengidentifikasi siapa mereka!")

# Pemilihan model
selected_model = st.selectbox(
    "Pilih Model Klasifikasi:",
    list(MODEL_PATHS.keys()),
    help="Pilih model yang akan digunakan untuk klasifikasi"
)

# Upload gambar
uploaded_file = st.file_uploader("Upload foto wajah selebriti...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Tampilkan gambar yang diupload
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_column_width=True)
        
        # Tombol prediksi
        if st.button("Prediksi"):
            with st.spinner("Memuat model dan melakukan prediksi..."):
                # Load model
                model = load_model(MODEL_PATHS[selected_model])
                
                if model is not None:
                    # Lakukan prediksi
                    predicted_celebrity, confidence, all_predictions = predict_celebrity(model, image)
                    
                    if predicted_celebrity is not None:
                        # Tampilkan hasil
                        st.success(f"Hasil Prediksi: {predicted_celebrity}")
                        st.info(f"Tingkat Kepercayaan: {confidence:.2%}")
                        
                        # Tampilkan grafik probabilitas
                        fig, ax = plt.subplots(figsize=(10, 6))
                        y_pos = np.arange(len(CELEBRITY_CLASSES))
                        ax.barh(y_pos, all_predictions)
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(CELEBRITY_CLASSES)
                        ax.invert_yaxis()
                        ax.set_xlabel('Probabilitas')
                        ax.set_title('Probabilitas Prediksi untuk Setiap Selebriti')
                        st.pyplot(fig)
                    else:
                        st.error("Gagal melakukan prediksi. Silakan coba lagi.")
                else:
                    st.error("Gagal memuat model. Silakan coba model lain atau periksa file model.")
    except Exception as e:
        st.error(f"Error saat memproses file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("### Catatan")
st.markdown("""
- Pastikan gambar yang diupload memiliki wajah yang jelas
- Gambar akan otomatis diresize ke ukuran 224x224 pixel
- Model yang tersedia:
  - Baseline CNN: Model CNN sederhana
  - VGG16: Model berbasis arsitektur VGG16
  - InceptionV3: Model berbasis arsitektur InceptionV3
  - MobileNetV2: Model berbasis arsitektur MobileNetV2
""") 