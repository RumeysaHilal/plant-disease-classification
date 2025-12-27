import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- 1. AYARLAR ---
st.set_page_config(
    page_title="Bitki Doktoru",
    page_icon="🍃",
    layout="wide" # Yan yana sütunlar için geniş mod
)

MODEL_PATH = 'best_transfer_model_checkpoint.h5'

class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


IMG_SIZE = (128, 128)

# --- BAŞLIK ---
st.title("🌱 Bitki Hastalığı Tespit Sistemi")
st.write("---")

# --- 2. MODEL YÜKLEME ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

with st.spinner('Yapay zeka başlatılıyor...'):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        st.stop()

# --- 3. AKILLI TAHMİN FONKSİYONU ---
def predict_with_rotation(original_image, model):
    """
    Resim yataysa SADECE dikey (90, 270) açıları dener.
    Resim dikeyse her açıyı dener.
    """
    best_leaf_score = 0
    best_leaf_scores_array = None
    best_leaf_image = None
    best_bg_score = 0
    best_bg_scores_array = None
    
    # Resmin boyutlarını al
    width, height = original_image.size
    
    # --- YENİ MANTIK BURADA ---
    if width > height:
        # Resim YATAY (Landscape)
        st.info("↔️ Yatay resim algılandı. Model sadece dikey (90° ve 270°) çevirerek analiz yapacak.")
        angles = [90, 270] # Sadece dikleştiren açılar
    else:
        # Resim DİKEY (Portrait) veya Kare
        angles = [0, 90, 180, 270] # Her ihtimali dene

    for angle in angles:
        # Döndür
        rotated_img = original_image.rotate(angle)
        
        # Boyutlandır ve İşle
        img_resized = ImageOps.fit(rotated_img, IMG_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Tahmin Et
        prediction = model.predict(img_array)
        scores = prediction[0]
        
        current_max_score = np.max(scores)
        current_index = np.argmax(scores)
        current_class = class_names[current_index]
        
        # Background Filtreleme Mantığı
        if "background" in current_class.lower():
            if current_max_score > best_bg_score:
                best_bg_score = current_max_score
                best_bg_scores_array = scores
        else:
            if current_max_score > best_leaf_score:
                best_leaf_score = current_max_score
                best_leaf_image = rotated_img
                best_leaf_scores_array = scores

    # Sonuç Seçimi
    if best_leaf_score > 0.40:
        return best_leaf_scores_array, best_leaf_image, "LEAF"
    else:
        return best_bg_scores_array, original_image, "BG"

# --- 4. ARAYÜZ DÜZENİ (SOL: RESİM | SAĞ: SONUÇ) ---
col1, col2 = st.columns([1, 1.5])

# --- SOL SÜTUN ---
with col1:
    st.header("1. Resim Yükle 📸")
    file = st.file_uploader("Bir yaprak fotoğrafı seçin", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file)
        image = ImageOps.exif_transpose(image) # Telefondan gelen dönme bilgisini düzelt
        
        st.write("---")
        st.image(image, caption='Yüklenen Resim', use_column_width=True)
    else:
        st.info("Analiz için sol taraftan resim yükleyiniz.")

# --- SAĞ SÜTUN ---
with col2:
    st.header("2. Analiz Sonuçları 📊")
    
    if file is not None:
        st.write("Resim hazır. Analiz başlatılıyor...")
        st.write("") 
        
        if st.button('Hastalığı Analiz Et 🔬', type="primary", use_container_width=True):
            
            with st.spinner('Yapay zeka inceliyor...'):
                
                # Fonksiyonu çağır
                scores_array, best_image, result_type = predict_with_rotation(image, model)
                
                st.divider()

                if scores_array is None:
                    st.error("Hata: Tahmin oluşturulamadı.")
                else:
                    # Top-3 Hesaplama
                    top_3_indices = np.argsort(scores_array)[-3:][::-1]
                    top_class = class_names[top_3_indices[0]]
                    
                    # Sonuçları Göster
                    if result_type == "BG" or "background" in top_class.lower():
                        st.warning("⚠️ **Uyarı:** Görüntüde net bir bitki yaprağı algılanamadı.")
                    else:
                        st.success(f"✅ **En Güçlü Teşhis:** {top_class}")
                        
                        st.write("---")
                        st.subheader("🔍 Detaylı Olasılıklar")
                        
                        for i in top_3_indices:
                            class_name = class_names[i]
                            probability = scores_array[i] * 100
                            
                            c1, c2 = st.columns([2, 3])
                            with c1:
                                st.write(f"**{class_name}**")
                            with c2:
                                st.progress(int(probability), text=f"%{probability:.1f}")

                        # Kullanılan açıyı göster
                        if best_image:
                            st.write("---")
                            with st.expander("👀 Yapay Zeka Resmi Nasıl Gördü?"):
                                st.image(best_image, caption="Analiz için kullanılan açı", width=200)
