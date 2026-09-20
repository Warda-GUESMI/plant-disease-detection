import os
import json
import torch
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from src.step7_prediction import PlantDiseasePredictor

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détection des Maladies des Plantes",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé pour une interface moderne et élégante
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 12px;
        padding: 20px;
        border-left: 6px solid #10B981;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card-danger {
        background-color: #FEF2F2;
        border-radius: 12px;
        padding: 20px;
        border-left: 6px solid #EF4444;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-healthy {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-disease {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Barre latérale (Sidebar) avec informations sur le projet
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
    st.title("🌿 Info Projet")
    st.markdown("""
    **Modèle** : EfficientNet-B3 (PyTorch)  
    **Dataset** : PlantVillage (38 classes)  
    **Spécialité** : Diagnostics phytosanitaires par IA  
    """)
    st.divider()
    st.subheader("🛠️ Guide Rapide")
    st.markdown("""
    1. Sélectionnez ou glissez-déposez une image de feuille.
    2. Le modèle pré-entraîné analyse l'image.
    3. Obtenez l'espèce, le diagnostic et le taux de confiance.
    """)
    st.divider()
    st.caption("Projet Deep Learning & Computer Vision")

# En-tête principal
st.markdown('<div class="main-header">🌿 Diagnostic Automatique des Maladies des Plantes</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyse d\'image par réseau de neurones artificiels (EfficientNet-B3)</div>', unsafe_allow_html=True)

# Fonction de chargement du prédicteur mise en cache pour des performances optimales
@st.cache_resource
def load_predictor():
    checkpoint_path = 'models/best_model.pth'
    class_names_path = 'models/class_names.json'
    
    if not os.path.exists(checkpoint_path) or not os.path.exists(class_names_path):
        return None
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(class_names_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    
    return PlantDiseasePredictor(checkpoint_path, class_names, device=device)

predictor = load_predictor()

if predictor is None:
    st.warning("⚠️ **Modèle introuvable** : Assurez-vous d'avoir exécuté l'entraînement (`python main.py --step train`) pour générer `models/best_model.pth` et `models/class_names.json`.")
else:
    # Zone de dépôt d'image
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📷 Image de la Feuille")
        uploaded_file = st.file_uploader(
            "Téléchargez une photo de feuille (JPG, PNG, JPEG)", 
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Image téléchargée", use_container_width=True)

    with col_right:
        st.subheader("📋 Diagnostic & Résultats")
        if uploaded_file is not None:
            # Sauvegarder temporairement pour l'analyse
            temp_path = "temp_upload.jpg"
            image.save(temp_path)

            with st.spinner("🧠 Analyse en cours par le réseau EfficientNet-B3..."):
                try:
                    res = predictor.predict(temp_path, top_k=5)
                    
                    plant = res['plant'].replace('_', ' ')
                    disease = res['disease'].replace('_', ' ')
                    confidence = res['confidence'] * 100
                    is_healthy = res['is_healthy']

                    st.markdown("---")
                    
                    if is_healthy:
                        st.markdown(f"""
                        <div class="metric-card">
                            <span class="badge-healthy">✅ PLANTE SAINE</span>
                            <h3 style="margin-top: 10px; color: #065F46;">Plante : {plant}</h3>
                            <p style="font-size: 1.1rem; color: #047857;">Aucune maladie détectée avec <b>{confidence:.2f}%</b> de confiance.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card-danger">
                            <span class="badge-disease">🚨 MALADIE DÉTECTÉE</span>
                            <h3 style="margin-top: 10px; color: #991B1B;">Plante : {plant}</h3>
                            <h4 style="color: #DC2626;">Maladie : {disease}</h4>
                            <p style="font-size: 1.1rem; color: #B91C1C;">Niveau de confiance : <b>{confidence:.2f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.write("📊 **Top-5 des probabilités :**")
                    
                    for item in res['top_k']:
                        class_name = item['class'].replace('___', ' ➔ ').replace('_', ' ')
                        prob = item['probability'] * 100
                        st.write(f"**{class_name}** ({prob:.1f}%)")
                        st.progress(float(item['probability']))

                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {str(e)}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            st.info("👆 Veuillez importer une image dans le panneau de gauche pour voir le diagnostic.")
