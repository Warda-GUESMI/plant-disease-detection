# 🌿 Détection Automatique des Maladies des Plantes par Deep Learning

> **Projet complet de Computer Vision & Intelligence Artificielle** permettant de diagnostiquer l'état sanitaire d'une feuille de plante (saine ou malade) à partir d'une simple photo, en utilisant l'apprentissage profond.

---

## 📖 Table des Matières

1. [🎯 Objectif du Projet](#-objectif-du-projet)
2. [🧠 Comprendre le Projet de A à Z](#-comprendre-le-projet-de-a-à-z)
3. [🏗️ Architecture Technique du Pipeline](#️-architecture-technique-du-pipeline)
4. [🛠️ Technologies Utilisées](#️-technologies-utilisées)
5. [📁 Structure du Projet](#-structure-du-projet)
6. [🚀 Guide d'Installation Complet](#-guide-dinstallation-complet)
7. [💻 Utilisation Pas à Pas](#-utilisation-pas-à-pas)
8. [🌐 Lancer l'API Web REST](#-lancer-lapi-web-rest)
9. [📊 Résultats & Métriques](#-résultats--métriques)
10. [🔧 Dépannage (FAQ)](#-dépannage-faq)

---

## 🎯 Objectif du Projet

Les maladies des plantes provoquent chaque année d'énormes pertes agricoles dans le monde. La détection précoce est essentielle mais nécessite l'expertise d'un agronome, ce qui n'est pas toujours accessible aux petits agriculteurs.

**Notre solution** : Un modèle d'Intelligence Artificielle capable de :
- 📷 Analyser une simple photo d'une feuille
- 🔍 Identifier la plante (Pomme, Tomate, Raisin, Maïs, etc.)
- 🩺 Détecter automatiquement la maladie (mildiou, tache noire, oïdium, etc.) ou confirmer que la plante est saine
- 📊 Donner un pourcentage de confiance sur son diagnostic

---

## 🧠 Comprendre le Projet de A à Z

### 1. Qu'est-ce que le Deep Learning ?

Le **Deep Learning** (apprentissage profond) est une branche de l'Intelligence Artificielle qui utilise des **réseaux de neurones artificiels** inspirés du cerveau humain. Ces réseaux apprennent à reconnaître des motifs (formes, couleurs, textures) directement à partir des données, sans qu'on ait besoin de leur expliquer manuellement ce qu'ils doivent regarder.

### 2. Qu'est-ce que la Computer Vision ?

La **Computer Vision** (vision par ordinateur) permet aux machines de "voir" et d'interpréter des images comme un humain. Nous utilisons ici des **CNN (Convolutional Neural Networks)** qui sont spécialisés dans l'analyse d'images.

### 3. Qu'est-ce que le Transfer Learning ?

Le **Transfer Learning** est une technique puissante où l'on prend un modèle déjà entraîné sur des millions d'images (**ImageNet** : 14 millions d'images de tout et n'importe quoi) et on le "recycle" pour notre tâche spécifique. C'est comme apprendre à reconnaître des maladies de plantes à quelqu'un qui sait déjà reconnaître visuellement le monde entier : c'est beaucoup plus rapide qu'apprendre à voir depuis zéro !

### 4. Pourquoi EfficientNet-B3 ?

EfficientNet-B3 est un modèle de pointe conçu par **Google** qui obtient d'excellents résultats de classification tout en restant relativement léger. Il est parfait pour notre projet car il :
- ✅ Atteint 95%+ de précision sur ImageNet
- ✅ N'est pas trop gourmand en mémoire (fonctionne même sur CPU)
- ✅ Contient 1536 features extractibles ➔ idéal pour du Transfer Learning

### 5. Les 7 étapes détaillées du projet

Notre projet suit un pipeline complet de Machine Learning en **7 étapes clés** :

| Étape | Nom | Description |
|-------|-----|-------------|
| **1** | **Data Cleaning** | On nettoie le dataset pour retirer les images corrompues et les doublons |
| **2** | **Split** | On divise les données en 3 groupes : Entraînement (70%), Validation (15%), Test (15%) |
| **3** | **Transforms** | On transforme et augmente artificiellement les images (rotations, contrastes, etc.) |
| **4** | **Model** | On construit l'architecture EfficientNet-B3 + une tête de classification personnalisée |
| **5** | **Training** | On entraîne le modèle en 2 phases (Head-only puis Fine-tuning complet) |
| **6** | **Evaluation** | On mesure les performances avec Accuracy, F1-score, Matrice de confusion |
| **7** | **Prediction / API** | On déploie le modèle en API REST FastAPI pour utilisation réelle |

---

## 🏗️ Architecture Technique du Pipeline

```text
┌──────────────────────────────────────────────────────────────┐
│              📁 DATASET PLANTVILLAGE (KAGGLE)               │
│              ~54 000 images / 38 classes                    │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│   🧹 STEP 1 : DATA CLEANING                                  │
│   • Validation OpenCV (fichiers illisibles/corrompus)        │
│   • Suppression des doublons par pHash (ImageHash)           │
│   • Filtrage des classes vides                               │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│   ✂️ STEP 2 : STRATIFIED SPLIT                                │
│   Train (70%) │ Val (15%) │ Test (15%)                       │
│   Stratification par classe (respect des proportions)        │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│   🔄 STEP 3 : TRANSFORMS & DATA AUGMENTATION                 │
│   Train : RandomResizedCrop, Flips, Rotations, ColorJitter  │
│   Val/Test : Resize + Normalize (moyennes ImageNet)          │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│   🧠 STEP 4 : MODÈLE EFFICIENTNET-B3                          │
│   Backbone : EfficientNet-B3 pré-entraîné sur ImageNet       │
│   Head : Dropout ➔ Linear(1536→512) ➔ BatchNorm ➔          │
│          GELU ➔ Dropout ➔ Linear(512→NUM_CLASSES)           │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│   🏋️ STEP 5 : FINE-TUNING PROGRESSIF                          │
│   Phase 1 (Head only) : entraînement de la tête seule        │
│   Phase 2 (Full) : dégel progressif du backbone              │
│   Optim : AdamW + Cosine Annealing LR + AMP (Mixed Prec)     │
│   Loss : Cross-Entropy + Label Smoothing                     │
│   Early Stopping (patience = 7)                              │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│   📊 STEP 6 : ÉVALUATION COMPLÈTE                             │
│   Accuracy, Macro F1, Weighted F1                            │
│   Precision, Recall (par classe)                             │
│   Top-3 & Top-5 Accuracy                                     │
│   Matrice de Confusion Normalisée                            │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│   🔮 STEP 7 : PRÉDICTION & DÉPLOIEMENT                        │
│   Script CLI : prédiction directe sur une image              │
│   API REST FastAPI : endpoint /predict via HTTP POST         │
│   Interface Swagger sur http://localhost:8000/docs           │
└──────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies Utilisées

| Catégorie | Bibliothèques |
|-----------|---------------|
| **Langage** | Python 3.10+ |
| **Deep Learning** | PyTorch, Torchvision |
| **Computer Vision** | OpenCV, Pillow, Albumentations |
| **Détection Doublons** | ImageHash (pHash) |
| **Machine Learning** | Scikit-Learn |
| **Analyse de Données** | NumPy, Pandas |
| **Visualisation** | Matplotlib, Seaborn |
| **API Web** | FastAPI, Uvicorn |
| **Barre de progression** | tqdm |

---

## 📁 Structure du Projet

```text
plant_disease_detection/
│
├── data/                          # 📁 Dataset brut (à télécharger)
│   └── PlantVillage/
│       ├── Apple___Apple_scab/
│       ├── Apple___healthy/
│       ├── Tomato___Late_blight/
│       └── ... (38 classes)
│
├── data_clean/                    # 🧹 Dataset nettoyé (auto-généré)
│   └── PlantVillage_clean/
│
├── splits/                        # ✂️ Séparation train/val/test (auto)
│   ├── train/
│   ├── val/
│   └── test/
│
├── src/                           # 🐍 Modules Python du pipeline
│   ├── __init__.py
│   ├── step1_cleaning.py          # Nettoyage des données
│   ├── step2_split.py             # Split stratifié
│   ├── step3_transforms.py        # Augmentations d'images
│   ├── step4_model.py             # Architecture EfficientNet-B3
│   ├── step5_training.py          # Boucle d'entraînement
│   ├── step6_evaluation.py        # Métriques et évaluation
│   └── step7_prediction.py        # Prédiction et inférence
│
├── models/                        # 💾 Modèles sauvegardés
│   ├── best_model.pth             # Meilleur checkpoint
│   └── class_names.json           # Noms des classes
│
├── results/                       # 📊 Graphiques & rapports
│   ├── cleaning_report.csv
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   └── training_curves.png
│
├── api.py                         # 🌐 Serveur API REST FastAPI
├── main.py                        # 🚀 Script principal CLI
├── run_all.py                     # ⚡ Automatisation complète
├── prepare_dataset.py             # 📦 Préparation du dataset
├── fast_setup.py                  # ⚡ Copie rapide des images
├── requirements.txt               # 📄 Liste des dépendances
└── README.md                      # 📖 Documentation
```

---

## 🚀 Guide d'Installation Complet

### ✅ Prérequis
- Python 3.10+ installé sur votre ordinateur
- 8 Go de RAM minimum (16 Go recommandés)
- Windows 10/11, Linux ou macOS
- Environ 2 Go d'espace disque pour le dataset

### 📥 Étape 1 : Télécharger le Dataset PlantVillage depuis Kaggle
1. Rendez-vous sur : [PlantVillage Dataset sur Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. Cliquez sur **"Download"** (nécessite un compte Kaggle gratuit).
3. Enregistrez le fichier `archive.zip` dans votre dossier **Téléchargements** (`Downloads`).

### 📦 Étape 2 : Extraire les fichiers du projet
1. Décompressez le projet dans votre dossier de travail.
2. Ouvrez ce dossier dans **VS Code**.

### 🐍 Étape 3 : Installer les dépendances Python
Ouvrez un terminal PowerShell dans VS Code (**Terminal ➔ New Terminal**) et lancez :

```powershell
python -m pip install -r requirements.txt
```
*Cette commande installe toutes les bibliothèques nécessaires (~500 Mo).*

### 📁 Étape 4 : Placer les données au bon endroit
Exécutez le script de préparation automatique :

```powershell
python prepare_dataset.py
```
Ce script va automatiquement :
- Chercher `archive.zip` dans vos Téléchargements
- L'extraire dans `data/PlantVillage/`
- Trier et organiser les 38 classes de plantes

---

## 💻 Utilisation Pas à Pas

### 🎯 Option A : Tout Lancer d'un Coup (Recommandé pour Débutants)

```powershell
python main.py --step all --epochs 5 --batch_size 16 --workers 0
```
Cela va exécuter automatiquement les étapes 1 à 6 du pipeline (Cleaning ➔ Split ➔ Train ➔ Evaluation).

### 🎯 Option B : Étape par Étape (Pour Comprendre)

#### 1️⃣ Nettoyer les données
```powershell
python main.py --step clean
```
Retire les images corrompues et les doublons perceptuels.

#### 2️⃣ Séparer en Train/Val/Test
```powershell
python main.py --step split
```
Crée les dossiers `splits/train/`, `splits/val/`, `splits/test/` de manière stratifiée.

#### 3️⃣ Entraîner le modèle
```powershell
python main.py --step train --epochs 15 --batch_size 16 --workers 0
```
Entraîne le réseau EfficientNet-B3 sur vos données.

#### 4️⃣ Évaluer les performances
```powershell
python main.py --step eval
```
Affiche l'Accuracy, le F1-score, le Top-K et sauvegarde la matrice de confusion dans `results/`.

#### 5️⃣ Prédire sur une nouvelle image
```powershell
python main.py --step predict --image "chemin/vers/une/image.jpg"
```
Exemple automatique (prend une image aléatoire du test set) :

```powershell
$img = (Get-ChildItem "splits/test" -File -Recurse | Select-Object -First 1).FullName
python main.py --step predict --image "$img"
```

---

## 🌐 Lancer l'API Web REST

Le projet embarque une API REST professionnelle avec **FastAPI**, permettant à d'autres applications (Web, Mobile) d'interroger votre modèle via des requêtes HTTP.

### 1. Démarrer le serveur API
```powershell
python api.py
```
Le terminal affichera :
```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 2. Interface Swagger interactive
Ouvrez votre navigateur web sur : [http://localhost:8000/docs](http://localhost:8000/docs)

Vous découvrirez une interface interactive vous permettant de :
- ✅ Tester l'état du serveur (`GET /health`)
- 📤 Uploader une image de feuille (`POST /predict`)
- 📥 Recevoir un diagnostic JSON instantané

### 3. Exemple de réponse JSON

```json
{
  "plant": "Tomato",
  "disease": "Late_blight",
  "confidence": 0.94,
  "is_healthy": false,
  "top_k": [
    {"class": "Tomato___Late_blight", "probability": 0.94},
    {"class": "Tomato___Early_blight", "probability": 0.03}
  ]
}
```

---

## 🎨 Lancer l'Interface Web Visuelle (Streamlit)

Si vous préférez une interface graphique moderne et dynamique pour glisser-déposer vos photos de feuilles directement dans votre navigateur :

### 1. Démarrer l'application Streamlit
```powershell
streamlit run app_web.py
```

### 2. Fonctionnalités de l'Interface
- 📤 **Glisser-Déposer / Importation d'image** : Accepte les formats JPG, JPEG et PNG.
- 🎨 **Affichage Dynamique** : Visualisation instantanée de la photo et du diagnostic.
- 🚨 **Badges Visuels & Indicateurs** : Alerte rouge pour maladie détectée ou badge vert pour plante saine.
- 📊 **Graphique de Probabilités** : Graphique Streamlit présentant le Top-5 des diagnostics probables.

---

## 📊 Résultats & Métriques

Voici les principales métriques calculées sur le jeu de données de test :

| Métrique | Description | Valeur Typique |
|----------|-------------|----------------|
| **Accuracy** | Précision globale | 95%+ (avec 15+ epochs) |
| **Macro F1** | Moyenne F1 par classe (équilibrée) | 92%+ |
| **Weighted F1** | F1 pondéré par le nombre d'images | 95%+ |
| **Precision** | Justesse des prédictions positives | 94%+ |
| **Recall** | Capacité à trouver toutes les maladies | 93%+ |
| **Top-3 Accuracy** | Bonne réponse parmi les 3 meilleures | 98%+ |
| **Top-5 Accuracy** | Bonne réponse parmi les 5 meilleures | 99%+ |

Tous les résultats et graphiques sont automatiquement sauvegardés dans le dossier `results/`.

---

## 🔧 Dépannage (FAQ)

### ❓ Q1 : Erreur `pip n'est pas reconnu`
**Solution** : Utilisez `python -m pip install ...` au lieu de `pip install ...`.

### ❓ Q2 : Erreur `UnicodeEncodeError` avec les émojis
**Solution** : PowerShell ne gère pas les emojis Unicode par défaut sur certaines configurations. Exécutez :
```powershell
chcp 65001
```
Ou utilisez `run_all.py` qui configure automatiquement la console UTF-8.

### ❓ Q3 : Le dataset n'est pas trouvé
**Solution** : Vérifiez que `archive.zip` est bien placé dans votre dossier `Downloads` et relancez :
```powershell
python prepare_dataset.py
```

### ❓ Q4 : L'entraînement est très lent sur CPU
**Solution** : Pour tester rapidement le code, réduisez les arguments :
- `--epochs 2` (au lieu de 15 ou 30)
- `--batch_size 8` (au lieu de 16 ou 32)
Si possible, utilisez un GPU (NVIDIA CUDA ou Google Colab).

### ❓ Q5 : `ModuleNotFoundError`
**Solution** : Réinstallez les dépendances du projet :
```powershell
python -m pip install -r requirements.txt --upgrade
```

### ❓ Q6 : Comment améliorer les performances ?
**Solutions** :
1. Augmenter le nombre d'époques : `--epochs 30`
2. Activer le GPU si disponible
3. Utiliser des augmentations de données plus poussées
4. Prolonger la Phase 2 de fine-tuning complet

---

## 👤 À Propos du Projet

- **Type** : Projet Deep Learning & Computer Vision de bout en bout
- **Framework** : PyTorch avec Transfer Learning (EfficientNet-B3)
- **Domaine** : Agriculture Intelligente (*Smart Farming*) / AgriTech
- **Cas d'usage** : Détection précoce des maladies pour agriculteurs, agronomes et applications mobiles agricoles

---

## 📝 Licence

Projet open-source à usage éducatif et professionnel.

🌿 **Merci d'avoir utilisé ce projet ! Bonne détection de maladies !** 🌿
