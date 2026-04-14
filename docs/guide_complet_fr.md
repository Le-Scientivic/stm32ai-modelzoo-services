# Guide Complet en Français — STM32 AI Model Zoo Services  
## Quantization pour Object Detection, Structure du Projet et Utilisation

---

## Table des matières

1. [Introduction et vue d'ensemble du projet](#1-introduction-et-vue-densemble-du-projet)
2. [Structure du dépôt](#2-structure-du-dépôt)
   - 2.1 [Architecture de haut niveau](#21-architecture-de-haut-niveau)
   - 2.2 [Dossier `object_detection` en détail](#22-dossier-object_detection-en-détail)
   - 2.3 [Dossier `common` — Services partagés](#23-dossier-common--services-partagés)
   - 2.4 [Organisation des artefacts d'expérience](#24-organisation-des-artefacts-dexpérience)
3. [Prérequis et installation de l'environnement](#3-prérequis-et-installation-de-lenvironnement)
   - 3.1 [Prérequis système](#31-prérequis-système)
   - 3.2 [Cloner le dépôt](#32-cloner-le-dépôt)
   - 3.3 [Création de l'environnement Python](#33-création-de-lenvironnement-python)
   - 3.4 [Installation des dépendances](#34-installation-des-dépendances)
   - 3.5 [Sous-modules Git (code applicatif)](#35-sous-modules-git-code-applicatif)
   - 3.6 [Outils STEdgeAI](#36-outils-stedgeai)
   - 3.7 [Option Docker](#37-option-docker)
4. [Workflow utilisateur — Vue d'ensemble](#4-workflow-utilisateur--vue-densemble)
   - 4.1 [Modes opérationnels disponibles](#41-modes-opérationnels-disponibles)
   - 4.2 [Fichier de configuration YAML](#42-fichier-de-configuration-yaml)
5. [Entraînement d'un modèle de détection d'objets](#5-entraînement-dun-modèle-de-détection-dobjets)
   - 5.1 [Préparer le jeu de données](#51-préparer-le-jeu-de-données)
   - 5.2 [Configuration YAML pour l'entraînement](#52-configuration-yaml-pour-lentraînement)
   - 5.3 [Lancer l'entraînement](#53-lancer-lentraînement)
6. [Quantization pour Object Detection](#6-quantization-pour-object-detection)
   - 6.1 [Concepts et objectifs de la quantization](#61-concepts-et-objectifs-de-la-quantization)
   - 6.2 [Workflow exact de quantization dans ce dépôt](#62-workflow-exact-de-quantization-dans-ce-dépôt)
   - 6.3 [Paramètres de configuration requis](#63-paramètres-de-configuration-requis)
   - 6.4 [Gestion des données de calibration](#64-gestion-des-données-de-calibration)
   - 6.5 [Commandes et flux d'exécution](#65-commandes-et-flux-dexécution)
   - 6.6 [Services enchaînés incluant la quantization](#66-services-enchaînés-incluant-la-quantization)
   - 6.7 [Quantization avancée (ONNX — outil d'inspection)](#67-quantization-avancée-onnx--outil-dinspection)
   - 6.8 [Pièges courants et conseils de dépannage](#68-pièges-courants-et-conseils-de-dépannage)
7. [Évaluation du modèle](#7-évaluation-du-modèle)
8. [Benchmarking](#8-benchmarking)
9. [Déploiement sur carte STM32](#9-déploiement-sur-carte-stm32)
10. [Adapter le projet à un dataset ou un modèle personnalisé](#10-adapter-le-projet-à-un-dataset-ou-un-modèle-personnalisé)
11. [Visualisation et suivi des expériences](#11-visualisation-et-suivi-des-expériences)
12. [Modèles supportés pour Object Detection](#12-modèles-supportés-pour-object-detection)
13. [Annexes](#13-annexes)

---

## 1. Introduction et vue d'ensemble du projet

**STM32 AI Model Zoo Services** est un ensemble de scripts et de services Python destinés à faciliter l'intégration de modèles d'intelligence artificielle sur les microcontrôleurs et microprocesseurs STM32 de STMicroelectronics. Il s'utilise en complément du dépôt [STM32 Model Zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/), qui fournit une collection de modèles de référence pré-entraînés et optimisés pour les cibles STM32.

Le projet offre :

- Des **scripts de ré-entraînement ou de fine-tuning** à partir de vos propres données (BYOD — Bring Your Own Data, BYOM — Bring Your Own Model).
- Un ensemble de **services individuels** : entraînement, quantization, évaluation, benchmark, prédiction, déploiement.
- Des **services enchaînés** (chained services) pour exécuter plusieurs étapes successives en un seul lancement (ex. : entraîner → quantizer → évaluer → benchmarker).
- Des **exemples de code applicatif** générés automatiquement à partir des modèles IA de l'utilisateur.

**Cas d'usage disponibles :**  
Image Classification, Object Detection, Pose Estimation, Face Detection, Semantic Segmentation, Instance Segmentation, Depth Estimation, Neural Style Transfer, Re-Identification, Audio Event Detection, Speech Enhancement, Human Activity Recognition, Hand Posture Recognition, Arc Fault Detection.

**Frameworks supportés :** TensorFlow 2.18.0 / Keras 3.8.0, PyTorch 2.7.1, ONNX 1.16.1.

---

## 2. Structure du dépôt

### 2.1 Architecture de haut niveau

```
stm32ai-modelzoo-services/
├── README.md                    ← Guide d'entrée principale
├── requirements.txt             ← Dépendances Python globales
├── __init__.py
│
├── common/                      ← Services partagés entre tous les cas d'usage
│   ├── benchmarking/
│   ├── compression/
│   ├── data_augmentation/
│   ├── deployment/
│   ├── evaluation/
│   ├── model_utils/
│   ├── onnx_utils/
│   ├── optimization/
│   ├── prediction/
│   ├── quantization/            ← Moteur de quantization (TFLite + ONNX)
│   ├── registries/
│   ├── stm32ai_dc/              ← Interface STEdgeAI Developer Cloud
│   ├── stm32ai_local/           ← Interface STEdgeAI Core local
│   ├── stm_ai_runner/
│   ├── training/
│   └── utils/
│
├── object_detection/            ← Cas d'usage détection d'objets (TensorFlow)
│   ├── stm32ai_main.py          ← Point d'entrée principal
│   ├── user_config.yaml         ← Fichier de configuration utilisateur
│   ├── user_config_pt.yaml      ← Config utilisateur PyTorch
│   ├── config_file_examples/    ← Exemples de configs prêts à l'emploi
│   ├── config_file_examples_pt/ ← Exemples configs PyTorch
│   ├── datasets/                ← (Dossier à remplir avec vos datasets)
│   ├── docs/                    ← Documentation détaillée
│   ├── tf/                      ← Code source TensorFlow
│   │   └── src/
│   │       ├── experiments_outputs/  ← Sorties des expériences (créé à l'exécution)
│   │       └── ...
│   └── pt/                      ← Code source PyTorch
│
├── image_classification/        ← Cas d'usage classification d'images
├── pose_estimation/
├── face_detection/
├── semantic_segmentation/
├── instance_segmentation/
├── depth_estimation/
├── audio_event_detection/
├── speech_enhancement/
├── human_activity_recognition/
├── hand_posture/
├── arc_fault_detection/
├── neural_style_transfer/
├── re_identification/
│
├── application_code/            ← Code embarqué (submodules Git)
├── docker/                      ← Configuration Docker
├── api/                         ← API REST (optionnel)
└── tutorials/                   ← Tutoriels généraux
```

### 2.2 Dossier `object_detection` en détail

```
object_detection/
├── stm32ai_main.py              ← Point d'entrée : lance tous les services
├── user_config.yaml             ← Votre fichier de configuration principal
├── user_config_pt.yaml          ← Config PyTorch
│
├── config_file_examples/        ← Exemples de configurations pour chaque service
│   ├── training_config.yaml
│   ├── quantization_config.yaml
│   ├── evaluation_config.yaml
│   ├── benchmarking_config.yaml
│   ├── deployment_config.yaml
│   ├── chain_tqeb_config.yaml   ← Train → Quantize → Evaluate → Benchmark
│   ├── chain_tqe_config.yaml    ← Train → Quantize → Evaluate
│   ├── chain_eqe_config.yaml    ← Evaluate float → Quantize → Evaluate quantizé
│   ├── chain_qb_config.yaml     ← Quantize → Benchmark
│   ├── chain_eqeb_config.yaml   ← Evaluate → Quantize → Evaluate → Benchmark
│   ├── chain_qd_config.yaml     ← Quantize → Deploy
│   └── ...
│
├── datasets/                    ← Vos datasets locaux (à créer/remplir)
│
├── docs/                        ← Documentation Markdown
│   ├── README_OVERVIEW.md       ← Vue d'ensemble et tutoriel complet
│   ├── README_TRAINING.md       ← Guide entraînement
│   ├── README_QUANTIZATION.md   ← Guide quantization (TFLite PTQ)
│   ├── README_QUANTIZATION_TOOL.md ← Outil quantization avancé (ONNX)
│   ├── README_EVALUATION.md     ← Guide évaluation
│   ├── README_BENCHMARKING.md   ← Guide benchmarking
│   ├── README_DEPLOYMENT_STM32N6.md
│   ├── README_DEPLOYMENT_STM32H7.md
│   ├── README_DEPLOYMENT_MPU.md
│   ├── README_MODELS.md         ← Description des modèles TF
│   ├── README_MODELS_TORCH.md   ← Description des modèles PyTorch
│   ├── README_DATASETS.md
│   ├── README_LR_SCHEDULE.md
│   ├── README_PREDICTION.md
│   ├── README_OVERVIEW_TORCH.md
│   └── tuto/                    ← Tutoriels spécifiques (YOLO, custom dataset…)
│
├── tf/                          ← Code Python TensorFlow
│   └── src/
│       ├── experiments_outputs/ ← Résultats des runs (créé automatiquement)
│       │   ├── YYYY_MM_DD_HH_MM_SS/   ← Un dossier par run
│       │   │   ├── logs/
│       │   │   ├── saved_models/
│       │   │   ├── quantized_models/
│       │   │   └── stm32ai_main.log
│       │   └── mlruns/          ← Logs MLflow
│       └── ...
│
└── pt/                          ← Code Python PyTorch
```

### 2.3 Dossier `common` — Services partagés

Le dossier `common/` contient tous les modules réutilisables par les différents cas d'usage :

| Sous-dossier | Rôle |
|---|---|
| `quantization/` | Moteurs de quantization TFLite et ONNX (PTQ, mixed precision, inspection) |
| `training/` | Boucles d'entraînement, callbacks, planificateurs LR |
| `evaluation/` | Calcul de métriques (mAP, accuracy, etc.) |
| `benchmarking/` | Interface STEdgeAI pour mesurer empreinte mémoire et temps d'inférence |
| `deployment/` | Génération du code C et déploiement sur carte STM32 |
| `data_augmentation/` | Fonctions d'augmentation d'images |
| `model_utils/` | Utilitaires de chargement/sauvegarde de modèles |
| `onnx_utils/` | Utilitaires ONNX |
| `stm32ai_dc/` | Connexion au STEdgeAI Developer Cloud |
| `stm32ai_local/` | Appel à STEdgeAI Core installé localement |
| `prediction/` | Service de prédiction/inférence |
| `optimization/` | Compression et optimisation de modèles |
| `utils/` | Fonctions utilitaires générales |

### 2.4 Organisation des artefacts d'expérience

Chaque lancement du script crée un dossier horodaté dans `experiments_outputs/` (configurable via Hydra) :

```
experiments_outputs/
└── 2024_03_15_10_30_45/
    ├── logs/                    ← Logs TensorBoard + ClearML
    ├── saved_models/            ← Modèles Keras/ONNX sauvegardés
    │   ├── best_model.keras
    │   └── last_model.keras
    ├── quantized_models/        ← Modèles quantizés (.tflite / .onnx)
    │   └── best_model_quantized.tflite
    └── stm32ai_main.log         ← Log complet du run
```

---

## 3. Prérequis et installation de l'environnement

### 3.1 Prérequis système

| Composant | Version requise |
|---|---|
| Python | **3.12.9** |
| TensorFlow | 2.18.0 |
| PyTorch | 2.7.1 |
| ONNX | 1.16.1 |
| Keras | 3.8.0 |
| GPU (optionnel) | NVIDIA avec CUDA/cuDNN |
| STEdgeAI Core (optionnel) | Dernière version |
| STM32CubeIDE (déploiement) | Dernière version |

> **⚠️ Important (Windows)** : Évitez les espaces dans les chemins de fichiers (Python, STM32CubeIDE, STEdgeAI). Activez le support des longs chemins dans le registre Windows si nécessaire.

### 3.2 Cloner le dépôt

```bash
git clone https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git --depth 1
cd stm32ai-modelzoo-services
```

Si vous avez besoin des modèles pré-entraînés du Model Zoo (pour la quantization, l'évaluation ou le déploiement), clonez également :

```bash
# Dans le même dossier parent
git clone https://github.com/STMicroelectronics/stm32ai-modelzoo.git --depth 1
```

### 3.3 Création de l'environnement Python

**Avec venv (recommandé sous Linux/Mac) :**
```bash
python -m venv st_zoo
source st_zoo/bin/activate          # Linux/Mac
# st_zoo\Scripts\activate.bat       # Windows
```

**Avec conda :**
```bash
conda create -n st_zoo python=3.12.9
conda activate st_zoo
```

**GPU NVIDIA avec conda (optionnel) :**
```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' \
  > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

### 3.4 Installation des dépendances

```bash
pip install -r requirements.txt
```

Les principales bibliothèques installées sont :

```
tensorflow==2.18.0          # Framework ML principal (TFLite converter)
keras==3.8.0                # API de haut niveau
torch==2.7.1                # PyTorch (support PyTorch)
onnx==1.16.1                # Format ONNX
onnxruntime==1.20.1         # Inférence ONNX
mlflow==2.20.0              # Suivi des expériences
hydra-core==1.3.2           # Gestion des configurations
pycocotools==2.0.7          # Métriques COCO (mAP)
neural-compressor==3.4.1    # Quantization avancée Intel
clearml==1.16.5             # Suivi alternatif des expériences
```

### 3.5 Sous-modules Git (code applicatif)

Le code embarqué (C/C++ pour STM32) est fourni sous forme de sous-modules Git. Initialisez-les uniquement si vous utilisez le déploiement :

```bash
git submodule update --init --recursive
# Ou pour un cas d'usage spécifique :
git submodule update --init application_code/object_detection/STM32N6
```

### 3.6 Outils STEdgeAI

Pour le benchmarking et le déploiement, vous avez le choix entre :

1. **STEdgeAI Developer Cloud** (en ligne) : créez un compte sur [stedgeai-dc.st.com](https://stedgeai-dc.st.com/home) et configurez `tools.stedgeai.on_cloud: True` dans le YAML.

2. **STEdgeAI Core** (local) : téléchargez depuis [st.com/stedgeai-core](https://www.st.com/en/development-tools/stedgeai-core.html) et indiquez le chemin dans `tools.stedgeai.path_to_stedgeai`.

### 3.7 Option Docker

Un environnement Docker complet est disponible avec toute la stack logicielle pré-installée :

```bash
# Consulter la documentation Docker
cat docker/README.md

# Construire et lancer l'image
cd docker
docker build -t st-modelzoo .
./launcher.sh
```

---

## 4. Workflow utilisateur — Vue d'ensemble

### 4.1 Modes opérationnels disponibles

Tout le projet est piloté par le paramètre `operation_mode` dans le fichier YAML. Il existe des **services individuels** et des **services enchaînés** :

| `operation_mode` | Opérations effectuées |
|---|---|
| `training` | Entraîne un modèle depuis zéro ou par fine-tuning |
| `evaluation` | Évalue la précision d'un modèle float ou quantizé |
| `quantization` | Quantize un modèle float en entiers (TFLite ou ONNX) |
| `prediction` | Applique un modèle sur des images et affiche les détections |
| `benchmarking` | Mesure les empreintes mémoire et temps d'inférence sur STM32 |
| `deployment` | Génère le code C et flashe la carte STM32 |
| `chain_tqe` | Train → Quantize → Evaluate |
| `chain_tqeb` | Train → Quantize → Evaluate → Benchmark |
| `chain_eqe` | Evaluate (float) → Quantize → Evaluate (quantizé) |
| `chain_qb` | Quantize → Benchmark |
| `chain_eqeb` | Evaluate → Quantize → Evaluate → Benchmark |
| `chain_qd` | Quantize → Deploy |

Dans les noms de chaînes : **t** = training, **q** = quantization, **e** = evaluation, **b** = benchmarking, **d** = deployment.

### 4.2 Fichier de configuration YAML

**Toute exécution est pilotée par un unique fichier YAML.** Il définit le mode opérationnel, le modèle, le dataset, les paramètres de prétraitement, d'entraînement, de quantization, de déploiement, ainsi que la configuration Hydra/MLflow.

**Structure générale d'un fichier YAML :**

```yaml
# ─── Mode opérationnel ───────────────────────────────────────
operation_mode: quantization   # ou training, evaluation, chain_tqeb, etc.

# ─── Paramètres globaux du projet ────────────────────────────
general:
  project_name: mon_projet_od
  logs_dir: logs
  saved_models_dir: saved_models
  global_seed: 123
  deterministic_ops: False
  display_figures: True
  gpu_memory_limit: 16          # en Go
  num_threads_tflite: 4

# ─── Modèle ──────────────────────────────────────────────────
model:
  model_type: st_yoloxn
  model_name: st_yoloxn_d033_w025
  model_path:                   # Chemin vers un modèle existant (optionnel)
  input_shape: (256, 256, 3)

# ─── Dataset ─────────────────────────────────────────────────
dataset:
  format: tfs
  dataset_name: coco
  class_names: [person]
  training_path: ./datasets/od_coco_2017_person/train
  validation_split: 0.1
  test_path:
  quantization_path: ./datasets/od_coco_2017_person/quantization
  quantization_split: 0.05      # utiliser 5% du dataset de quantization

# ─── Prétraitement ───────────────────────────────────────────
preprocessing:
  rescaling:
    scale: 1/127.5
    offset: -1
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb

# ─── Post-traitement ─────────────────────────────────────────
postprocessing:
  confidence_thresh: 0.001
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  plot_metrics: False
  max_detection_boxes: 100

# ─── Quantization ────────────────────────────────────────────
quantization:
  quantizer: TFlite_converter
  quantization_type: PTQ
  quantization_input_type: uint8
  quantization_output_type: float
  granularity: per_channel
  optimize: False
  export_dir: quantized_models

# ─── Hydra (gestion des outputs) ─────────────────────────────
hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}

# ─── MLflow (suivi des métriques) ────────────────────────────
mlflow:
  uri: ./tf/src/experiments_outputs/mlruns
```

**Lancement de base :**

```bash
# Depuis le dossier object_detection/
python stm32ai_main.py
# ou avec un fichier de config spécifique :
python stm32ai_main.py --config-path ./config_file_examples/ --config-name quantization_config.yaml
# ou en surchargeant un paramètre :
python stm32ai_main.py operation_mode=quantization
```

---

## 5. Entraînement d'un modèle de détection d'objets

### 5.1 Préparer le jeu de données

Le projet supporte plusieurs formats de datasets. Voici les correspondances :

| `dataset_name` | `format` acceptés | Description |
|---|---|---|
| `coco` | `coco`, `tfs` | Format COCO natif (JSON) ou TFS TensorFlow |
| `pascal_voc` | `pascal_voc`, `tfs` | Format Pascal VOC XML ou TFS TensorFlow |
| `darknet_yolo` | `darknet_yolo`, `tfs` | Format YOLO Darknet txt ou TFS TensorFlow |
| `custom_dataset` | `tfs` | Uniquement TFS TensorFlow (déjà converti) |

**Format TFS TensorFlow (format interne du projet) :**  
La conversion vers le format TFS est effectuée automatiquement par le projet lors du premier run si `format` n'est pas `tfs`.

**Structure attendue pour `darknet_yolo` :**

```
dataset_directory/
├── train/
│   ├── image1.jpg
│   ├── image1.txt    ← annotations YOLO (class x_center y_center w h)
│   ├── image2.jpg
│   └── image2.txt
└── val/
    ├── val_image1.jpg
    └── val_image1.txt
```

**Téléchargement automatique** (COCO et Pascal VOC uniquement) :

```yaml
dataset:
  dataset_name: coco
  download_data: True   # Télécharge automatiquement si absent
```

### 5.2 Configuration YAML pour l'entraînement

Voici un exemple complet pour entraîner `st_yoloxn` sur COCO (détection de personnes) :

```yaml
operation_mode: training

general:
  project_name: coco_person_yoloxn
  logs_dir: logs
  saved_models_dir: saved_models
  gpu_memory_limit: 16
  num_threads_tflite: 4
  global_seed: 127

model:
  model_type: st_yoloxn
  model_name: st_yoloxn_d033_w025
  input_shape: (256, 256, 3)

dataset:
  format: tfs
  dataset_name: coco
  class_names: [person]
  training_path: ./datasets/od_coco_2017_person/train
  validation_split: 0.1
  test_path: ./datasets/od_coco_2017_person/test
  quantization_split: 0.05

preprocessing:
  rescaling:
    scale: 1/255
    offset: 0
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb

data_augmentation:
  random_contrast:
    factor: 0.4
  random_brightness:
    factor: 0.3
  random_flip:
    mode: horizontal
  random_translation:
    width_factor: 0.15
    height_factor: 0.15
    fill_mode: reflect
    interpolation: nearest
  random_rotation:
    factor: 0.02
    fill_mode: reflect
    interpolation: nearest

postprocessing:
  confidence_thresh: 0.001
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  plot_metrics: False
  max_detection_boxes: 100

training:
  dropout: null
  batch_size: 64
  epochs: 400
  optimizer:
    Adam:
      learning_rate: 0.0025
  callbacks:
    LRWarmupCosineDecay:
      initial_lr: 1.0e-05
      warmup_steps: 20
      max_lr: 0.00125
      hold_steps: 20
      decay_steps: 300
      end_lr: 1.0e-06
    EarlyStopping:
      monitor: val_loss
      patience: 60
      restore_best_weights: true
      verbose: 1

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

### 5.3 Lancer l'entraînement

```bash
# Depuis le dossier object_detection/
python stm32ai_main.py --config-path ./config_file_examples/ --config-name training_config.yaml

# Ou avec le fichier user_config.yaml modifié :
python stm32ai_main.py

# En surchargeant le mode depuis la ligne de commande :
python stm32ai_main.py operation_mode=training
```

Les modèles sauvegardés se trouvent dans `experiments_outputs/<date>/saved_models/`.

---

## 6. Quantization pour Object Detection

### 6.1 Concepts et objectifs de la quantization

La **quantization post-entraînement (PTQ — Post-Training Quantization)** est une technique d'optimisation qui convertit les poids et les activations d'un réseau de neurones de la virgule flottante (float32, ≈32 bits) vers des entiers (int8 ou uint8, ≈8 bits), **après** l'entraînement, **sans ré-entraîner** le modèle.

**Pourquoi quantizer pour l'embarqué STM32 ?**

| Avantage | Impact |
|---|---|
| **Réduction de la taille mémoire (Flash)** | ×4 environ (float32 → int8) |
| **Réduction de la RAM** | Les activations int8 sont 4× plus petites |
| **Accélération de l'inférence** | Les opérations entières sont plus rapides sur MCU |
| **Moindre consommation énergétique** | Calculs entiers moins coûteux |
| **Compatibilité STEdgeAI** | Les outils ST nécessitent des modèles quantizés (TFLite int8 ou ONNX QDQ) |

**Compromis à considérer :**  
La quantization entraîne généralement une légère perte de précision (mAP). L'objectif est de la minimiser via une bonne **calibration** sur des données représentatives. En pratique, la perte de mAP est souvent inférieure à 1–2 points.

**Vocabulaire clé :**
- **PTQ** : Post-Training Quantization — conversion après entraînement.
- **Calibration** : passage d'images réelles à travers le modèle pour mesurer les plages de valeurs des activations et calculer les paramètres de quantization (scale, zero-point).
- **Granularité** : par tenseur (`per_tensor`) ou par canal (`per_channel`). `per_channel` est plus précis mais légèrement plus lourd.
- **Types d'entrée/sortie** : `uint8` (0–255), `int8` (−128 à 127), `float` (pas de quantization sur l'I/O).
- **TFLite Converter** : outil de Google (inclus dans TensorFlow) utilisé pour la quantization PTQ en TFLite.
- **ONNX Quantizer** : outil alternatif supportant les précisions 4 bits, 8 bits, 16 bits et la quantization mixte.

### 6.2 Workflow exact de quantization dans ce dépôt

Le workflow standard de quantization est le suivant :

```
┌─────────────────────────────────────────────┐
│  1. Modèle float (.keras ou .onnx)          │
│     → spécifié via general.model_path       │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  2. Dataset de calibration                  │
│     → quantization_path dans [dataset]      │
│     (si absent : données aléatoires)        │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  3. Prétraitement des images de calibration │
│     → rescaling + resizing (même params     │
│       qu'à l'entraînement)                  │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  4. TFLite Converter PTQ                    │
│     → Calibration des activations           │
│     → Quantization des poids (int8/uint8)   │
│     → Granularité per_channel ou per_tensor │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  5. Export du modèle quantizé (.tflite)     │
│     → sauvegardé dans export_dir            │
│       (ex: quantized_models/)               │
└─────────────────────────────────────────────┘
```

### 6.3 Paramètres de configuration requis

**Section `general` :**

```yaml
general:
  model_path: chemin/vers/mon_modele.keras   # Modèle float à quantizer
```

**Section `quantization` :**

```yaml
quantization:
  quantizer: TFlite_converter     # Seule option pour TFLite PTQ
                                  # Alternative : "Onnx_quantizer"
  quantization_type: PTQ          # Seule option disponible : Post-Training Quantization
  quantization_input_type: uint8  # Type de l'entrée du modèle quantizé
                                  # Options : "uint8", "int8", "float"
  quantization_output_type: float # Type de la sortie du modèle quantizé
                                  # Options : "uint8", "int8", "float"
  granularity: per_channel        # Granularité de quantization
                                  # Options : "per_channel" (défaut), "per_tensor"
  optimize: False                 # Optimiser le modèle avant quantization
                                  # Recommandé True uniquement avec per_tensor
  export_dir: quantized_models    # Nom du dossier de sortie (relatif à experiments_outputs/)
```

**Paramètres détaillés :**

| Paramètre | Description | Valeurs possibles |
|---|---|---|
| `quantizer` | Outil de quantization | `TFlite_converter`, `Onnx_quantizer` |
| `quantization_type` | Type de quantization | `PTQ` (seul supporté) |
| `quantization_input_type` | Type de l'entrée du modèle quantizé | `uint8`, `int8`, `float` |
| `quantization_output_type` | Type de la sortie du modèle quantizé | `uint8`, `int8`, `float` |
| `granularity` | Granularité de la quantization des poids | `per_channel` (défaut), `per_tensor` |
| `optimize` | Pré-optimisation avant quantization | `True`, `False` |
| `export_dir` | Répertoire d'export du modèle quantizé | Nom de dossier (string) |

**Section `dataset` (pour la calibration) :**

```yaml
dataset:
  format: tfs
  dataset_name: coco
  class_names: [person]
  quantization_path: ./datasets/COCO_2017_person   # Dataset de calibration
  quantization_split: 0.4   # Utiliser 40% seulement (optionnel, pour accélérer)
  seed: 0                   # Graine aléatoire pour le split
```

**Section `preprocessing` (obligatoire, doit correspondre à l'entraînement) :**

```yaml
preprocessing:
  rescaling:
    scale: 1/127.5   # Doit être identique à l'entraînement !
    offset: -1
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb
```

**Section `postprocessing` (pour évaluation éventuelle) :**

```yaml
postprocessing:
  confidence_thresh: 0.001
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  plot_metrics: False
  max_detection_boxes: 100
```

### 6.4 Gestion des données de calibration

La calibration est l'étape clé qui détermine la qualité du modèle quantizé. Voici les règles suivies par le projet :

**Règle de sélection du dataset de calibration (machine à états) :**

```
Est-ce que quantization_path est défini ?
    ├── OUI → Utiliser quantization_path
    │         (avec quantization_split si défini)
    └── NON → Est-ce que training_path est défini ?
                  ├── OUI → Utiliser training_path
                  │         (avec quantization_split si défini)
                  └── NON → Quantization aléatoire
                             (données synthétiques, sans évaluation de précision)
```

**Recommandations :**

1. **Toujours fournir des données réelles** pour la calibration. Une quantization sur données aléatoires peut donner de mauvais résultats de précision mais peut être utile pour estimer rapidement l'empreinte mémoire.

2. **Utiliser `quantization_split`** si le dataset de calibration est trop grand (entraîne des temps de calibration longs). Un sous-ensemble de 200–1000 images est généralement suffisant.

3. **Les images de calibration doivent être représentatives** du domaine d'application réel. Plus elles couvrent la diversité des données de production, meilleure sera la calibration.

4. **Le prétraitement doit être identique** à celui utilisé lors de l'entraînement (`scale`, `offset`, `resizing`). Toute différence entraîne une dégradation de précision.

**Exemple avec split de quantization :**

```yaml
dataset:
  format: tfs
  dataset_name: coco
  class_names: [person]
  training_path: ./datasets/od_coco_2017_person/train  # Fallback si quantization_path absent
  quantization_path: ./datasets/od_coco_2017_person/train
  quantization_split: 0.05   # Utiliser seulement 5% → ~600 images sur ~12 000
  seed: 42
```

### 6.5 Commandes et flux d'exécution

**Quantization minimale avec fichier de config dédié :**

```bash
# Depuis le dossier object_detection/
python stm32ai_main.py --config-path ./config_file_examples/ --config-name quantization_config.yaml
```

**Surcharge du mode via la ligne de commande :**

```bash
python stm32ai_main.py operation_mode=quantization
```

**Exemple du fichier `quantization_config.yaml` (prêt à l'emploi) :**

```yaml
operation_mode: quantization

model:
  model_path: ../../stm32ai-modelzoo/object_detection/yolov2t/ST_pretrainedmodel_public_dataset/coco_2017_person/yolov2t_224/yolov2t_224.keras
  model_type: yolov2t

dataset:
  format: tfs
  dataset_name: coco
  class_names: [person]
  quantization_path: ./datasets/od_coco_2017_person/quantization

preprocessing:
  rescaling: { scale: 1/127.5, offset: -1 }
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb

postprocessing:
  confidence_thresh: 0.001
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  plot_metrics: False
  max_detection_boxes: 100

quantization:
  quantizer: TFlite_converter
  quantization_type: PTQ
  quantization_input_type: uint8
  quantization_output_type: float
  export_dir: quantized_models

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

**Flux d'exécution détaillé lors d'une quantization :**

1. **Chargement du YAML** par Hydra → construction de la configuration.
2. **Chargement du modèle float** depuis `model_path` (.keras).
3. **Chargement du dataset de calibration** depuis `quantization_path`, conversion au format TFS si nécessaire.
4. **Prétraitement des images** de calibration (rescaling + resizing).
5. **Calibration TFLite** : passage des images à travers le modèle pour mesurer les plages min/max des activations.
6. **Quantization PTQ** : application des paramètres de quantization (scale, zero-point) par couche.
7. **Export** du modèle `.tflite` quantizé dans `experiments_outputs/<date>/quantized_models/`.
8. **Logging** des métadonnées dans MLflow et le fichier de log.

**Trouver le modèle quantizé :**

```
object_detection/
└── tf/src/experiments_outputs/
    └── 2024_03_15_10_30_45/
        └── quantized_models/
            └── best_model_quantized.tflite  ← Votre modèle quantizé
```

### 6.6 Services enchaînés incluant la quantization

Ces services permettent de combiner plusieurs étapes en un seul lancement :

**`chain_eqe` — Évaluer le float, puis quantizer, puis évaluer le quantizé :**

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name chain_eqe_config.yaml
```

Utile pour comparer la précision avant et après quantization.

**`chain_qb` — Quantizer puis benchmarker sur STM32 :**

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name chain_qb_config.yaml
```

Utile pour estimer rapidement l'empreinte du modèle quantizé sur la cible.

**`chain_tqeb` — Entraîner → Quantizer → Évaluer → Benchmarker :**

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name chain_tqeb_config.yaml
```

Pipeline complet entraînement → déploiement. Exemple de config :

```yaml
operation_mode: chain_tqeb

model:
  model_type: yolov2t
  model_name: yolov2t
  input_shape: (192, 192, 3)

dataset:
  dataset_name: coco
  format: tfs
  class_names: [person]
  training_path: /local/data/od_coco_2017_person/train
  validation_split: 0.1
  test_path: /local/data/od_coco_2017_person/test
  quantization_split: 0.05

quantization:
  quantizer: TFlite_converter
  quantization_type: PTQ
  quantization_input_type: uint8
  quantization_output_type: float
  granularity: per_channel
  optimize: False
  export_dir: quantized_models

benchmarking:
  board: STM32H747I-DISCO

tools:
  stedgeai:
    optimization: balanced
    on_cloud: True
    path_to_stedgeai: C:/ST/STEdgeAI/x.y/Utilities/windows/stedgeai.exe
  path_to_cubeIDE: C:/ST/STM32CubeIDE_x.x.x/STM32CubeIDE/stm32cubeide.exe
```

**`chain_qd` — Quantizer puis déployer directement :**

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name chain_qd_config.yaml
```

### 6.7 Quantization avancée (ONNX — outil d'inspection)

Le projet propose également des outils de quantization avancés basés sur ONNX pour :
- Améliorer le mAP après quantization.
- Concevoir des modèles à précision mixte (poids en 4 bits, activations en 8 bits).
- Investiguer les problèmes de quantization couche par couche.

**Ces modes avancés ne sont disponibles qu'avec `quantizer: Onnx_quantizer`.**

**Mode `inspection` :** analyse la qualité de chaque tenseur quantizé (rapport signal/bruit — SNR) pour identifier les couches problématiques.

```yaml
quantization:
  quantizer: Onnx_quantizer
  quantization_type: PTQ
  operating_mode: inspection   # Mode spécial : analyse couche par couche
  iterative_quant_parameters:
    inspection_split: 0.1      # Utiliser 10% du dataset pour l'inspection
```

**Mode `full_auto` :** recherche automatique des tenseurs pouvant être quantizés en 4 bits sans dépasser un budget de dégradation mAP.

```yaml
quantization:
  quantizer: Onnx_quantizer
  operating_mode: full_auto
  iterative_quant_parameters:
    accuracy_tolerance: 1.0    # Tolérance de dégradation mAP (en %)
  onnx_extra_options:
    SmoothQuant: True
    SmoothQuantAlpha: 0.1      # < 0.5 pour pousser la complexité vers les activations
    SmoothQuantFolding: True
```

**Options ONNX supplémentaires :**

```yaml
quantization:
  quantizer: Onnx_quantizer
  quantization_type: PTQ
  granularity: per_channel
  target_opset: 21             # Recommandé pour 4 bits et 16 bits
  onnx_quant_parameters:
    WeightSymmetric: True
    ActivationSymmetric: False
    CalibMovingAverage: False
    QuantizeBias: True
    weight_type: Int8
    activ_type: Int8
    calibrate_method: MinMax
  onnx_extra_options:
    SmoothQuant: False
    SmoothQuantAlpha: 0.5
    SmoothQuantFolding: True
```

### 6.8 Pièges courants et conseils de dépannage

**❌ Problème : mAP fortement dégradée après quantization**

*Causes possibles et solutions :*
- **Prétraitement incohérent** : vérifiez que `scale` et `offset` dans `preprocessing.rescaling` sont identiques à ceux utilisés lors de l'entraînement. C'est la cause la plus fréquente.
- **Calibration sur données aléatoires** : fournissez un `quantization_path` pointant vers des images réelles représentatives.
- **Dataset de calibration trop petit** : augmentez `quantization_split` ou utilisez plus d'images.
- **Granularité inadaptée** : essayez `per_channel` si vous utilisez `per_tensor`.
- **Passez à l'ONNX quantizer** avec mode `inspection` pour identifier les couches problématiques.

**❌ Problème : Erreur lors du chargement du modèle**

*Causes possibles et solutions :*
- Vérifiez que `model_path` pointe vers un fichier `.keras` (et non `.tflite`) pour la quantization avec TFLite_converter.
- Pour la quantization ONNX, le modèle doit être au format `.onnx`.
- Les formats acceptés par mode sont :

| Mode | Format accepté pour `model_path` |
|---|---|
| `quantization` | Keras (`.keras`, `.h5`) |
| `evaluation` | Keras ou TFLite (`.tflite`) |
| `benchmarking` | Keras, TFLite ou ONNX |
| `deployment` | TFLite (`.tflite`) ou ONNX QDQ |

**❌ Problème : La quantization prend trop de temps**

*Solution :* réduire le dataset de calibration via `quantization_split`. Un split de 0.02–0.05 (2–5% du dataset) est souvent suffisant. Exemple :

```yaml
dataset:
  quantization_path: ./datasets/coco_person/train
  quantization_split: 0.03   # ~360 images sur 12 000
```

**❌ Problème : Le modèle quantizé est trop grand pour la cible**

*Solutions :*
- Utilisez `granularity: per_tensor` (plus compact que `per_channel`).
- Activez `optimize: True` avec `per_tensor`.
- Utilisez le mode ONNX `full_auto` pour quantizer certains poids en 4 bits.
- Réduisez la résolution d'entrée du modèle (`input_shape`).

**❌ Problème : Erreurs MLflow (chemins trop longs sur Windows)**

*Solution :* activez les longs chemins dans le registre Windows (`LongPathsEnabled = 1`) et configurez Git :

```bash
git config --system core.longpaths true
```

**❌ Problème : "No module named" ou erreurs d'importation**

*Solution :* vérifiez que votre environnement virtuel est activé et que toutes les dépendances sont installées :

```bash
pip install -r requirements.txt
```

---

## 7. Évaluation du modèle

Le service d'évaluation calcule les métriques de précision (mAP — mean Average Precision) sur un jeu de test ou de validation.

**Configuration minimale pour l'évaluation :**

```yaml
operation_mode: evaluation

model:
  model_path: chemin/vers/mon_modele.tflite   # float (.keras) ou quantizé (.tflite)
  model_type: yolov2t

dataset:
  format: tfs
  dataset_name: coco
  class_names: [person]
  test_path: ./datasets/od_coco_2017_person/test

preprocessing:
  rescaling: { scale: 1/127.5, offset: -1 }
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb

postprocessing:
  confidence_thresh: 0.001
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  max_detection_boxes: 100

evaluation:
  target: host   # "host" (Python), "stedgeai_host" (C sur PC), "stedgeai_n6" (sur STM32N6)
```

**Lancer l'évaluation :**

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name evaluation_config.yaml
```

**Évaluer avant et après quantization en une seule commande (`chain_eqe`) :**

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name chain_eqe_config.yaml
```

**Résultats :** les métriques mAP sont affichées dans la console et sauvegardées dans `stm32ai_main.log` et dans MLflow.

---

## 8. Benchmarking

Le benchmarking mesure les performances du modèle sur une cible STM32 réelle ou simulée : empreinte Flash, empreinte RAM, temps d'inférence.

**Configuration minimale :**

```yaml
operation_mode: benchmarking

model:
  model_path: chemin/vers/mon_modele_quantize.tflite
  model_type: st_yoloxn

tools:
  stedgeai:
    optimization: balanced   # "balanced", "time", "ram"
    on_cloud: True           # False pour STEdgeAI Core local
    path_to_stedgeai: C:/ST/STEdgeAI/x.y/Utilities/windows/stedgeai.exe

benchmarking:
  board: STM32N6570-DK   # Cartes disponibles : STM32N6570-DK, STM32H747I-DISCO,
                          # STM32H7B3I-DK, STM32F469I-DISCO, B-U585I-IOT02A,
                          # STM32L4R9I-DISCO, NUCLEO-H743ZI2, STM32H735G-DK,
                          # STM32F769I-DISCO, NUCLEO-G474RE, NUCLEO-F401RE,
                          # STM32F746G-DISCO
```

**Lancer le benchmarking :**

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name benchmarking_config.yaml
# ou :
python stm32ai_main.py operation_mode=benchmarking
```

**Résultats :** consultez `stm32ai_main.log` ou l'interface MLflow.

---

## 9. Déploiement sur carte STM32

Le service de déploiement génère automatiquement le code C embarqué et le charge sur la carte STM32.

**Cartes supportées :**
- **STM32N6570-DK** (performances maximales, NPU intégré)
- **STM32H747I-DISCO** (MCU haute performance)
- **STM32MP257F-EV1** (MPU Linux)

**Prérequis matériels :**
- Carte STM32 cible
- Module caméra compatible (pour la détection d'objets temps réel)
- STM32CubeIDE pour compiler le projet C généré

**Configuration minimale pour le déploiement STM32N6 :**

```yaml
operation_mode: deployment

general:
  project_name: coco_person_detection

model:
  model_type: st_yoloxn
  model_path: chemin/vers/mon_modele_int8.tflite   # TFLite int8 ou ONNX QDQ

dataset:
  dataset_name: coco
  class_names: [person]

preprocessing:
  resizing:
    interpolation: bilinear
    aspect_ratio: crop   # "crop" ou "fit" selon la carte
  color_mode: rgb

deployment:
  c_project_path: ../application_code/object_detection/STM32N6
  IDE: GCC
  verbosity: 1
  hardware_setup:
    serie: STM32N6
    board: STM32N6570-DK

tools:
  stedgeai:
    optimization: balanced
    on_cloud: True
  path_to_cubeIDE: C:/ST/STM32CubeIDE_x.x.x/STM32CubeIDE/stm32cubeide.exe
```

**Lancer le déploiement :**

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name deployment_n6_st_yoloxn_config.yaml
```

**Pipeline quantization → déploiement direct (`chain_qd`) :**

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name chain_qd_config.yaml
```

---

## 10. Adapter le projet à un dataset ou un modèle personnalisé

### Dataset personnalisé

**Option 1 : Format YOLO Darknet (recommandé pour débuter)**

```yaml
dataset:
  format: darknet_yolo
  dataset_name: darknet_yolo
  class_names: [chat, chien, voiture]
  data_dir: ./datasets/mon_dataset/   # Dossier contenant train/ et val/
```

**Option 2 : Format Pascal VOC (XML)**

```yaml
dataset:
  format: pascal_voc
  dataset_name: pascal_voc
  class_names: [chat, chien, voiture]
  data_dir: ./datasets/mon_dataset/tmp/
  train_images_path: ./datasets/mon_dataset/JPEGImages/
  train_xml_dir: ./datasets/mon_dataset/Annotations/
```

**Option 3 : Dataset déjà en format TFS (après une première conversion)**

```yaml
dataset:
  format: tfs
  dataset_name: custom_dataset
  class_names: [chat, chien, voiture]
  training_path: ./datasets/mon_dataset/train/
  test_path: ./datasets/mon_dataset/test/
```

**Option 4 : Exclure les images sans annotations**

```yaml
dataset:
  exclude_unlabeled: True
```

### Modèle personnalisé (BYOM — Bring Your Own Model)

**Fine-tuning d'un modèle pré-entraîné :**

```yaml
model:
  model_type: st_yoloxn
  model_name: st_yoloxn_d033_w025
  model_path: chemin/vers/mon_modele_preentraine.keras   # Poids à fine-tuner
  input_shape: (416, 416, 3)
```

**Quantization d'un modèle custom :**

```yaml
operation_mode: quantization

model:
  model_path: chemin/vers/mon_modele_custom.keras
  model_type: st_yoloxn   # Indiquer le type pour le post-traitement

quantization:
  quantizer: TFlite_converter
  quantization_type: PTQ
  quantization_input_type: uint8
  quantization_output_type: float
  granularity: per_channel
  export_dir: quantized_models
```

### Adapter le prétraitement

> **⚠️ Règle fondamentale** : les paramètres `scale` et `offset` de `preprocessing.rescaling` doivent être **strictement identiques** lors de l'entraînement, de la quantization, de l'évaluation et du déploiement.

| Intervalle de normalisation souhaité | `scale` | `offset` |
|---|---|---|
| [0.0, 1.0] | `1/255` | `0` |
| [-1.0, 1.0] | `1/127.5` | `-1` |

---

## 11. Visualisation et suivi des expériences

### MLflow

Chaque run est automatiquement enregistré dans MLflow. Pour visualiser :

```bash
cd object_detection/tf/src/experiments_outputs/
mlflow ui
# Ouvrir http://localhost:5000 dans un navigateur
```

### TensorBoard (pendant l'entraînement)

```bash
tensorboard --logdir ./tf/src/experiments_outputs/<date>/logs/
```

### ClearML (optionnel)

Pour utiliser ClearML, créez un fichier `clearml.conf` avec vos identifiants :

```ini
api {
    web_server: https://app.clear.ml
    api_server: https://api.clear.ml
    files_server: https://files.clear.ml
    credentials {
        "access_key" = "VOTRE_CLE"
        "secret_key" = "VOTRE_SECRET"
    }
}
```

Les expériences apparaissent automatiquement dans le projet nommé `general.project_name`.

### Log principal

Chaque run produit un fichier de log complet :

```
experiments_outputs/<date>/stm32ai_main.log
```

---

## 12. Modèles supportés pour Object Detection

### Modèles TensorFlow

| `model_type` | `model_name` possible | Description |
|---|---|---|
| `st_yoloxn` | `st_yoloxn`, `st_yoloxn_d033_w025`, `st_yoloxn_d100_w025`, `st_yoloxn_d050_w040` | YOLOX anchor-free, variante ST optimisée pour MCU |
| `yolov8n` | (utiliser `model_path`) | YOLOv8 nano d'Ultralytics |
| `yolov11n` | (utiliser `model_path`) | YOLOv11 nano d'Ultralytics |
| `yolov5u` | (utiliser `model_path`) | YOLOv5u d'Ultralytics |
| `st_yololcv1` | `st_yololcv1` | Version légère de Tiny YOLO v2, optimisée MCU |
| `yolov2t` | `yolov2t` | Tiny YOLOv2, très léger |
| `yolov4` | (utiliser `model_path`) | YOLOv4 |
| `yolov4t` | (utiliser `model_path`) | Tiny YOLOv4 |
| `face_detect_front` | (utiliser `model_path`) | BlazeFace pour la détection de visages |

### Modèles PyTorch (ONNX)

| Modèle | Description |
|---|---|
| `SSD_MobileNetV1_pt` | SSD avec backbone MobileNetV1 |
| `SSD_MobileNetV2_pt` | SSD avec backbone MobileNetV2 |
| `SSDLite_MobileNetV1_pt` | SSDLite avec backbone MobileNetV1 |
| `SSDLite_MobileNetV2_pt` | SSDLite avec backbone MobileNetV2 |
| `SSDLite_MobileNetV3Large_pt` | SSDLite avec MobileNetV3 Large |
| `SSDLite_MobileNetV3Small_pt` | SSDLite avec MobileNetV3 Small |
| `ST_YoloDv2Milli_pt` | ST YOLO D v2 Milli (PyTorch) |
| `ST_YoloDv2Tiny_pt` | ST YOLO D v2 Tiny (PyTorch) |

### Cartes STM32 cibles pour Object Detection

| Carte | Modèles recommandés |
|---|---|
| **STM32H747I-DISCO** | ST Yolo LC v1 |
| **STM32N6570-DK** | Tiny YOLOv2, ST YOLOX, YOLOv8, YOLOv11, BlazeFace, modèles PyTorch SSD |
| **STM32MP257F-EV1** | Modèles plus lourds via Linux/MPU |

---

## 13. Annexes

### Annexe A : Syntaxe YAML dans ce projet (Hydra)

Le projet utilise [Hydra](https://hydra.cc/) pour la gestion des configurations. Quelques règles :

- Les valeurs nulles s'écrivent `null` ou laisser le champ vide.
- Les listes YAML : `[item1, item2]` ou en multi-lignes avec `-`.
- Surcharge depuis la ligne de commande : `python stm32ai_main.py param=valeur section.param=valeur`.
- Les expressions comme `${now:%Y_%m_%d_%H_%M_%S}` sont des interpolations Hydra pour la date/heure.

### Annexe B : Pipeline complet recommandé (exemple de référence)

Voici le workflow complet typique pour un projet de détection d'objets :

```
1. Cloner le dépôt et installer les dépendances
   └─ pip install -r requirements.txt

2. Préparer le dataset (ex. COCO 2017 personne)
   └─ Télécharger et placer dans ./datasets/

3. Entraîner le modèle
   └─ python stm32ai_main.py operation_mode=training

4. Vérifier les métriques (mAP) sur TensorBoard/MLflow

5. Quantizer le modèle entraîné
   └─ python stm32ai_main.py operation_mode=quantization

6. Évaluer le modèle quantizé
   └─ python stm32ai_main.py operation_mode=evaluation

7. Comparer mAP float vs quantizé (chain_eqe)
   └─ python stm32ai_main.py operation_mode=chain_eqe

8. Benchmarker sur STM32 (empreinte, latence)
   └─ python stm32ai_main.py operation_mode=benchmarking

9. Déployer sur la carte
   └─ python stm32ai_main.py operation_mode=deployment
```

**Ou en une seule commande avec `chain_tqeb` :**

```bash
python stm32ai_main.py --config-path ./config_file_examples/ --config-name chain_tqeb_config.yaml
```

### Annexe C : Références utiles

| Ressource | Lien |
|---|---|
| STM32 Model Zoo (modèles pré-entraînés) | https://github.com/STMicroelectronics/stm32ai-modelzoo/ |
| STEdgeAI Developer Cloud | https://stedgeai-dc.st.com/home |
| STEdgeAI Core (outil local) | https://www.st.com/en/development-tools/stedgeai-core.html |
| Wiki installation | https://wiki.st.com/stm32mcu/index.php?title=AI:How_to_install_STM32_model_zoo |
| Hugging Face STMicroelectronics | https://huggingface.co/STMicroelectronics |
| Docker Model Zoo | Voir `docker/README.md` dans ce dépôt |
| Pascal VOC 2012 (YOLO Darknet) | https://public.roboflow.com/object-detection/pascal-voc-2012/1/download/darknet |

### Annexe D : Variables d'environnement utiles

```bash
# Limiter les logs TensorFlow (réduire les messages verbeux)
export TF_CPP_MIN_LOG_LEVEL=2

# Désactiver le GPU (forcer CPU)
export CUDA_VISIBLE_DEVICES=-1

# Définir la mémoire GPU maximale (alternative au yaml)
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### Annexe E : Structure d'un fichier YAML de quantization complet et annoté

```yaml
# Fichier de configuration pour la quantization d'un modèle
# d'object detection STM32 AI Model Zoo

# Mode : quantization seule
operation_mode: quantization

# ─── Paramètres globaux ──────────────────────────────────────
general:
  project_name: mon_projet_od          # Nom du projet (visible dans MLflow/ClearML)
  logs_dir: logs                       # Dossier des logs TensorBoard
  saved_models_dir: saved_models       # Dossier des modèles sauvegardés
  global_seed: 123                     # Reproductibilité des résultats
  deterministic_ops: False             # True = déterminisme total (plus lent)
  display_figures: True                # Afficher les courbes (si en local)
  gpu_memory_limit: 16                 # Limite GPU en Go (None = illimité)
  num_threads_tflite: 4               # Threads pour l'inférence TFLite

# ─── Modèle à quantizer ──────────────────────────────────────
model:
  # Chemin vers le modèle float (.keras) à quantizer
  model_path: ./saved_models/best_model.keras
  model_type: st_yoloxn                # Type de modèle (pour le post-traitement)

# ─── Dataset de calibration ──────────────────────────────────
dataset:
  format: tfs                          # Format du dataset
  dataset_name: coco                   # Nom du dataset
  class_names: [person]               # Classes détectées
  # Chemin vers les images de calibration (répertoire TFS)
  quantization_path: ./datasets/od_coco_2017_person/quantization
  # Utiliser seulement 5% des images de calibration (~600 img)
  quantization_split: 0.05
  seed: 42                            # Graine pour le split

# ─── Prétraitement (DOIT être identique à l'entraînement) ────
preprocessing:
  rescaling:
    scale: 1/127.5                     # Normalisation [-1, 1] → même qu'en training !
    offset: -1
  resizing:
    aspect_ratio: fit                  # "fit" = redimensionnement avec distorsion
    interpolation: nearest             # Méthode d'interpolation
  color_mode: rgb                      # Format de couleur

# ─── Post-traitement (pour évaluation éventuelle) ────────────
postprocessing:
  confidence_thresh: 0.001            # Seuil de confiance minimum
  NMS_thresh: 0.5                     # Seuil Non-Maximum Suppression
  IoU_eval_thresh: 0.5               # Seuil IoU pour TP/FP
  plot_metrics: False                 # Tracer les courbes PR
  max_detection_boxes: 100           # Nombre max de boîtes par image

# ─── Paramètres de quantization ──────────────────────────────
quantization:
  quantizer: TFlite_converter         # Outil : TFLite PTQ
  quantization_type: PTQ              # Post-Training Quantization
  quantization_input_type: uint8      # Type entrée du modèle quantizé
  quantization_output_type: float     # Type sortie (float = compatible post-proc)
  granularity: per_channel            # per_channel (précis) ou per_tensor (compact)
  optimize: False                     # Optimisation pré-quantization
  export_dir: quantized_models        # Dossier de sortie du .tflite

# ─── Gestion des sorties (Hydra) ─────────────────────────────
hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}

# ─── Suivi des métriques (MLflow) ────────────────────────────
mlflow:
  uri: ./tf/src/experiments_outputs/mlruns
```

---

*Document rédigé sur la base de la documentation officielle du dépôt [stm32ai-modelzoo-services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) — version 4.0. Pour toute mise à jour, se référer aux fichiers README dans le dossier `object_detection/docs/`.*
