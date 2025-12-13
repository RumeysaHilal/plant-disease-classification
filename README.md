# 🌱 AI-Powered Plant Disease Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-MobileNetV2-green?style=for-the-badge)

A robust, web-based end-to-end computer vision application designed to identify plant diseases from leaf images. Built using **Transfer Learning with MobileNetV2**, this project addresses real-world deployment challenges such as image orientation, aspect ratio distortion, and background noise.

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features & Innovations](#-key-features--innovations)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Model Details](#-model-details)
- [Challenges & Solutions](#-challenges--solutions)
- [Contact](#-contact)

---

## 🔍 Project Overview

Early detection of plant diseases is a critical factor in agricultural productivity and food security. Manual inspection is often slow, error-prone, and requires expert knowledge. 

This project automates the diagnosis process using Deep Learning. By leveraging the **Plant Village Dataset**, the model can classify various diseases across different plant species (e.g., Pepper, Potato, Tomato) with high precision. The application is deployed via **Streamlit**, providing a simple and accessible interface for farmers and agricultural technicians.

---

## 🌟 Key Features & Innovations

Unlike standard "Hello World" classification projects, this system includes **production-grade logic** to handle messy, real-world data:

### 1. 🔄 Smart Rotation Invariance (TTA - Test Time Augmentation)
Convolutional Neural Networks (CNNs) are not naturally rotation-invariant. A leaf photographed upside down might be misclassified.
* **The Solution:** The system acts as an ensemble at inference time. It generates 4 variations of the uploaded image (**0°, 90°, 180°, 270°**), predicts on all of them, and selects the result with the highest confidence score.

### 2. ⚡ Adaptive Aspect Ratio Optimization
To optimize computational resources and improve accuracy:
* **Landscape Images:** Detected automatically via `width > height`. The system intelligently restricts the search space to **90° and 270°** rotations to correct the orientation before resizing.
* **Portrait Images:** The system scans all 4 cardinal directions.
* *Benefit:* Prevents "squashing" distortion when resizing rectangular images to a square (128x128) input.

### 3. 📊 Top-3 Probabilistic Output
Fine-grained classification (e.g., distinguishing *Pepper Bell Bacterial Spot* from *Potato Early Blight*) can be challenging due to morphological similarities.
* Instead of a single "Black Box" answer, the system displays the **Top-3 most likely diagnoses** with their confidence percentages, providing an analytical tool for the user.

### 4. 🛡️ Background Suppression & Robustness
* **The Problem:** Standard models force a prediction even if the user uploads a photo of a cat or a wall.
* **The Solution:** A specific class for **"Background_without_leaves"** is integrated. If the model predicts this class or if the confidence score is below a certain threshold, the system triggers a **"No Leaf Detected"** warning.

---

## 🏗️ System Architecture

The data pipeline follows a structured path from upload to diagnosis:

1.  **Input Layer:** User uploads an image (JPG/PNG/JPEG).
2.  **Preprocessing Module:**
    * **Exif Correction:** Auto-fixes orientation metadata from mobile cameras.
    * **Smart Rotation Logic:** Generates image batches based on aspect ratio.
    * **Resizing:** Bilinear interpolation to `128x128` pixels.
    * **Normalization:** Scales pixel values to `[-1, 1]` (MobileNetV2 specific).
3.  **Inference Engine:** The pre-trained MobileNetV2 model processes the batch.
4.  **Decision Module:** * Aggregates scores from all rotations.
    * Checks against the "Background" class.
    * Filters results based on confidence thresholds.
5.  **UI/UX Layer:** Streamlit renders the results, Top-3 charts, and visualizes the "Best Angle" used for the diagnosis.

---

## 💻 Tech Stack

* **Core Logic:** Python 3.x
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** PIL (Pillow), NumPy
* **Web Framework:** Streamlit
* **Model Architecture:** MobileNetV2 (Pre-trained on ImageNet)

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_PROJECT_NAME.git](https://github.com/YOUR_USERNAME/YOUR_PROJECT_NAME.git)
cd YOUR_PROJECT_NAME