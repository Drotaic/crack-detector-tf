# Crack Detector TF

AI-powered structural crack detector using TensorFlow and MobileNetV2 — designed for UAV-based damage assessment.

## 📌 Overview
This project demonstrates a binary image classifier to detect cracks in structures using transfer learning with MobileNetV2.

## 🚁 Application
Useful for drones inspecting earthquake-damaged buildings, infrastructure, or bridges.

## 🧠 Model
- Architecture: MobileNetV2
- Classes: Crack / No-Crack
- Input: 224x224 RGB images

## 🛠️ How to Use
1. Place your dataset inside `data/`
2. Run `notebooks/training_notebook.ipynb` to train the model
3. Use `scripts/predict.py` to make predictions

## 📊 Output
Model outputs prediction: `crack` or `no-crack` along with confidence.

## 🔧 Requirements
See `requirements.txt` to install dependencies.

## 🔮 Future Work
- Deploy as real-time drone module
- Improve robustness to low-light/crushed surfaces
