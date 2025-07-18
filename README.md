# ✍️ Handwritten Digit Recognizer – Streamlit App

Welcome to the **AI-powered Handwritten Digit Recognizer**!  
This simple yet powerful app uses a neural network trained on the MNIST dataset to classify handwritten digits (0–9). Upload an image and let the model guess what digit you've written!

---

## 🎯 Features

- ✅ Upload any digit image (JPG/PNG)
- 🧠 Neural network trained on MNIST
- 🖼️ Real-time digit prediction
- ⚡ Built with **PyTorch + Streamlit**
- 📐 Auto-resizes and pre-processes your image
- 🎨 (Optional) Draw your digit directly in-browser using canvas (upcoming)

---

## 🧠 Model Overview

The digit recognizer is a **FeedForward Neural Network** with:

- 🔸 Input: 28x28 grayscale images
- 🔸 Layers:
  - FC1: 784 → 128
  - ReLU
  - FC2: 128 → 64
  - ReLU
  - FC3: 64 → 10 (Logits for digits 0–9)

Trained using **Adam optimizer** and **CrossEntropyLoss** for 3 epochs on the MNIST training set.

---

## 📦 Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/your-username/digit-recognizer.git
cd digit-recognizer
