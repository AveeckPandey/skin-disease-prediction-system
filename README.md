<div align="center">

# 🔬 Skin Disease Detection AI

**Deep Learning · Grad-CAM Explainability · FastAPI · React**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

*An end-to-end AI-powered web application for dermoscopic skin disease classification, featuring Grad-CAM visual explainability.*

[Features](#-features) · [Demo](#-demo) · [Architecture](#-architecture) · [Getting Started](#-getting-started) · [API Reference](#-api-reference) · [Model Details](#-model-details)

</div>

---

## 🧠 What This Project Does

This application allows users to upload an image of a skin condition and receive:

- An **AI-generated diagnosis** across 8 common skin diseases (bacterial, viral, fungal, and parasitic)
- A **confidence score** for the prediction
- A **Grad-CAM heatmap** that visually highlights the exact regions of the image the model used to make its decision — making the AI transparent and interpretable

This bridges the gap between powerful deep learning models and real-world explainability, a critical requirement in clinical and medical AI applications.

> ⚠️ **Medical Disclaimer:** This tool is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for any skin concerns.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🩺 **Multi-class Classification** | Diagnoses 8 skin conditions across 4 disease categories |
| 📊 **Confidence Scoring** | Outputs a softmax probability score per prediction |
| 🔥 **Grad-CAM Explainability** | Highlights diagnostically relevant image regions with a heatmap overlay |
| 🖱️ **Drag-and-Drop Upload** | Intuitive UI supporting JPG, PNG, WEBP, BMP formats |
| ⚡ **Real-time Inference** | Sub-second predictions via an optimised FastAPI backend |
| 🎨 **Modern Dark UI** | Clean, responsive interface built with React |

---

## 🖼️ Demo

| Step | Preview |
|:-:|:-:|
| **Upload a skin image** | Drag & drop or browse to select |
| **AI analyses the image** | Model runs inference in milliseconds |
| **View diagnosis + heatmap** | See the prediction, confidence score, and Grad-CAM overlay |

> Screenshots of real predictions are included in `result-1.png` through `result-8.png` in the root of this repository.

---

## 🦠 Supported Conditions

| Category | Conditions |
|---|---|
| 🦠 **Viral** | Chickenpox, Shingles |
| 🧫 **Bacterial** | Cellulitis, Impetigo |
| 🍄 **Fungal** | Athlete's Foot, Nail Fungus, Ringworm |
| 🪱 **Parasitic** | Cutaneous Larva Migrans |

---

## 🏗️ Architecture

```
skin-disease-detection/
├── backend/
│   ├── main.py              # FastAPI app — /predict endpoint
│   ├── model/
│   │   └── model.h5         # Trained CNN model weights
│   └── utils/
│       └── gradcam.py       # Grad-CAM heatmap generation
├── frontend/
│   ├── src/
│   │   └── App.jsx          # React UI
│   ├── package.json
│   └── public/
└── README.md
```

**Stack at a glance:**

```
React (frontend) ──── POST /predict ────► FastAPI (backend)
                                                │
                                          TensorFlow CNN
                                                │
                                         Grad-CAM module
                                                │
                            ◄── JSON: prediction + confidence + heatmap (base64)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- pip / npm

---

### 🔧 Backend Setup

```bash
# Clone the repository
git clone https://github.com/your-username/skin-disease-detection.git
cd skin-disease-detection/backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be live at `http://127.0.0.1:8000`.  
Interactive Swagger docs available at `http://127.0.0.1:8000/docs`.

---

### 🎨 Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The app will open at `http://localhost:3000`.

---

## 🔌 API Reference

### `POST /predict`

Accepts a skin image and returns the predicted condition, confidence score, and a base64-encoded Grad-CAM heatmap.

**Request**

```
Content-Type: multipart/form-data
Body: file (image — JPG, PNG, WEBP, or BMP)
```

**Response**

```json
{
  "prediction": "VI-chickenpox",
  "confidence": 0.9998,
  "heatmap": "<base64-encoded JPEG string>"
}
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| **Architecture** | Convolutional Neural Network (CNN) |
| **Input Size** | 224 × 224 RGB |
| **Output** | Softmax probability over N disease classes |
| **Explainability** | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| **Framework** | TensorFlow / Keras |

**Why Grad-CAM?**

Standard deep learning models are often criticised as "black boxes." This project integrates Grad-CAM to produce saliency heatmaps that visually explain which skin regions the model focused on — essential for building trust in medical AI systems.

---

## 📦 Dependencies

**Backend**
```
fastapi
uvicorn
tensorflow / keras
numpy
opencv-python
Pillow
python-multipart
```

**Frontend**
```
react
axios
```

---

## 🖥️ Usage

1. Open the app at `http://localhost:3000`
2. Drag and drop a skin image (or click to browse)
3. Click **Run Analysis**
4. View the predicted condition, confidence score, and Grad-CAM heatmap

---

## 🙏 Acknowledgements

- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) — Selvaraju et al., ICCV 2017
- [FastAPI](https://fastapi.tiangolo.com/) — modern, fast web framework for building APIs with Python
- [React](https://reactjs.org/) — the library for web and native user interfaces

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ — feel free to ⭐ this repo if you found it useful!**

</div>
