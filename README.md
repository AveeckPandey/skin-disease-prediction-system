# 🔬 Skin Disease Detection AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

**An AI-powered web application for dermoscopic skin disease classification with Grad-CAM explainability.**

</div>

---

## 📸 Demo

| Upload & Predict | Grad-CAM Heatmap |
|:-:|:-:|
| Upload a skin image and get an instant diagnosis | Visual explanation of where the model focused |

> ⚠️ **Medical Disclaimer:** This tool is intended for educational and research purposes only. It does not replace professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for any skin concerns.

---

## ✨ Features

- 🩺 **Multi-class skin disease classification** across bacterial, viral, and fungal conditions
- 🌡️ **Confidence scoring** — percentage confidence for each prediction
- 🔥 **Grad-CAM heatmap** — visual explanation highlighting the regions the model used to make its decision
- 🖱️ **Drag-and-drop upload** — supports JPG, PNG, WEBP, BMP
- ⚡ **Real-time inference** via FastAPI backend
- 🎨 **Clean, modern dark UI** built with React

---

## 🦠 Supported Conditions

| Category | Conditions |
|---|---|
| **Viral (VI-)** | Chickenpox, Shingles |
| **Bacterial (BA-)** | Cellulitis, Impetigo |
| **Fungal (FU-)** | Athlete's Foot, Nail Fungus, Ringworm |
| **Parasitic (PA-)** | Cutaneous Larva Migrans |

---

## 🗂️ Project Structure

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
Body: file (image)
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

- **Architecture:** Convolutional Neural Network (CNN)
- **Input size:** 224×224 RGB
- **Output:** Softmax over N disease classes
- **Explainability:** Grad-CAM (Gradient-weighted Class Activation Mapping) overlaid on the original image to highlight diagnostically relevant regions

---

## 📦 Dependencies

### Backend
```
fastapi
uvicorn
tensorflow / keras
numpy
opencv-python
Pillow
python-multipart
```

### Frontend
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

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391) — Selvaraju et al.
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
