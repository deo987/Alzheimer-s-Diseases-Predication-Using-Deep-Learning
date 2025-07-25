
# Alzheimer Detection with Deep Learning (Backend + Frontend)

A web-based application that uses deep learning models to detect Alzheimer’s disease from uploaded brain scan images. Built with **Python (Flask)** for the backend, with pretrained `.h5` models, and a **React.js** frontend. Data can be stored using **MySQL** or **MongoDB**.

---

## 🚀 Project Structure

AlzheimersApp/
│
├── backend/
│ ├── main.py # Flask backend code
│ ├── requirements.txt # Python dependencies
│ ├── static/
│ ├── templates/
│ ├── uploads/
│ └── models/ # (optional) model files (.h5)
│ ├── cnn_model.h5
│ ├── transfer_model.h5
│ └── hybrid_model.h5
│
├── frontend/
│ ├── public/
│ ├── src/
│ ├── package.json
│ └── README.md # Frontend instructions
│
├── .gitignore
└── README.md # This file

---

## ⚙️ Features

- **Upload brain scan images**
- **Use pre-trained deep learning models** (`cnn_model.h5`, `transfer_model.h5`, `hybrid_model.h5`) for detection
- **Display prediction results** in the frontend
- Choose between **MySQL** (relational) or **MongoDB** (document-based) for persistence
- **Modular design**: easily extend models or UI

---

## 🧪 Backend Setup (Flask + Models)

### 1. Create & activate Python environment:
cd backend
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# or:
venv\Scripts\activate      # Windows

pip install -r requirements.txt

python main.py

