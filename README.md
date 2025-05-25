# COGNIFLOW AI: Intelligent Adaptive Traffic Signal and Vehicle Classification System

This project proposes an AI-powered adaptive traffic management system to address urban congestion by dynamically adjusting signal timings based on real-time vehicle detection and classification using deep learning.

The system consists of two primary components:

1. **Interface Mode (User-facing)**
   - A React-based frontend.
   - A FastAPI backend for processing video input and generating predictions.

2. **Model-only Mode**
   - Standalone scripts for training and evaluating vehicle detection/classification models using YOLOv11 and YOLOv8.

---

## 🚀 Features

- Real-time vehicle detection and classification using YOLO.
- Number plate recognition via YOLOv11.
- Adaptive signal control based on real-time traffic conditions.
- Frontend dashboard for live monitoring and interaction.
- REST API powered by FastAPI for modular backend logic.

---

## 🧠 Technologies Used

- **Frontend:** React, Tailwind CSS
- **Backend:** Python, FastAPI, OpenCV, Uvicorn
- **ML Models:** YOLOv11 (number plate detection), YOLOv8 (vehicle classification)
- **Simulation:** SUMO (Simulation of Urban Mobility)
- **Tools:** PyTorch, Roboflow, LabelImg, COCO/Cityscapes datasets

---




✅ Responsive UI – Optimized for both desktop and mobile users.
---

## 🧑‍💻 Getting Started

### Prerequisites

- Node.js (v16+)
- Python (3.9+)
- pip
- virtualenv (recommended)
- Git

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/pandeyaditya8707/FinalYearProject
cd cogniflow-ai
cd frontend
npm install
npm run dev


cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start FastAPI server
uvicorn app.main:app --reload


cd model
python detect.py --source path/to/video.mp4 --weights yolov8.pt --conf 0.25

---

Let me know if you'd like:
- Docker support added
- GitHub actions / CI setup
- Deployment on platforms like Vercel or Render
- Installation scripts or `.env` template files

I'm happy to help!
Aditya Pandey, Avanish Shukla, Anshima Goel, Aniket Mishra.  
“COGNIFLOW AI: Intelligent Adaptive Traffic Signal and Vehicle Classification System.”  
Ajay Kumar Garg Engineering College, 2025.
