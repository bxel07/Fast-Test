# FastAPI Computer Vision Project

A FastAPI application with computer vision capabilities using OpenCV, YOLOv8, and EasyOCR.

## Project Structure
```
fastapi-cv-project/
├── .gitignore
├── README.md
├── requirements.txt
├── app/
│   └── main.py
└── venv/
```

## Installation Guide

### 1. Clone the Repository


### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
uvicorn app.main:app --reload

If you're creating this project from scratch:

1. Create `.gitignore` file:
```
# Virtual Environment
venv/
env/
ENV/
