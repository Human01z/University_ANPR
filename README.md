# University ANPR System

An AI-powered Automatic Number Plate Recognition (ANPR) system developed as an undergraduate project.

The system combines Computer Vision, Deep Learning, OCR, Arduino-based vehicle detection, and a Django web application to automatically recognize vehicle number plates at a university entrance.

---

## Features

- Custom-trained YOLO Number Plate Detector
- Continuous Vehicle Tracking
- Rolling Frame Buffer
- IR Sensor Trigger System
- Smart OCR using PaddleOCR
- Multi-frame OCR Voting
- Vehicle Entry/Exit Classification
- Django Web Dashboard
- Vehicle Database
- Event Logging
- Automatic Image Upload
- Green Bounding Box Visualization

---

## Technologies Used

Python

YOLOv11 (Ultralytics)

PyTorch

OpenCV

PaddleOCR

Django

SQLite

Arduino UNO

IR Sensors

USB / RTSP Camera

Bootstrap

HTML

CSS

JavaScript

---

## System Architecture

Camera

↓

Continuous Video Capture

↓

Rolling Frame Buffer

↓

YOLO Plate Detection

↓

Plate Tracking

↓

IR Trigger

↓

Event Queue

↓

Smart OCR

↓

Sri Lankan Plate Validation

↓

Django Backend

↓

Website Dashboard

---

## Installation

### 1 Clone Repository

git clone https://github.com/YOUR_USERNAME/University_ANPR.git

cd University_ANPR

---

### 2 Create Virtual Environment

Windows

python -m venv .venv

.venv\Scripts\activate

Linux

python3 -m venv .venv

source .venv/bin/activate

---

### 3 Install Dependencies

pip install -r requirements.txt

---

### 4 Download YOLO Model

Place

v3.pt

inside

models/

---

### 5 Configure Camera

Inside

tapo_trigger_gpu.py

Change

CAMERA_MODE

to

usb

or

rtsp

Configure

RTSP_URL

or

USB_CAMERA_INDEX

---

### 6 Configure Arduino

Update

ARDUINO_PORT

Example

COM3

COM5

COM9

---

### 7 Run Django

cd backend

python manage.py migrate

python manage.py createsuperuser

python manage.py runserver

---

### 8 Run ANPR

Open another terminal

Activate virtual environment

Run

python tapo_trigger_gpu.py

---

## Folder Structure

backend/

Django Website

ai/

AI Detection System

models/

YOLO Model

arduino/

Arduino Sketch

captures/

Debug Images

---

## Team

• Senuja Jayamanna



---

## License

MIT License
