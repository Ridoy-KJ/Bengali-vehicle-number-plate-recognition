# 🚗 Bengali Vehicle Number Plate Recognition Using Deep Learning and OCR

An end-to-end computer vision system for detecting and recognizing **Bengali vehicle number plates** from real-world images.  
This undergraduate final-year project combines **image preprocessing**, **deep learning**, **optical character recognition (OCR)**, and a **Flask web interface**.

> 🔗 **Dataset:** [Bangladeshi Vehicle Number Plates — Kaggle](https://www.kaggle.com/datasets/ridoykj/bangladeshi-vehicle-number-plates)  
> 📓 **Kaggle Notebook:** [Automatic Bengali Number Plate Recognition](https://www.kaggle.com/code/ridoykj/automatic-bengali-number-plate-recognition)

---

## 📌 Overview

Automatic Number Plate Recognition (ANPR) is an important computer vision application used in traffic monitoring, parking management, toll collection, and vehicle identification.

This project focuses on **Bengali number plates**, which present additional challenges compared with English written plates, including:

- Bengali characters  
- Variations in fonts, plate designs, and vehicle types  
- Different lighting, angles, and image-quality conditions  
- Real-world noise such as blur, shadows, and background clutter  

The system processes an input vehicle image, extracts the number plate region, recognizes the Bengali characters, and displays the result through a Flask-based web application.

---

## ✨ Key Features

- Real-world Bengali vehicle number plate dataset  
- Image preprocessing and enhancement using OpenCV  
- Deep learning-based recognition using TensorFlow and InceptionResNetV2  
- Bengali text extraction using Tesseract OCR  
- Flask web interface for uploading and processing images  
- Modular pipeline covering data preparation, model development, OCR, and deployment  

---

## 🧠 Project Pipeline

```text
Input Vehicle Image
        ↓
Image Preprocessing (OpenCV)
        ↓
Number Plate Localization / Extraction
        ↓
Deep Learning Model (InceptionResNetV2)
        ↓
Bengali Text Recognition (Tesseract OCR)
        ↓
Recognized Number Plate Text
        ↓
Flask Web Interface Output
```

---

## 🗂️ Dataset

The project uses the **Bangladeshi Vehicle Number Plates** dataset, collected and curated through real-world fieldwork.

| Item | Details |
|---|---|
| Dataset name | Bangladeshi Vehicle Number Plates |
| Total files | 3,631 |
| Dataset size | Approximately 8 GB |
| Data type | Vehicle number plate images |
| Language | Bengali |
| Source | Kaggle |
| Dataset link | [Bangladeshi Vehicle Number Plates](https://www.kaggle.com/datasets/ridoykj/bangladeshi-vehicle-number-plates) |

The dataset was used to develop and evaluate a recognition pipeline for Bengali vehicle number plates under real-world conditions. [34]

---

## 🛠️ Technologies Used

| Category | Tools / Libraries |
|---|---|
| Programming language | Python |
| Deep learning | TensorFlow, Keras |
| CNN architecture | InceptionResNetV2 |
| Image processing | OpenCV |
| OCR | Tesseract OCR |
| Web interface | Flask |
| Development environment | Jupyter Notebook, Kaggle |
| Data platform | Kaggle |

---

## 🧪 Methodology

### 1. Data Collection and Curation

A dataset of **3,631 images** was collected and curated through fieldwork. The images contain Bengali vehicle number plates captured in real-world environments.

### 2. Image Preprocessing

OpenCV was used to improve image quality and prepare the images for recognition. Preprocessing steps included resizing, noise reduction, contrast adjustment, and plate-region extraction.

### 3. Deep Learning Model

A TensorFlow-based deep learning model was developed using **InceptionResNetV2**. Transfer learning was used to extract meaningful visual features from number plate images.

### 4. OCR-Based Text Extraction

After the plate region was processed, **Tesseract OCR** was used to extract the Bengali characters from the number plate.

### 5. Web Application

A **Flask** web interface was implemented so users can upload a vehicle image and view the recognized number plate output.

---

## 🚀 How to Run

> Full setup instructions will be updated after the complete source code and model files are added.

```bash
# Clone the repository
git clone [https://github.com/your-username/bengali-vehicle-number-plate-recognition.git](https://github.com/your-username/bengali-vehicle-number-plate-recognition.git)

# Navigate to the project folder
cd bengali-vehicle-number-plate-recognition

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app/app.py
```

Then open the application in your browser:

```text
http://127.0.0.1:5000/
```

---

## 📊 Results

Evaluation metrics and sample prediction outputs will be added after the final experiment results are organized.

Planned results section:

- Model training and validation accuracy  
- Character recognition performance  
- Sample input and output images  
- Training curves  
- Example Flask application screenshots  

---

## 🖼️ Sample Output

> Sample images, detected number plates, and OCR outputs will be added soon.

---

## 📁 Repository Structure

```text
bengali-vehicle-number-plate-recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── plate_detection.py
│   ├── train_model.py
│   ├── ocr.py
│   └── utils.py
│
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── models/
│   └── README.md
│
├── results/
│   ├── sample_outputs/
│   └── training_curves.png
│
└── docs/
    └── architecture.png
```

---

## 🚧 Project Status

This repository is currently being organized.

The following will be added soon:

- Complete source code  
- Jupyter notebooks  
- Flask web application files  
- Model training and evaluation details  
- Sample images and prediction outputs  
- Installation and reproduction instructions  

---

## ⚠️ Limitations

- Recognition performance may reduce for blurred, low-light, or heavily angled images.  
- Tesseract OCR may face difficulty with certain Bengali fonts, plate designs, or image-quality conditions.  
- The system is trained for Bangladeshi vehicle number plates and may not generalize directly to plates from other countries.  

---

## 🔮 Future Work

- Improve plate localization using object detection models such as YOLO  
- Replace or complement Tesseract with an end-to-end Bengali text recognition model  
- Improve performance under low-light and motion-blur conditions  
- Deploy the application to a cloud platform  
- Add real-time recognition using webcam or CCTV footage  

---

## 📄 License

The associated Kaggle notebook, **Automatic Bengali Number Plate Recognition**, is released under the **Apache 2.0 License**. [31]

---

## 👤 Author

**Ridoy Kanto Joy**  
Undergraduate Final Year Project  
[GitHub](https://github.com/Ridoy-KJ) | [Kaggle](https://www.kaggle.com/ridoykj)
