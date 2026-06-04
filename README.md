# 🛡️ Deepfake Detection System using MobileNetV2

## 📌 Overview

This project is an AI-powered Deepfake Detection System developed to identify whether a facial image is **Real** or **Fake (AI Generated/Manipulated)**. The solution leverages **Transfer Learning with MobileNetV2** and provides predictions through an interactive **Streamlit web application**.

The project was developed as part of a FinTech fraud prevention use case focused on mitigating identity fraud and impersonation attacks in digital financial systems.

---

## 🎯 Problem Statement

As digital financial services continue to grow, fraudsters increasingly use deepfake technology to impersonate individuals and bypass identity verification systems.

This project addresses:

- Identity Fraud Detection
- Deepfake Face Detection
- Customer Onboarding Verification
- KYC Fraud Prevention

---

## 🚀 Features

- Upload facial images for analysis
- Detect Real vs Fake images
- Display confidence score
- MobileNetV2 Transfer Learning
- Streamlit-based web interface
- Lightweight and easy to deploy

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- NumPy
- Streamlit
- Pillow

---

## 📂 Project Structure

```text
deepfake-detector/
│
├── app.py
├── model.h5
├── requirements.txt
├── README.md
└── dataset/
    ├── real/
    └── fake/
```

---

## 📊 Dataset

### Source
Kaggle Real and Fake Face Detection Dataset

The dataset consists of:

- Real Face Images
- Fake / AI Generated Face Images

### Preprocessing

The following preprocessing techniques were applied:

- Image Resizing (224 × 224)
- Pixel Normalization
- Horizontal Flipping
- Rotation Augmentation
- Zoom Augmentation

---

## 🧠 Model Architecture

### Base Model
**MobileNetV2 (Pretrained on ImageNet)**

Transfer Learning was used to leverage pretrained image features and improve learning efficiency.

### Classification Head

```python
GlobalAveragePooling2D()
Dense(64, activation='relu')
Dropout(0.3)
Dense(1, activation='sigmoid')
```

### Fine-Tuning

- Most pretrained layers frozen
- Final layers fine-tuned on deepfake dataset
- Adam optimizer used for training

---

## ⚙️ Training Configuration

| Parameter | Value |
|------------|---------|
| Image Size | 224 × 224 |
| Batch Size | 16 |
| Epochs | 6 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |

---

## 📈 Results

| Metric | Score |
|----------|---------|
| Training Accuracy | ~82% |
| Validation Accuracy | ~55% |

> Deepfake detection is a challenging problem. Performance can be further improved using larger datasets, video-based analysis, and advanced architectures.

---

## ▶️ How to Run

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/deepfake-detector.git
cd deepfake-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run app.py
```

---

## 🔄 Application Workflow

1. User uploads an image.
2. Image is resized and normalized.
3. MobileNetV2 extracts facial features.
4. Model predicts whether the image is Real or Fake.
5. Confidence score is displayed.

---

## 🏦 FinTech Use Cases

- Digital KYC Verification
- Fraud Prevention
- Identity Verification Systems
- Customer Onboarding Security
- Banking and Financial Applications

---

## 🔮 Future Enhancements

- Real-time Video Deepfake Detection
- Face Liveness Detection
- Voice Deepfake Detection
- Multi-modal Fraud Detection
- Cloud Deployment
- API Integration

---

## 👨‍💻 Author

**Mrutyunjaya Debata**

B.Tech – Computer Engineering

**Skills**
- Python
- SQL
- Power BI
- Machine Learning
- Deep Learning
- Data Analytics

---

## 📜 License

This project is developed for educational, research, and internship evaluation purposes.

---

⭐ If you found this project useful, consider giving the repository a star.
