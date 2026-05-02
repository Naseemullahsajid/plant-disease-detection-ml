# plant-disease-detection-ml
Deep learning-based plant disease classification system using PyTorch (ResNet/CNN), with full ML pipeline including data preprocessing, training, evaluation, and deployment via FastAPI and Docker.
# 🌿 Plant Disease Detection API

End-to-End Deep Learning System (PyTorch + FastAPI + Docker)


---

Overview

A **production-ready deep learning system** for detecting plant diseases from leaf images using CNN / ResNet (PyTorch).
This project demonstrates a **complete ML lifecycle** — from data preprocessing to deployment via FastAPI and Docker.

---

Key Highlights

* 🧠 Transfer Learning with ResNet18
* 📊 93% Accuracy on validation data
* 🔍 Error Analysis (Confusion Matrix + Misclassifications)
* ⚡ FastAPI for real-time inference
* 🐳 Dockerized for production deployment
* 📦 Modular and scalable codebase

---



## 🚀 Features

* End-to-end ML pipeline
* Image classification (8 disease classes)
* REST API for predictions
* Top-K predictions with confidence
* Ready for cloud deployment

---

## 📊 Model Performance

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 93%   |
| Precision | 91%   |
| Recall    | 89%   |
| F1 Score  | 90%   |

---

## 🌐 API Demo

```bash
http://127.0.0.1:8000/docs
```

---

## 🎯 Sample Prediction

```json
{
  "prediction": "Tomato__Target_Spot",
  "confidence": 0.67
}
```

---

## 📂 Project Structure

1. split_data.py
        ↓
2. eda.py
        ↓
3. visualize.py
        ↓
4. data_loader.py
        ↓
5. test_dataloader.py  
        ↓
6. model.py
        ↓
7. early_stopping.py
        ↓
8. train.py   → best_model.pth
        ↓
9. evaluate.py
        ↓
10. infer.py
        ↓
11. app.py (FastAPI)
        ↓
12. Dockerfile





## 👨‍💻 Author

**Naseem Ullah Sajid**
📧 [naseemsajid653@gmail.com](mailto:naseemsajid653@gmail.com)
🔗 https://www.linkedin.com/in/naseem-ullah-sajid-0963641a7




