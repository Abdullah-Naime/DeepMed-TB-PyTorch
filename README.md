# 🫁 TBVision: Chest X-Ray Tuberculosis Classifier

A modular deep learning and computer vision portfolio project designed to classify chest X-rays as Normal or TB-related using PyTorch transfer learning and SQLite database logging.

## 🚀 Features
- **Transfer Learning:** Uses ResNet-18 pre-trained on ImageNet, with custom fine-tuning on layer4 to capture medical X-ray textures.
- **Data Pipeline:** Custom PyTorch `Dataset` and `DataLoader` with stratified train/validation/test splits and image augmentation.
- **Database Persistence:** Automatically logs scan predictions, confidence scores, and validation status into a local SQLite database (`tb_vision.db`).
- **Evaluation Metrics:** Evaluates performance using classification reports, confusion matrices, and optimized recall metrics for medical screening.

## 🛠️ Tech Stack
- **Python 3 / PyTorch / Torchvision** (Deep Learning & CNN model)
- **SQLite 3** (Metadata & inference logging)
- **Scikit-Learn & Matplotlib** (Metrics evaluation & visualization)
- **Google Colab & WSL** (Development environment)

## 📦 Project Structure
```text
TBVision/
├── main.py                   # CLI interface & workflow runner
├── database.py               # SQLite connection & logging manager
├── tb_vision_resnet18.pth    # Saved fine-tuned model weights
├── .gitignore                # Ignored cache & database files
└── README.md                 # Project documentation
