# 👁️ Eye Disease Classifier  

A deep learning project that uses **ResNet50** (pretrained on ImageNet) to classify eye diseases from retinal images.  
The model was fine-tuned on a custom dataset and achieves **~85% accuracy on the test set** ✅.  

Deployed on **Hugging Face Spaces** for real-time inference.  

---

## 🌐 Live Demo  

🔗 Try it here: [Eye Disease Classifier - Hugging Face Space](https://huggingface.co/spaces/arneshpal/eye-disease-classifier)  

---

## 📸 Screenshot  

| Classifier UI |
| :---: |
| <img width="1267" height="729" alt="Demo Screenshot" src="https://github.com/user-attachments/assets/139fbeab-0192-4971-a7b4-699e1806e8c5" /> |

---

## 📊 Dataset & Training  

- Dataset: Retinal images (4 classes) stored in **ImageFolder format**.  
- Split:  
  - 🏋️ Training: 2700 images  
  - 🔎 Validation: 676 images  
  - 🧪 Testing: 844 images  
- Preprocessing:  
  - Resize → `(224,224)`  
  - Normalization → ImageNet mean/std  
- Model: **ResNet50** pretrained weights, final FC layer modified to 4 classes.  
- Training setup:  
  - Optimizer: **Adam** (LR = 0.001)  
  - Loss: **CrossEntropyLoss**  
  - Epochs: **10**  
  - Early checkpointing with resume support  

---

## 📈 Results  

- ✅ Training Accuracy: ~86%  
- ✅ Validation Accuracy: ~86%  
- 🧪 Test Accuracy: **85.31%**  

Model Checkpoints:  
- 🔒 Final weights → `resnet50_final.pth`  
- 📦 Full model → `resnet50_full_model.pth`  

---

## 🚀 Features  

- 📂 Upload retinal image & get classification in real-time  
- 🧠 Uses **Transfer Learning (ResNet50)**  
- ⚡ Fast & lightweight deployment on Hugging Face Spaces  
- 🛠️ Training pipeline with checkpoint saving & resume support  

---

## 🛠️ Technologies  

- **Python** 🐍  
- **PyTorch** 🔥  
- **Torchvision** 🖼️  
- **scikit-learn** 📊  
- **tqdm** ⏳  
- **Hugging Face Spaces (Gradio UI)** 🤗  

---

## 📥 Usage (Local Training)  

```bash
# Clone repo
git clone https://github.com/your-username/eye-disease-classifier.git
cd eye-disease-classifier

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Evaluate on test set
python evaluate.py
