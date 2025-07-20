# Vlm-clip   RN50

#  PathoVision: AI-Driven Breast Lesion Classification using CLIP

> Classifying benign and malignant breast lesions using Vision–Language Models.

---

##  Abstract

**PathoVision** is an AI-powered framework leveraging **CLIP (Contrastive Language–Image Pre-Training)** to classify breast lesions as **benign** or **malignant**, including **8 subtypes**. Unlike traditional models, CLIP uses natural language prompts to interpret visual data, enabling:

- Zero-shot inference  
- High adaptability  
- Explainability in medical imaging  

---

##  Motivation

Breast cancer is a leading cause of death in women worldwide. Histopathological diagnosis is:
-  Time-consuming  
-  Prone to human error  
-  Dependent on labeled datasets  

**PathoVision** aims to assist pathologists with a robust, explainable, and efficient classification tool.

---

##  Dataset

**Datasets Used:**
- BreakHis  
- PCam  
- CBIS-DDSM  

**Classes:**

>  **Benign**
- Adenosis  
- Fibroadenoma  
- Tubular Adenoma  
- Phyllodes Tumor  

>  **Malignant**
- Ductal Carcinoma  
- Lobular Carcinoma  
- Mucinous Carcinoma  
- Papillary Carcinoma  

**Preprocessing:**
- Converted to RGB  
- Resized to `224×224`  
- Organized into `train`, `val`, and `test` folders  

---

##  Methodology

###  Model Architectures

- `Zero-Shot CLIP (ViT-L/14 & RN50)`  
  → Cosine similarity with text prompts  
- `CLIP + Logistic Regression`  
  → Visual embeddings + Scikit-learn classifier  
- `CLIP + Custom Classifier`  
  → CLIP encoder + FC + Dropout + ReLU  
- `ResNet50 (Fine-tuned)`  
  → End-to-end classification head  

###  Training Setup

- **Loss:** CrossEntropy + Label Smoothing  
- **Optimizer:** Adam (weight decay: `1e-5`)  
- **Scheduler:** StepLR  
- **Epochs:** 35  
- **Hardware:** NVIDIA GPU recommended  

---

##  Results

| Class              | Precision | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| Ductal Carcinoma   | 0.91      | 0.89   | 0.90     |
| Papillary Carcinoma| 0.87      | 0.88   | 0.87     |
| Lobular Carcinoma  | 0.88      | 0.86   | 0.87     |
| Mucinous Carcinoma | 0.89      | 0.90   | 0.89     |
| Adenosis           | 0.93      | 0.92   | 0.92     |
| Fibroadenoma       | 0.86      | 0.85   | 0.85     |
| Phyllodes Tumor    | 0.84      | 0.85   | 0.84     |
| Tubular Adenoma    | 0.87      | 0.89   | 0.88     |

>  **Average F1-Score:** `0.88`  
>  **CLIP Zero-Shot Accuracy:** `85%+`  
>  **Fine-Tuned Accuracy:** `Up to 92%`  

 Predictions saved to `clip_predictions.csv`

---

##  Evaluation Metrics

-  Accuracy  
-  Precision  
-  Recall  
-  F1 Score  
-  Confusion Matrix  
-  Train/Validation Curves  

---

##  System Requirements

###  Hardware
- CPU: Intel i5/i7 or equivalent  
- RAM: 8GB+  
- GPU (recommended): NVIDIA  

###  Software
- Python 3.10+  
- PyTorch  
- OpenAI CLIP  
- Scikit-learn, NumPy, Matplotlib  
- Google Colab / Jupyter Notebook  

---

##  Research Inspiration

- [CLIP by OpenAI](https://openai.com/blog/clip)  
- Pathology-specific VLMs (e.g., PathologyVLM, SupCon-ViT)  
- BreakHis dataset benchmarks  

 For 25+ reviewed research papers, see `research_analysis.md`

---

##  Conclusion

**PathoVision** showcases the capability of Vision–Language Models in histopathology:

-  Zero-shot clinical prompt-based classification  
-  Reliable across 8 detailed lesion subtypes  
-  Strong performance even with limited labeled data  
-  Opens doors to privacy-aware, explainable AI in healthcare  

---

##  Future Work

-  Incorporate multimodal data (e.g., patient reports + images)  
-  Add explainability tools (e.g., Grad-CAM for CLIP)  
-  Explore federated learning across hospitals  

---

##  Contributing

Coming soon...  
Feel free to open issues or fork the repo to contribute!

---

##  Disclaimer

This project is **not** a medical device and is intended **for research purposes only**. It should not be used for clinical decision-making without expert review.

---

##  Contact

For collaborations or queries, reach out via GitHub or email.


