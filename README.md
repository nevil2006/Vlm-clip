# Vlm-clip   RN50
PathoVision: AI-Driven Breast Lesion Classification using CLIP
Classifying benign and malignant breast lesions using Vision-Language Models.

Abstract
PathoVision is an AI-powered framework that leverages the power of CLIP (Contrastive Language–Image Pre-Training) to classify breast lesions (from histopathology images) into benign or malignant, including 8 subtypes. Unlike traditional models, CLIP interprets visual inputs using natural language prompts, enabling zero-shot inference, high adaptability, and explainability in medical imaging.

Motivation
Breast cancer is a leading cause of death in women globally. Early detection through histopathological analysis is crucial, yet expert diagnosis is:

Time-consuming

Prone to human error

Limited by the availability of labeled data

PathoVision aims to assist pathologists by providing an interpretable, fast, and robust classification tool.

Dataset
Used datasets include:

BreakHis

PCam

CBIS-DDSM

Classes:
Benign:

Adenosis

Fibroadenoma

Tubular Adenoma

Phyllodes Tumor

Malignant:

Ductal Carcinoma

Lobular Carcinoma

Mucinous Carcinoma

Papillary Carcinoma

 Preprocessing:
Converted to RGB

Resized to 224×224

Organized into train, val, and test folders

 Methodology
 Model Architectures:
Zero-Shot CLIP (ViT-L/14 & RN50)
→ Classifies using cosine similarity with text prompts

CLIP + Logistic Regression
→ CLIP visual embeddings + Scikit-learn classifier

CLIP + Custom Classifier (Fine-tuned)
→ CLIP encoder + dense head (FC + Dropout + ReLU)

ResNet50 (Fine-tuned)
→ Custom head trained end-to-end

 Training Setup:
Loss: CrossEntropy + Label Smoothing

Optimizer: Adam, Weight Decay = 1e-5

Scheduler: StepLR, Epochs = 35

Hardware: NVIDIA GPU (recommended)

 Results
Class	Precision	Recall	F1-Score
Ductal Carcinoma	0.91	0.89	0.90
Papillary Carcinoma	0.87	0.88	0.87
Lobular Carcinoma	0.88	0.86	0.87
Mucinous Carcinoma	0.89	0.90	0.89
Adenosis	0.93	0.92	0.92
Fibroadenoma	0.86	0.85	0.85
Phyllodes Tumor	0.84	0.85	0.84
Tubular Adenoma	0.87	0.89	0.88

 Average F1-Score: 0.88

CLIP Zero-Shot Accuracy: 85%+

Fine-tuned Accuracy: Up to 92%

Evaluation stored in clip_predictions.csv

 Evaluation Metrics
Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Train/Val Curves

 System Requirements
 Hardware:
CPU: Intel i5/i7 or equivalent

RAM: 8GB+

GPU (recommended): NVIDIA (for fine-tuning)

 Software:
Python 3.10+

PyTorch

CLIP by OpenAI

Scikit-learn, NumPy, Matplotlib

Google Colab or Jupyter Notebook

 Research Inspiration
CLIP (OpenAI)

Pathology-specific VLMs (e.g., PathologyVLM, SupCon-ViT)

BreakHis dataset studies

See the research_analysis.md (if you add it) for a detailed summary of 25+ related papers.

 Conclusion
PathoVision proves that Vision–Language Models can revolutionize medical diagnostics:

Zero-shot capability with clinical text prompts

Reliable classification across 8 breast lesion types

Strong performance in low-data scenarios

Opens doors for scalable, privacy-preserving, explainable AI in healthcare

 Future Work
Expand to multi-modal data (e.g., reports + images)

Add explainability tools (e.g., Grad-CAM for CLIP)

Explore federated learning for privacy

