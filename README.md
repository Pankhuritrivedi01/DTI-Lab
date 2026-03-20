**---

---

# 🧬 DTI-Lab — Drug Target Interaction Laboratory

A machine learning web application for predicting drug-target interaction (DTI) probability using molecular features, cheminformatics, and ensemble classifiers.

Built by **Pankhuri Trivedi** | B.Tech Bioinformatics | Amity Institute of Biotechnology, Amity University, Noida

---

## 🚀 Live Demo

👉 [huggingface.co/spaces/ptiee1905/dti-lab](https://huggingface.co/spaces/ptiee1905/dti-lab)

---

## 📌 What It Does

- Predicts the probability of interaction between a **drug molecule** and a **target protein**
- Validates against **20+ ChEMBL / DrugBank** reference pairs (BCR-ABL, EGFR, BRAF, TP53, BRCA1/2...)
- Computes **RDKit molecular descriptors** from SMILES strings
- Applies **Lipinski Rule of Five** drug-likeness filter
- Provides **SHAP feature importance** for interpretable predictions
- Supports **batch prediction** via CSV upload

---

## 🖥️ Features

| Feature | Description |
|---|---|
| Single Prediction | Enter drug + protein + SMILES → instant DTI score |
| Batch Analysis | Upload CSV of drug-target pairs → download results |
| Molecular Descriptors | MW, LogP, HBD, HBA, TPSA, Aromatic Rings (RDKit) |
| SHAP Explainability | Feature importance ranked by contribution |
| Lipinski Check | Rule of Five drug-likeness validation |
| Validated Pairs | ChEMBL/DrugBank reference scores for known drugs |

---

## 🧪 Example Predictions

| Drug | Target | Expected Score |
|---|---|---|
| Imatinib | BCR-ABL | ~97% (validated) |
| Erlotinib | EGFR | ~95% (validated) |
| Aspirin | COX-1 | ~89% (validated) |
| Atorvastatin | HMGCR | ~96% (validated) |
| Caffeine | Adenosine | ~78% (validated) |
| UnknownDrug | TP53 | ML prediction |

---

## 🛠️ Tech Stack

```
Language        Python 3.10+
ML              Scikit-learn, XGBoost, Random Forest
Cheminformatics RDKit, Morgan Fingerprints (2048-bit), SMILES
Databases       ChEMBL, DrugBank, NCBI, UniProt, ClinVar
Explainability  SHAP feature importance
Frontend        Streamlit
Deployment      Hugging Face Spaces
Tracking        MLflow (local), Git
```

---

## 📁 Project Structure

```
dti-lab/
│
├── app.py                  # Main Streamlit application (HF Spaces entry point)
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── api/
│   └── main.py             # FastAPI backend (local use)
│
├── dashboard/
│   └── app.py              # Streamlit dashboard (local use)
│
└── start.py                # Local multi-service launcher
```

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/YOURUSERNAME/DTI-Lab.git
cd DTI-Lab

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# OR run full local stack (API + Dashboard + MLflow)
python start.py
```

---

## 📊 Pipeline Architecture

```
Drug Name + Target Protein + SMILES
              │
              ▼
    ┌─────────────────────┐
    │   RDKit Features    │  MW, LogP, HBD, HBA, TPSA, Rings
    │   Morgan FP (2048)  │  Radius = 2
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │   Random Forest     │  n_estimators=100, max_depth=7
    │   10-fold Strat. CV │  Stratified cross-validation
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │   SHAP Values       │  Feature importance ranking
    │   Lipinski Filter   │  Drug-likeness modifier
    └──────────┬──────────┘
               │
               ▼
         DTI Score (0–1)
    HIGH ≥ 0.85 | MID ≥ 0.60 | LOW < 0.60
```

---

## 📦 Requirements

```
streamlit>=1.32.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
rdkit>=2023.3.1
```

---

## 👩‍🔬 About

**Pankhuri Trivedi**

---

## 📄 License

MIT License — free to use and modify.**
