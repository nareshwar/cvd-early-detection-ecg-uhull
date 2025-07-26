# ECG-CVD-AI

This project uses deep learning (LSTM and Transformer models) for early detection of cardiovascular disease (CVD) from ECG time-series data. It also integrates explainable AI (XAI) to make model predictions interpretable and clinically meaningful.

## 📊 Goals
- Compare LSTM vs Transformer for ECG classification
- Apply Explainable AI (e.g. SHAP, saliency maps, attention)
- Use PhysioNet Challenge 2020 ECG data
- Evaluate on AUC, F1-score, specificity, sensitivity

## 📁 Data
Due to size restrictions, ECG datasets are not included in this repo.  
To download:
```bash
kaggle competitions download -c physionet-challenge-2020
unzip physionet-challenge-2020.zip -d data/raw/
 
