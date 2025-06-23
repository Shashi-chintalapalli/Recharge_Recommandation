# Vodafone Idea Recharge Recommendation

## Goal
Recommend optimal recharge plan (data/combo/top-up) using usage history.

## Models Used
- Random Forest (scikit-learn)
- Deep Learning (TensorFlow)

## Folders
- `data/` – Dataset
- `models/` – Saved models
- `scripts/` – Training & prediction
- `api/` – FastAPI backend

## How to Run
1. `python scripts/train_model.py`
2. `python scripts/predict_plan.py`
3. (Optional) `uvicorn api.main:app --reload`
