# Telco Customer Churn (ML Project)

End-to-end churn analysis and modeling on Telco dataset with:
- Reusable EDA/helpers (src/utils.DataFrameUtils)
- Clean preprocessing (ColumnTransformer) and model selection
- Scripted EDA/training under `notebook/` (Python scripts)

## Quickstart

```bash
pip install -r requirements.txt
python notebook/eda.py      # optional EDA run
python notebook/model_trainer.py  # train and save best_model.pkl
```

## GridSearch + Metrikler

Eğitim scripti küçük bir GridSearch ile en iyi modeli seçer (RF/LogReg/XGB/CatBoost). Aşağıdaki metrikler artifacts/metrics.json dosyasına kaydedilir:

- accuracy, precision, recall, f1, roc_auc
- feature_importance.png: ağaç/katsayı tabanlı önem grafiği

Çıktılar: `artifacts/metrics.json`, `artifacts/feature_importance.png`

## Streamlit Arayüzü

Eğitim sonrası interaktif arayüzü çalıştırın:

```bash
streamlit run streamlit_app.py
```

Form ile özellikleri girin; model (best_model.pkl) üzerinden churn tahmini ve olasılığı gösterilir.

## Tests & CI

```bash
pip install pytest
pytest -q
```

CI workflow runs lint (soft) + tests on push.
