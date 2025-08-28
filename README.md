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

## Tests & CI

```bash
pip install pytest
pytest -q
```

CI workflow runs lint (soft) + tests on push.
