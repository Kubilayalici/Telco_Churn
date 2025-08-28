# Telco Müşteri Terk (Churn) Projesi

Telco veri seti üzerinde uçtan uca terk (churn) analizi ve modelleme.

- Yeniden kullanılabilir EDA yardımcıları: `src/utils.DataFrameUtils`
- Temiz ön işleme (ColumnTransformer) ve model seçimi (GridSearch)
- Script tabanlı EDA / eğitim: `notebook/eda.py` ve `notebook/model_trainer.py`
- İnteraktif arayüz: Streamlit (`streamlit_app.py`)

## Hızlı Başlangıç (Yerelde Çalıştırma)

1) Bağımlılıkları kurun (önerilen Python 3.10):

```bash
pip install -r requirements.txt
```

2) (İsteğe bağlı) EDA çalıştırın:

```bash
python notebook/eda.py
```

3) Modeli eğitin (artefaktlar oluşturulur):

```bash
python notebook/model_trainer.py
```

Bu adım sonunda:
- `best_model.pkl`: Eğitim pipeline + en iyi sınıflandırıcı
- `artifacts/metrics.json`: accuracy, precision, recall, f1, roc_auc
- `artifacts/feature_importance.png`: En önemli 30 özellik grafiği

4) Streamlit arayüzünü başlatın:

```bash
streamlit run streamlit_app.py
```

Tarayıcıda açılan sayfadan özellikleri girerek tahmin ve (mümkünse) olasılık çıktısını görebilirsiniz.

Notlar (Kurulum/Troubleshooting):
- XGBoost/CatBoost/LightGBM kurulumu bazı ortamlarda ek araçlar gerektirebilir. Kurulumda sorun yaşarsanız `notebook/model_trainer.py` içindeki GridSearch adaylarını geçici olarak daraltabilirsiniz (ör. yalnızca RandomForest/LogReg).
- Windows’ta sanal ortam (virtualenv/conda) kullanmanız tavsiye edilir.

## GridSearch ve Metrikler

Eğitim scripti küçük bir GridSearch ile en iyi modeli seçer (RandomForest/LogReg/XGB/CatBoost). Aşağıdaki metrikler kaydedilir:
- accuracy, precision, recall, f1, roc_auc (artifacts/metrics.json)
- özellik önemi grafiği (artifacts/feature_importance.png)

## Test ve CI

Lokal test:

```bash
pip install pytest
pytest -q
```

GitHub Actions üzerinde her push’ta lint (yumuşak) ve testler çalıştırılır.
