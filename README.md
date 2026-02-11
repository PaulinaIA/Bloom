<div align="center">

# Bloom

![mi imagen](assets/logo.png)

**Tu fertilidad en tus manos**

[![Python](https://img.shields.io/badge/Python-3.10+-9b59b6?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-e91e63?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-4caf50?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-8e44ad?style=for-the-badge)]()

<br>

*Sistema de predicción adaptativa de fertilidad que combina Machine Learning con personalización individual para ayudar a mujeres y parejas a entender sus ciclos menstruales.*

<br>

[Notebook](#-quickstart) · [Arquitectura](#-architecture) · [Resultados](#-results) · [API](#-api-reference) · [Deployment](#-deployment)

</div>

---

## 📋 Problem

> **El 15% de las parejas enfrentan problemas de infertilidad.** Muchas no tienen patologías graves—solo falta de sincronización y conocimiento de sus biomarcadores.

Las apps existentes usan predicciones genéricas basadas en un ciclo "promedio" de 28 días. Pero cada persona tiene patrones únicos.

## 💡 Solution

Bloom implementa un **sistema híbrido** que combina:

| Componente | Descripción |
|:-----------|:------------|
| **Modelo poblacional** | Aprende patrones de miles de usuarias |
| **Personalización adaptativa** | Ajusta predicciones según historial individual |
| **Ponderación dinámica** | Usuarias regulares → más peso a historial personal |

---

## 🎯 Targets

| Objetivo | Métrica | Impacto en Producto |
|:---------|:-------:|:--------------------|
| Predecir día de ovulación | **95% accuracy** (±2 días) | Ventana fértil |
| Estimar duración del ciclo | **RMSE < 2 días** | Próximo período |
| Detectar anomalías | **Recall > 90%** | Alertas de salud |

---

## 🏗 Architecture

```
╔════════════════════════════════════════════════════════════════════════╗
║                     BLOOM PREDICTION SYSTEM v1.0                       ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐    ║
║   │  DataLoader  │ ─▶ │FeatureEngine │ ─▶ │   Model Pipeline     │    ║
║   │  (Marquette) │    │              │    │                      │    ║
║   └──────────────┘    └──────────────┘    │  ┌────────────────┐  │    ║
║                                           │  │CyclePredictor  │  │    ║
║                                           │  └────────────────┘  │    ║
║                                           │  ┌────────────────┐  │    ║
║                                           │  │OvulationClass  │  │    ║
║                                           │  └────────────────┘  │    ║
║                                           │  ┌────────────────┐  │    ║
║                                           │  │AnomalyDetector │  │    ║
║                                           │  └────────────────┘  │    ║
║                                           └──────────────────────┘    ║
║                                                      │                ║
║                                                      ▼                ║
║                              ┌────────────────────────────────────┐   ║
║                              │       AdaptivePredictor            │   ║
║                              │  (combines all + personalization)  │   ║
║                              └────────────────────────────────────┘   ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/yourusername/bloom-fertility.git
cd bloom-fertility
pip install -r requirements.txt
```

### Usage

```python
from bloom import BloomAdaptivePredictor, MarquetteDataLoader

# Load data
loader = MarquetteDataLoader(filepath='data/marquette.csv')
df = loader.load()

# Train system
bloom = BloomAdaptivePredictor()
bloom.fit(df)

# Add user history
bloom.add_user_cycle(user_id=1, data={'cycle_length': 28, 'ovulation_day': 14})

# Get prediction
prediction = bloom.predict(user_id=1, features=current_features)

print(f"Next cycle: {prediction.predicted_cycle_length} days")
print(f"Ovulation: day {prediction.ovulation.predicted_day}")
print(f"Fertile window: day {prediction.fertile_window_start}-{prediction.fertile_window_end}")
```

---

## 📊 Results

### Ovulation Prediction

| Metric | Value | Target |
|:-------|:-----:|:------:|
| Exact accuracy | ~45% | - |
| Within ±1 day | ~75% | - |
| **Within ±2 days** | **~92%** | **95%** |
| MAE | ~1.2 days | - |

### Cycle Length Prediction

| Metric | Value | Target |
|:-------|:-----:|:------:|
| **RMSE** | **~1.8 days** | **< 2 days** |
| MAE | ~1.4 days | - |
| R² | ~0.85 | - |

---

## 📖 API Reference

### `BloomAdaptivePredictor`

```python
class BloomAdaptivePredictor:
    def fit(self, df: pd.DataFrame) -> 'BloomAdaptivePredictor'
    def add_user_cycle(self, user_id: int, data: Dict) -> None
    def predict(self, user_id: int, features: pd.Series) -> BloomPrediction
    def get_user_stats(self, user_id: int) -> Dict
```

### `BloomPrediction`

```python
@dataclass
class BloomPrediction:
    user_id: int
    predicted_cycle_length: float
    cycle_confidence_interval: Tuple[float, float]
    ovulation: OvulationPrediction
    fertile_window_start: int
    fertile_window_end: int
    anomaly_alerts: List[AnomalyAlert]
    prediction_source: str
```

---

## 🚢 Deployment

### AWS Architecture

```
React Native App → API Gateway → Lambda → Aurora + S3 + DynamoDB
```

### Export Models

```python
import joblib

joblib.dump(bloom.cycle_predictor.model, 'models/cycle_predictor.joblib')
joblib.dump(bloom.ovulation_classifier.regressor, 'models/ovulation_regressor.joblib')
```

---

## 📁 Project Structure

```
bloom-fertility/
├── README.md
├── requirements.txt
├── bloom_fertility_system.ipynb    # Main notebook
├── bloom/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineer.py
│   ├── predictors.py
│   ├── anomaly_detector.py
│   └── visualizer.py
├── models/
│   ├── cycle_predictor.joblib
│   ├── ovulation_regressor.joblib
│   └── bloom_config.json
└── data/
    └── marquette.csv
```

---

## 📚 Dataset

**Universidad de Marquette - Planificación Familiar Natural**

- 1,666 registros de ciclos menstruales
- Variables: duración ciclo, fase lútea, día ovulación, intensidad sangrado, factores de salud

---

## 👩‍💻 Team

<table>
  <tr>
    <td align="center">
      <b>Katherine Soto</b><br>
      <sub>Co-founder</sub>
    </td>
    <td align="center">
      <b>Paulina Peralta</b><br>
      <sub>Co-founder</sub>
    </td>
  </tr>
</table>

<div align="center">
<i>Creado por mujeres, para mujeres</i> 🌸
</div>

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**[⬆ Back to top](#-bloom)**

</div>
