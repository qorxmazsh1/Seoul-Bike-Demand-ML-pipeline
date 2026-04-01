 Seoul Bike Sharing — End-to-End ML Pipeline

Predicting hourly bike rental demand in Seoul using a full data science pipeline:  
**PostgreSQL → Python → Feature Engineering → Machine Learning**

---

## Dataset

**Seoul Bike Sharing Demand** dataset from UCI / Kaggle  
- **8,760 rows** — hourly records for 1 full year (2017–2018)  
- **Target:** `rented_bike_count` — number of bikes rented per hour  
- **Features:** temperature, humidity, wind speed, solar radiation, rainfall, snowfall, season, holiday, hour

---

##  Pipeline Overview

```
Raw CSV
   │
   ▼
PostgreSQL (raw_data table)
   │
   ▼
SQL Cleaning  ──►  cleaned_data table
   │  • Remove non-functioning days
   │  • Deduplicate
   │  • IQR outlier removal
   │  • Type conversions
   ▼
Feature Engineering
   ├── Python (pandas)   ──►  fe_python_data table
   │    • Month, day of week, weekend flag
   │    • Time-of-day buckets
   │    • Log transform of target
   │    • Interaction terms (temp×humidity, temp×solar)
   │    • Season & time encoding
   │    • Rush hour flag
   │
   └── SQL (window functions) ──►  fe_sql_data table
        • Time bucket & rush hour
        • Z-score normalization (temp, humidity)
        • Rolling averages by season & hour
        • Temperature category
        • Bad weather flag
        • Bike count quartile
        • Log transform
   │
   ▼
Model Training & Evaluation
   │  • Linear Regression
   │  • Ridge
   │  • Lasso
   │  • Random Forest  ◄── Best
   │  • Gradient Boosting
   ▼
Results stored in PostgreSQL (model_results table)
```

---

##  Results

| Model             |  RMSE  |  MAE   |   R²   | CV R² |
|-------------------|--------|--------|--------|-------|
| **Random Forest** | 132.60 |  86.50 | 0.9471 | 0.8889|
| Gradient Boosting | 151.39 | 105.68 | 0.9311 | 0.8864|
| Linear Regression | 233.97 | 180.70 | 0.8355 | 0.4284|
| Ridge             | 233.97 | 180.70 | 0.8355 | 0.4284|
| Lasso             | 234.00 | 180.73 | 0.8354 | 0.4293|

**Best model: Random Forest** with **R² = 0.947** on test set.

### Top Feature Importances (Random Forest)

| Feature                  | Importance |
|--------------------------|------------|
| Temperature (°C)         | 35.8%      |
| Hour                     | 28.5%      |
| Solar Radiation (MJ/m²)  | 11.3%      |
| Humidity (%)             | 8.0%       |
| Dew Point Temperature    | 4.9%       |

---

##  Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

- **Database:** PostgreSQL + SQLAlchemy
- **Data Processing:** pandas, NumPy, SQL window functions
- **Visualization:** matplotlib, seaborn
- **ML:** scikit-learn (LinearRegression, Ridge, Lasso, RandomForestRegressor, GradientBoostingRegressor)

---

##  Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn sqlalchemy psycopg2
```

### Setup

1. Clone the repo:
```bash
git clone https://github.com/qorxmazsh1/seoul-bike-demand-ml-pipeline.git
cd seoul-bike-demand-ml-pipeline
```

2. Set up PostgreSQL and update the `DB_CONFIG` in the notebook:
```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'ml_pipeline_db',
    'user': 'your_username',   # use env variable in production
    'password': 'your_password'
}
```

3. Download the dataset:  
   [Seoul Bike Sharing Demand — Kaggle](https://www.kaggle.com/datasets/saurabhshahane/seoul-bike-sharing-demand-prediction)

4. Run the notebook:
```bash
jupyter notebook ml_pipeline.ipynb
```

---

##  Project Structure

```
├── ml_pipeline.ipynb    # Main notebook (full pipeline)
├── README.md
└── venv/                # Virtual environment (not tracked)
```

---

##  Key Takeaways

- **SQL-based feature engineering** (window functions, z-scores, NTILE) integrates seamlessly with sklearn pipelines
- **Random Forest** significantly outperforms linear models for this non-linear demand prediction task
- **Temperature and Hour** are the strongest predictors of bike demand
- Cross-validation R² (0.889) confirms the model generalizes well

---

##  Author

**qorxmazsh1**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)]([https://linkedin.com/in/qorxmazsh1](https://www.linkedin.com/in/gorkhmaz-shahbazli-433111250/))
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:shahbazliqorxmaz@gmail.com)
