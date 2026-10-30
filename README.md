# Fraud Detection Using Big Data (Apache Spark + PySpark)

A complete academic mini-project that demonstrates large-scale transaction fraud detection with Apache Spark.

## Project overview
This project generates a synthetic bank transaction dataset, processes it with Spark, trains a Logistic Regression classifier using Spark ML, evaluates the model, and exports predictions and summary metrics.

**Technology:** Python 3, Apache Spark, PySpark 3.5.1, Spark MLlib  
**Dataset:** Synthetic transaction data (100,000 rows by default; no real customer or banking data)  
**Output:** Model metrics, confusion matrix, transaction-level predictions, and fraud-risk summaries.

## Easiest setup: Google Colab (recommended for Windows users)
This route avoids local Windows Hadoop / `winutils.exe` configuration.

1. Go to https://colab.research.google.com/
2. Create a **New notebook**.
3. Upload `Fraud_Detection_Big_Data.zip` using the folder icon on the left.
4. Run these cells in order:

```python
!unzip -qo Fraud_Detection_Big_Data.zip
%cd Fraud_Detection_Big_Data
!pip -q install -r requirements.txt
!python src/generate_data.py
!python src/train_model.py
```

5. Open the `outputs` folder in Colab's left Files panel. Spark writes output into folders; open the folder whose name ends in `.csv` and download its `part-....csv` file.
6. For a downloadable ZIP of all output reports, run:

```python
!zip -qr fraud_detection_outputs.zip outputs
from google.colab import files
files.download("fraud_detection_outputs.zip")
```

If you already have a dataset, you can replace `data/transactions.csv` but keep the same column names and formats described below.

## Run locally in VS Code (Windows)
**Recommended:** Python 3.9–3.11 and Java 17. Local Spark on Windows may require Hadoop winutils configuration. If you previously saw `HADOOP_HOME` errors, use Colab instead.

1. Extract the ZIP.
2. In VS Code choose **File → Open Folder** and select `Fraud_Detection_Big_Data`.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

```powershell
py -m venv .venv
```

5. Activate it:
   - PowerShell: `.venv\Scripts\Activate.ps1`
   - Command Prompt: `.venv\Scripts\activate.bat`

6. Install dependencies:

```bash
pip install -r requirements.txt
```

7. Generate the dataset:

```bash
python src/generate_data.py
```

8. Train and evaluate the model:

```bash
python src/train_model.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. Do not download random `winutils.exe` files from unknown websites.

## Project folder structure

```text
Fraud_Detection_Big_Data/
├── data/
│   └── transactions.csv             # generated dataset
├── models/                           # trained Spark ML model
├── outputs/
│   ├── metrics.json
│   ├── confusion_matrix.csv/
│   ├── predictions.csv/
│   └── fraud_summary.csv/
├── src/
│   ├── generate_data.py              # creates synthetic data
│   └── train_model.py                # Spark processing + ML model
├── requirements.txt
└── README.md
```

## Dataset columns
- `transaction_id`: unique transaction number
- `transaction_amount`: transaction value
- `hour`: transaction hour (0–23)
- `distance_from_home_km`: distance from customer's usual location
- `transactions_last_24h`: recent transaction count
- `merchant_risk_score`: merchant risk score (0–1)
- `device_changed`: 1 if a new device was used, otherwise 0
- `international`: 1 if international, otherwise 0
- `is_fraud`: target label (1 = synthetic fraud, 0 = normal)

## What the analysis does
1. Reads the CSV using Spark DataFrames.
2. Splits records into training and testing sets.
3. Assembles numeric features with `VectorAssembler`.
4. Trains a Logistic Regression model with Spark ML.
5. Calculates ROC AUC, accuracy, precision, recall, and F1.
6. Writes transaction predictions, a confusion matrix, and fraud summary reports.

## Academic note / limitations
This is an educational demonstration using generated synthetic data. Results are not suitable for real banking decisions. Real fraud detection requires carefully governed, representative data, privacy controls, human review, and monitoring for changing fraud patterns.
