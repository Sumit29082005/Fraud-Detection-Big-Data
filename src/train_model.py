
import os
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_json, sum as spark_sum, count

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "transactions.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Start Spark
spark = (
    SparkSession.builder
    .appName("FraudDetectionBigData")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("\nReading transaction dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. "
            "Run python src/generate_data.py first."
        )

    df = spark.read.csv(
        str(DATA_FILE),
        header=True,
        inferSchema=True
    )

    # Feature columns
    feature_columns = [
        "transaction_amount",
        "hour",
        "distance_from_home_km",
        "transactions_last_24h",
        "merchant_risk_score",
        "device_changed",
        "international"
    ]

    # Check required columns
    required_columns = feature_columns + [
        "transaction_id",
        "is_fraud"
    ]

    missing = [
        name for name in required_columns
        if name not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Convert features and label to numeric types
    for name in feature_columns + ["is_fraud"]:
        df = df.withColumn(name, col(name).cast("double"))

    df = df.dropna(
        subset=feature_columns + ["is_fraud"]
    )

    print("Total transactions:", df.count())

    # Assemble features
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features",
        handleInvalid="skip"
    )

    # Logistic Regression model
    classifier = LogisticRegression(
        featuresCol="features",
        labelCol="is_fraud",
        maxIter=20
    )

    pipeline = Pipeline(
        stages=[assembler, classifier]
    )

    # Split dataset
    train_data, test_data = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    print("\nTraining model...")

    model = pipeline.fit(train_data)

    print("Model training completed!")

    # Predictions
    predictions = model.transform(test_data)

    # Evaluation
    evaluator = BinaryClassificationEvaluator(
        labelCol="is_fraud",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc = evaluator.evaluate(predictions)

    # Confusion matrix values
    results = predictions.select(
        col("is_fraud").alias("actual"),
        col("prediction").alias("predicted")
    )

    tp = results.filter(
        (col("actual") == 1) &
        (col("predicted") == 1)
    ).count()

    tn = results.filter(
        (col("actual") == 0) &
        (col("predicted") == 0)
    ).count()

    fp = results.filter(
        (col("actual") == 0) &
        (col("predicted") == 1)
    ).count()

    fn = results.filter(
        (col("actual") == 1) &
        (col("predicted") == 0)
    ).count()

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn
    }

    # Save metrics
    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    # Save predictions as CSV
    # Convert probability vector into JSON string
    prediction_output = predictions.select(
        "transaction_id",
        "is_fraud",
        "prediction",
        "probability"
    ).withColumn(
        "probability",
        to_json(vector_to_array(col("probability")))
    )

    prediction_output.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv(str(OUTPUT_DIR / "predictions"))

    # Save confusion matrix
    confusion_data = [
        ("True Positive", tp),
        ("True Negative", tn),
        ("False Positive", fp),
        ("False Negative", fn)
    ]

    confusion_df = spark.createDataFrame(
        confusion_data,
        ["category", "count"]
    )

    confusion_df.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv(str(OUTPUT_DIR / "confusion_matrix"))

    # Save fraud summary
    fraud_summary = df.groupBy("is_fraud").agg(
        count("*").alias("total_transactions")
    )

    fraud_summary.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv(str(OUTPUT_DIR / "fraud_summary"))

    # Save trained model
    model.write().overwrite().save(str(MODEL_DIR / "fraud_model"))

    print("\n========== PROJECT RESULTS ==========")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    print(f"AUC       : {auc:.4f}")

    print("\nResults saved in outputs folder.")
    print("Trained model saved in models/fraud_model folder.")
    print("\nPROJECT COMPLETED SUCCESSFULLY!")

finally:
    spark.stop()
