"""Generate a reproducible synthetic transaction dataset for the project."""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "transactions.csv"

def main(row_count=100_000):
    random.seed(42)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "transaction_id", "transaction_amount", "hour",
            "distance_from_home_km", "transactions_last_24h",
            "merchant_risk_score", "device_changed", "international", "is_fraud"
        ])
        fraud_count = 0
        for tx_id in range(1, row_count + 1):
            amount = round(random.lognormvariate(5.0, 1.0), 2)
            hour = random.randrange(24)
            distance = round(random.expovariate(1 / 12), 2)
            recent_count = random.randint(0, 12)
            merchant_risk = round(random.random(), 3)
            device_changed = int(random.random() < 0.08)
            international = int(random.random() < 0.06)

            # Synthetic rule-based fraud probability with a low overall base rate.
            score = (
                -5.0
                + 0.000018 * min(amount, 100000)
                + 0.035 * min(distance, 150)
                + 0.16 * recent_count
                + 2.2 * merchant_risk
                + 1.5 * device_changed
                + 1.3 * international
                + (0.7 if hour <= 4 else 0.0)
            )
            probability = 1 / (1 + pow(2.718281828, -score))
            is_fraud = int(random.random() < probability)
            fraud_count += is_fraud
            writer.writerow([
                tx_id, amount, hour, distance, recent_count,
                merchant_risk, device_changed, international, is_fraud
            ])

    print(f"SUCCESS: Created {row_count:,} transactions: {OUTPUT}")
    print(f"Synthetic fraud labels generated: {fraud_count:,}")

if __name__ == "__main__":
    main()
