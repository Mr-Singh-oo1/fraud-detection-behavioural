import pandas as pd, numpy as np, joblib
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix
from difflib import get_close_matches
from datetime import datetime

# Step 1: Required columns
REQUIRED_COLS = [
    "transaction_id", "event_time", "user_id", "device_id", "amount",
    "merchant_category", "channel", "payment_method", "country",
    "typing_speed", "nav_speed", "geo_distance_km", "is_fraud"
]

print("✅ Required columns in your dataset:")
for col in REQUIRED_COLS:
    print("-", col)

# Step 2: Load dataset
data_dir = Path.cwd() / "data"
if not data_dir.exists():
    raise FileNotFoundError("❌ No 'data' folder found in current directory.")

files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))
if not files:
    raise FileNotFoundError("❌ No .parquet or .csv files found in 'data' folder.")

print("\n📁 Available datasets in 'data' folder:")
for i, f in enumerate(files):
    print(f"{i+1}. {f.name}")

choice = int(input("\n🔢 Enter the number of the dataset to use: ")) - 1
selected_file = files[choice]
print(f"\n✅ Selected: {selected_file.name}")

if selected_file.suffix == ".parquet":
    df = pd.read_parquet(selected_file).reset_index(drop=True)
elif selected_file.suffix == ".csv":
    df = pd.read_csv(selected_file).reset_index(drop=True)
else:
    raise ValueError("Unsupported file format.")

# Step 3: Validate and fill missing columns
print("\n🔍 Validating required columns...")
missing = [col for col in REQUIRED_COLS if col not in df.columns]
if not missing:
    print("✅ All required columns are present.")
else:
    for col in missing:
        print(f"⚠️ Missing column: {col}")
        suggestion = get_close_matches(col, df.columns, n=1)
        if suggestion:
            print(f"   💡 Did you mean: {suggestion[0]}?")
        print(f"   ➕ Filling '{col}' with default value.")
        df[col] = 0 if col != "event_time" else pd.Timestamp("now")

# Step 4: Defensive fill
print("\n🧼 Cleaning and formatting columns...")
df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce").fillna(pd.Timestamp("now"))
for col in ["typing_speed", "nav_speed", "geo_distance_km", "amount"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
df["transaction_id"] = df["transaction_id"].astype(str)
df = df.sort_values("event_time")

# Step 5: Time features
print("🕒 Creating time-based features...")
df["hour"] = df["event_time"].dt.hour
df["dayofweek"] = df["event_time"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)

# Step 6: Velocity and amount anomaly
print("\n📈 Computing transaction velocity and amount anomalies...")
df["tx_count_1d"] = df.groupby("user_id")["event_time"].transform(lambda x: x.expanding().count())
df["tx_count_7d"] = df.groupby("user_id")["event_time"].transform(lambda x: x.expanding().count())

rolling_avg = (
    df.set_index("event_time")
      .groupby("user_id")["amount"]
      .rolling("30d").mean()
      .rename("avg_amount_30d")
      .reset_index()
)
df = pd.merge(df, rolling_avg, on=["user_id", "event_time"], how="left")
df["avg_amount_30d"] = pd.to_numeric(df["avg_amount_30d"], errors="coerce").fillna(0)

df["amount_over_user_avg"] = df["amount"] / (df["avg_amount_30d"] + 1e-6)
df["amount_over_user_avg_log1p"] = np.log1p(df["amount_over_user_avg"])
df["is_amount_5x_user_avg"] = (df["amount_over_user_avg"] > 5).astype(int)
print("✅ Amount anomaly features computed.")

# Step 7: Unique devices in past 30 days
print("\n🔄 Computing unique_devices_30d...")
df["unique_devices_30d"] = 0
user_groups = df.groupby("user_id")
for i, (user_id, group) in enumerate(user_groups):
    group = group.sort_values("event_time")
    result = []
    for j, row in group.iterrows():
        window_start = row["event_time"] - pd.Timedelta(days=30)
        window = group[(group["event_time"] >= window_start) & (group["event_time"] <= row["event_time"])]
        result.append(window["device_id"].nunique())
    df.loc[group.index, "unique_devices_30d"] = result
    if (i + 1) % 100 == 0 or i == len(user_groups) - 1:
        print(f"   ✅ Processed {i + 1} users...")
df["unique_devices_30d"] = pd.to_numeric(df["unique_devices_30d"], errors="coerce").fillna(0).astype(int)
print("✅ Finished computing unique_devices_30d.")

# Step 8: Velocity features
print("\n📊 Computing velocity features (1h and 24h)...")
velocity_1h = (
    df.set_index("event_time")
      .groupby("user_id")["user_id"]
      .rolling("1h").count()
      .rename("velocity_1h")
      .reset_index()
)
velocity_24h = (
    df.set_index("event_time")
      .groupby("user_id")["user_id"]
      .rolling("1d").count()
      .rename("velocity_24h")
      .reset_index()
)

for vdf, name in [(velocity_1h, "velocity_1h"), (velocity_24h, "velocity_24h")]:
    dupes = vdf.duplicated(subset=["user_id", "event_time"], keep=False)
    if dupes.any():
        print(f"⚠️ Found {dupes.sum()} duplicate {name} rows — aggregating")
        vdf = vdf.groupby(["user_id", "event_time"], as_index=False).mean()
        if name not in vdf.columns:
            last_col = vdf.columns[-1]
            print(f"🔧 Renaming column '{last_col}' to '{name}'")
            vdf.rename(columns={last_col: name}, inplace=True)

    df = pd.merge(df, vdf, on=["user_id", "event_time"], how="left")
    if name in df.columns:
        df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0).astype(int)
        print(f"✅ {name} merged and cleaned.")
    else:
        print(f"❌ Merge failed: {name} not found — filling with 0")
        df[name] = 0

# Step 9: Behavioral scoring
print("\n🧠 Computing behavioral fraud risk score...")
def score_typing(x): return 2 if x < 1 or x > 6 else 1 if x < 2 or x > 4 else 0
def score_nav(x): return 2 if x < 0.1 or x > 1 else 1 if x < 0.3 or x > 0.6 else 0
def score_geo(x): return 2 if x > 1000 else 1 if x > 50 else 0
def score_amt(x): return 2 if x > 2 else 1 if x > 1 else 0
def score_velocity(x): return 2 if x > 5 else 1 if x > 2 else 0

df["fraud_risk_score_weighted"] = (
    df["typing_speed"].apply(score_typing) +
    df["nav_speed"].apply(score_nav) +
    df["geo_distance_km"].apply(score_geo) +
    df["velocity_1h"].apply(score_velocity) +
    df["velocity_24h"].apply(lambda x: 1 if x > 20 else 0) +
    df["amount_over_user_avg"].apply(score_amt)
)
print("✅ Behavioral scoring complete.")

# Step 10: Final features
print("\n🧮 Preparing final feature set for model...")
df["log_amount"] = np.log1p(df["amount"])
FEATURES = [
    "log_amount", "hour", "dayofweek", "is_weekend",
    "typing_speed", "nav_speed",
    "tx_count_1d", "tx_count_7d", "avg_amount_30d", "amount_over_user_avg_log1p",
    "is_amount_5x_user_avg", "fraud_risk_score_weighted", "unique_devices_30d",
    "channel", "payment_method", "country", "merchant_category"
]
for col in FEATURES:
    if col not in df.columns:
        df[col] = "missing" if df[col].dtype == object else 0
print("✅ Final feature set prepared.")

#Step 11: Load model and predict
print("\n🤖 Loading trained fraud model and scoring transactions...")
model_path = Path(r"C:\Fraud Dataset\artifacts\xgb_behavioral_pipeline.joblib")
pipe = joblib.load(model_path)

X = df[FEATURES]
y = df["is_fraud"].astype(int)
probs = pipe.predict_proba(X)[:, 1]
df["model_score"] = probs
print("✅ Model scores computed.")

# Step 12: Flag alerts
print("\n🚨 Flagging high-risk transactions...")
threshold = 0.85
df["selected_flag"] = (df["model_score"] >= threshold).astype(int)
print(f"✅ Alerts flagged using threshold {threshold}")

# ✅ Defensive check before metrics
if "selected_flag" not in df.columns:
    print("❌ selected_flag column missing — creating fallback")
    df["selected_flag"] = (df["model_score"] >= threshold).astype(int)

# Step 12.5: Data Quality Indicators
print("\n📋 Data Quality Summary:")
print(f"Total records: {len(df)}")
print(f"Missing values: {df.isnull().sum().sum()} across {df.isnull().any().sum()} columns")
print(f"Duplicate transactions: {df['transaction_id'].duplicated().sum()}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"Unique devices: {df['device_id'].nunique()}")
print(f"Fraud rate: {df['is_fraud'].mean():.4f}")

# Step 12.6: Feature Distribution Checks
print("\n📈 Feature Distribution Overview:")
for col in ["amount", "typing_speed", "nav_speed", "geo_distance_km", "fraud_risk_score_weighted"]:
    print(f"{col:<25} min={df[col].min():.2f}  max={df[col].max():.2f}  mean={df[col].mean():.2f}")

# Step 13: Metrics
print("\n📊 Evaluation Metrics:")
precision, recall, _ = precision_recall_curve(df["is_fraud"].astype(int), df["model_score"])
pr_auc = auc(recall, precision)
roc_auc = roc_auc_score(df["is_fraud"].astype(int), df["model_score"])
precision_at_threshold = float(df[df["selected_flag"] == 1]["is_fraud"].mean())
y_pred = df["selected_flag"]
tn, fp, fn, tp = confusion_matrix(df["is_fraud"].astype(int), y_pred).ravel()

# Step 13.5: Ranking Metrics and Precision@K
print("\n📌 Precision & Ranking Metrics:")

# Precision@1% of dataset
k = int(len(df) * 0.01)
top_k_indices = np.argsort(df["model_score"].values)[-k:][::-1]
precision_at_1pct = df.iloc[top_k_indices]["is_fraud"].mean()
top_k_threshold = df.iloc[top_k_indices]["model_score"].min()
print("\n🧠 Metric Interpretation:")
print("PR-AUC: Measures how well the model ranks frauds across all thresholds. Closer to 1 is better.")
print(f"PR-AUC: {pr_auc:.4f}")
print("ROC-AUC: Measures overall discrimination ability. 0.5 is random guessing.")
print(f"ROC-AUC: {roc_auc:.4f}")
print("Precision@1%: Of the top 1% scored transactions, this is the actual fraud rate.")
print(f"Precision@1%: {precision_at_1pct:.4f}")
print("Top-k threshold: Minimum score required to be in the top 1% — useful for alerting.")
print(f"Top-k threshold: {top_k_threshold:.4f}")

#Step 14: Show top 5 alerts
print("\n🚨 Top 5 flagged alerts:")
print(df[df["selected_flag"] == 1].sort_values("model_score", ascending=False).head(5)[[
    "transaction_id", "user_id", "amount", "merchant_category", "model_score", "fraud_risk_score_weighted"
]].to_string(index=False))

#Step 15: Save alerts
print("\n💾 Saving flagged alerts to alerts.csv...")
alerts = df[df["selected_flag"] == 1].copy()
alerts.to_csv("alerts.csv", index=False)
print("✅ Saved all flagged alerts to alerts.csv")
print(f"\n📦 Total alerts flagged: {len(alerts)}")
print(f"📌 Alert fraud rate: {alerts['is_fraud'].mean():.4f}")