from pathlib import Path
import io
import pandas as pd
from flask import Flask, render_template_string, request, send_file, flash, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [ROOT / "dataset_phishing.csv", ROOT / "dataset" / "dataset_phishing.csv"]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)

if DATA_PATH is None:
    raise FileNotFoundError("dataset_phishing.csv not found. Keep it beside app.py in GitHub.")

df = pd.read_csv(DATA_PATH).drop_duplicates().copy()
if "status" not in df.columns:
    raise ValueError("Dataset must contain a 'status' column.")
FEATURE_COLUMNS = [c for c in df.columns if c not in ("url", "status")]
X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
y = (df["status"].astype(str).str.strip().str.lower() == "phishing").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
MODELS = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]),
    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
}
ACCURACIES = {}
for name, model in MODELS.items():
    model.fit(X_train, y_train)
    ACCURACIES[name] = accuracy_score(y_test, model.predict(X_test))

HTML = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phishing Website Detection</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Arial,sans-serif;background:#f3f6fb;color:#172033}
header{background:#101c36;color:white;padding:28px 18px;text-align:center}header h1{margin:0 0 8px;font-size:27px}
header p{margin:0;color:#cbd5e1}.wrap{max-width:900px;margin:28px auto;padding:0 16px}
.card{background:white;border-radius:14px;padding:24px;margin-bottom:18px;box-shadow:0 5px 22px #17203312}
h2{margin-top:0;color:#14264b}.btn{background:#2458d3;color:white;border:0;border-radius:8px;padding:12px 18px;font-weight:bold;cursor:pointer}
input,select{padding:11px;border:1px solid #cbd5e1;border-radius:8px;width:100%;margin:8px 0 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.metric{background:#eef3ff;padding:15px;border-radius:10px}.metric b{display:block;font-size:21px;color:#1d4ed8}
.notice{background:#fff7df;padding:14px;border-radius:9px;color:#72520a;line-height:1.5}
.result{padding:18px;border-radius:10px;font-size:20px;font-weight:bold}.phish{background:#fee2e2;color:#991b1b}.legit{background:#dcfce7;color:#166534}
small{color:#64748b}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:9px;border-bottom:1px solid #e2e8f0}
</style></head><body>
<header><h1>Phishing Website Detection</h1><p>Machine Learning based classification using four supervised algorithms</p></header>
<div class="wrap">
<div class="card"><h2>Project Overview</h2><div class="grid">
<div class="metric"><b>{{ records }}</b><small>Dataset records</small></div>
<div class="metric"><b>{{ features }}</b><small>Numeric input features</small></div>
<div class="metric"><b>4</b><small>ML algorithms</small></div>
<div class="metric"><b>{{ best }}</b><small>Best test accuracy</small></div>
</div></div>
<div class="card"><h2>Predict from feature CSV</h2>
<p>Upload a CSV containing the same 87 numeric feature columns used by this project. The app will add a prediction for each row.</p>
<form method="post" action="/predict" enctype="multipart/form-data">
<label><b>Select model</b></label><select name="model">{% for name in model_names %}<option>{{name}}</option>{% endfor %}</select>
<label><b>Upload feature CSV</b></label><input type="file" name="file" accept=".csv" required>
<button class="btn" type="submit">Predict CSV</button></form>
<div class="notice"><b>Important limitation:</b> This notebook excludes raw URL text from training. It does not include URL-to-feature extraction. Therefore this app cannot reliably classify a pasted URL directly. It requires the 87 numeric features in CSV format.</div>
</div>
<div class="card"><h2>Model Test Accuracy</h2><table><tr><th>Model</th><th>Accuracy</th></tr>{% for name, score in accuracies.items() %}<tr><td>{{name}}</td><td>{{"%.2f"|format(score*100)}}%</td></tr>{% endfor %}</table></div>
<div class="card"><h2>How to use</h2><ol><li>Prepare a CSV with the 87 numeric feature columns.</li><li>Choose one of the four models.</li><li>Upload the CSV and click Predict CSV.</li><li>Download the output CSV containing predictions.</li></ol><small>Academic demonstration only. Do not use predictions as the only security check.</small></div>
</div></body></html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML, records=len(df), features=len(FEATURE_COLUMNS),
        best=f"{max(ACCURACIES.values())*100:.2f}%", model_names=list(MODELS),
        accuracies=ACCURACIES)

@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("file")
    model_name = request.form.get("model", "")
    if not uploaded or not uploaded.filename.lower().endswith(".csv"):
        flash("Please upload a CSV file.")
        return redirect(url_for("home"))
    if model_name not in MODELS:
        flash("Please select a valid model.")
        return redirect(url_for("home"))
    try:
        incoming = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_COLUMNS if c not in incoming.columns]
        if missing:
            return ("CSV is missing required feature columns: " + ", ".join(missing[:12]) +
                    (" ..." if len(missing) > 12 else ""), 400)
        features = incoming[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        predictions = MODELS[model_name].predict(features)
        output = incoming.copy()
        output["prediction"] = ["Phishing" if int(v) == 1 else "Legitimate" for v in predictions]
        output["prediction_code"] = predictions
        buffer = io.BytesIO()
        output.to_csv(buffer, index=False)
        buffer.seek(0)
        return send_file(buffer, mimetype="text/csv", as_attachment=True,
                         download_name="phishing_predictions.csv")
    except Exception as exc:
        return ("Could not process CSV: " + str(exc), 400)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
