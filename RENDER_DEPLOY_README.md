# Phishing Website Detection — Flask deployment

This deployment uses Flask (not Streamlit) and can be hosted on Render.

## Important project limitation
The original notebook trains on 87 numeric website features and explicitly excludes the raw `url` column. It does not implement URL-to-feature extraction. Therefore this website accepts a CSV containing those 87 feature columns; it does **not** claim to scan a pasted URL.

## Run locally
```bash
python -m pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Deploy on Render
1. Push `app.py` and the updated `requirements.txt` to the root of your existing GitHub repository.
2. Keep `dataset_phishing.csv` in the repository root (as it currently is).
3. Sign in at https://render.com/ using GitHub.
4. Choose **New +** → **Web Service** and connect `Phishing-Website-Detection`.
5. Use these settings:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
   - Instance type: Free (if available)
6. Click **Create Web Service**. When deploy completes, Render displays the public `onrender.com` URL.

## GitHub project URL
https://github.com/ku2507u0731-create/Phishing-Website-Detection
