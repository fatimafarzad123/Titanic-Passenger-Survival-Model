# AIC354 Assignment 2 – Developing an End-to-End Machine Learning System

**Student:** Fatima Farzad · **Reg No:** SP24-BSE-025
**Course:** Machine Learning Fundamentals (AIC354) · **Instructor:** Dr. Rao Muhammad Adeel Nawab

## Live Demos

| Task | System | Live Link |
|---|---|---|
| Task 1 | Titanic Passenger Survival Prediction | PASTE_TITANIC_DEPLOYMENT_LINK_HERE |
| Task 2 | Heart Disease Prediction | PASTE_HEART_DEPLOYMENT_LINK_HERE |

## Repository Contents

```
Task1-Titanic-Survival-Prediction/   Instructor's Titanic notebook (run end-to-end), data, trained model
Task2-Heart-Disease-Prediction/      Heart Disease notebook (same steps as Titanic), UCI data, trained model
streamlit-apps/                      Web apps for Streamlit Community Cloud (titanic_app.py, heart_app.py)
pythonanywhere-app/                  Flask app for PythonAnywhere (serves both models)
```

## Task 2 – Heart Disease Prediction System

- **Dataset:** UCI Heart Disease, Cleveland data (https://archive.ics.uci.edu/dataset/45/heart+disease) – 303 patients, 297 after removing rows with missing values.
- **4 selected input attributes:** Chest Pain Type (`cp`), Exercise-Induced Angina (`exang`), Number of Major Vessels (`ca`), Thallium Stress Test (`thal`).
- **Output:** Heart Disease – Yes / No.
- **Model:** Label Encoding + Support Vector Classifier (same as the Titanic system), 80/20 train/test split.
- **Result:** 75% accuracy on the test data (precision 0.90 / recall 0.58 for the Heart Disease class).

## Running the Notebooks Locally

```
cd Task2-Heart-Disease-Prediction      (or Task1-Titanic-Survival-Prediction)
python -m venv venv
venv\Scripts\activate                  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
jupyter notebook
```

## Deployment Option A – Streamlit Community Cloud (recommended, free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub.**
3. Repository: `fatimafarzad123/Titanic-Passenger-Survival-Model`, Branch: `main`, Main file path: `streamlit-apps/heart_app.py`
4. Open **Advanced settings**, choose Python **3.12**, then **Deploy**.
5. Repeat with Main file path `streamlit-apps/titanic_app.py` for the Titanic app.

## Deployment Option B – PythonAnywhere (free)

1. Sign up at https://www.pythonanywhere.com (Beginner account).
2. **Files** tab → create folder `/home/<username>/mlapp` and upload everything inside `pythonanywhere-app/`.
3. **Consoles** tab → open a Bash console and run:
   ```
   mkvirtualenv mlenv --python=python3.12
   pip install -r ~/mlapp/requirements.txt
   ```
4. **Web** tab → **Add a new web app** → **Manual configuration** → Python 3.12.
5. On the Web tab set: **Source code** = `/home/<username>/mlapp`, **Virtualenv** = `mlenv`.
6. Open the **WSGI configuration file** link, delete everything and paste:
   ```python
   import sys
   sys.path.insert(0, "/home/<username>/mlapp")
   from flask_app import app as application
   ```
7. Click **Reload**. Your links are `https://<username>.pythonanywhere.com/titanic` and `https://<username>.pythonanywhere.com/heart`.

## Note on Retraining

The `.pkl` models were trained with scikit-learn 1.9.1. If you re-run a notebook and get a new `svc_trained_model.pkl`, copy it into `streamlit-apps/` (and `pythonanywhere-app/`) with the matching name (`heart_svc_trained_model.pkl` or `titanic_svc_trained_model.pkl`), and set the `scikit-learn==` version in the app's `requirements.txt` to the version you used (`pip show scikit-learn`).

## Credits

Titanic notebook and Streamlit demo app by Mr. Mohsin Afzal and Dr. Rao Muhammad Adeel Nawab ([ml-teaching-notebooks](https://github.com/haqnawaz99/ml-teaching-notebooks)). The Heart Disease system follows the same steps.
