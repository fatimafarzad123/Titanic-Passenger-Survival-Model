"""
Flask web app for PythonAnywhere - serves BOTH trained models.
Student: Fatima Farzad (SP24-BSE-025) - AIC354 Assignment 2

  /          -> home page with links
  /titanic   -> Task 1: Titanic Passenger Survival Prediction
  /heart     -> Task 2: Heart Disease Prediction

Each page recreates the exact same LabelEncoders used in the notebooks
(fit on the same fixed category lists) and loads the saved SVC model.
"""

import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).parent
app = Flask(__name__)

SYSTEMS = {
    "titanic": {
        "title": "Titanic Passenger Survival Prediction",
        "icon": "🚢",
        "model": pickle.load(open(HERE / "titanic_svc_trained_model.pkl", "rb")),
        "fields": {  # column name -> (label, list of categories)
            "PClass": ("Passenger Class", ["First", "Second", "Third"]),
            "Gender": ("Gender", ["Male", "Female"]),
            "Sibling": ("Siblings/Spouses Aboard", ["Zero", "One", "Two", "Three"]),
            "Embarked": ("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"]),
        },
        "labels": {1: ("SURVIVED", "good"), 0: ("NOT SURVIVED", "bad")},
        "note": "Support Vector Classifier · 80% test accuracy (100-row Titanic sample)",
    },
    "heart": {
        "title": "Heart Disease Prediction",
        "icon": "❤️",
        "model": pickle.load(open(HERE / "heart_svc_trained_model.pkl", "rb")),
        "fields": {
            "ChestPain": ("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]),
            "ExerciseAngina": ("Exercise-Induced Angina", ["Yes", "No"]),
            "Vessels": ("Major Vessels Coloured by Fluoroscopy", ["Zero", "One", "Two", "Three"]),
            "Thal": ("Thallium Stress Test (Thal)", ["Normal", "Fixed Defect", "Reversible Defect"]),
        },
        "labels": {1: ("HEART DISEASE", "bad"), 0: ("NO HEART DISEASE", "good")},
        "note": "Support Vector Classifier · 75% test accuracy (UCI Cleveland, 297 patients) · not a medical tool",
    },
}
for s in SYSTEMS.values():
    s["encoders"] = {col: LabelEncoder().fit(cats) for col, (_, cats) in s["fields"].items()}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#f5f6f8;margin:0;color:#1d2330}
 .box{max-width:560px;margin:40px auto;background:#fff;border-radius:12px;padding:28px 32px;box-shadow:0 2px 10px #0001}
 h1{font-size:1.5rem;margin:0 0 6px} p.note{color:#667;font-size:.9rem;margin:0 0 20px}
 label{display:block;font-weight:600;margin:14px 0 6px} select{width:100%;padding:9px;border-radius:8px;border:1px solid #ccd;font-size:1rem}
 button{margin-top:22px;width:100%;padding:12px;border:0;border-radius:8px;background:#c0392b;color:#fff;font-size:1rem;font-weight:600;cursor:pointer}
 .res{margin-top:22px;padding:16px;border-radius:8px;text-align:center;font-size:1.3rem;font-weight:700}
 .good{background:#e6f6ea;color:#1e7b34} .bad{background:#fdecea;color:#a12a1f}
 a{color:#c0392b} footer{margin-top:22px;font-size:.8rem;color:#889}
</style></head><body><div class="box">
{% if system %}
 <h1>{{ system.icon }} {{ system.title }}</h1><p class="note">{{ system.note }}</p>
 <form method="post">
 {% for col, (label, cats) in system.fields.items() %}
  <label for="{{ col }}">{{ label }}</label>
  <select name="{{ col }}" id="{{ col }}">
  {% for c in cats %}<option {% if chosen.get(col) == c %}selected{% endif %}>{{ c }}</option>{% endfor %}
  </select>
 {% endfor %}
 <button type="submit">Predict</button></form>
 {% if result %}<div class="res {{ result[1] }}">{{ result[0] }}</div>{% endif %}
 <footer><a href="/">← Home</a></footer>
{% else %}
 <h1>AIC354 – Assignment 2</h1><p class="note">Fatima Farzad · SP24-BSE-025</p>
 <p>🚢 <a href="/titanic">Task 1: Titanic Passenger Survival Prediction</a></p>
 <p>❤️ <a href="/heart">Task 2: Heart Disease Prediction</a></p>
{% endif %}
<footer>Developed by Fatima Farzad (SP24-BSE-025) · Machine Learning Fundamentals</footer>
</div></body></html>"""


@app.route("/")
def home():
    return render_template_string(PAGE, title="AIC354 Assignment 2", system=None)


@app.route("/<name>", methods=["GET", "POST"])
def predict(name):
    system = SYSTEMS.get(name)
    if system is None:
        return "Not found", 404
    chosen, result = {}, None
    if request.method == "POST":
        chosen = {col: request.form.get(col) for col in system["fields"]}
        if all(chosen[col] in cats for col, (_, cats) in system["fields"].items()):
            encoded = pd.DataFrame({col: system["encoders"][col].transform([chosen[col]]) for col in system["fields"]})
            result = system["labels"][int(system["model"].predict(encoded)[0])]
    return render_template_string(PAGE, title=system["title"], system=system, chosen=chosen, result=result)


if __name__ == "__main__":
    app.run(debug=True)
