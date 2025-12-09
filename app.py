from flask import Flask, render_template, request
from main import predict_stress

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        sleep = float(request.form["sleep_hours"])
        hr = float(request.form["heart_rate"])
        work = float(request.form["work_stress"])
        result = predict_stress(sleep, hr, work)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)  