from flask import Flask, render_template, request
from main import predict_stress

# Tell Flask to look for templates in current folder
app = Flask(__name__, template_folder='.')

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        sleep = float(request.form["sleep_hours"])
        hr = float(request.form["heart_rate"])
        work = float(request.form["work_stress"])
        result = predict_stress(sleep, hr, work)
    return render_template("index.html", result=result)

# Only use this for local testing
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
