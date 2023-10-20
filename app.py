from flask import Flask, render_template, request
import joblib


app = Flask(__name__,  static_folder='static')

model_loaded = joblib.load('ml_models/models/ebw.pkl')
scaler_loaded = joblib.load('ml_models/scalers/MinMaxScaler.pkl')


@app.route('/', methods=["get", "post"])
def predict():
    data = None
    if request.method == "POST":
        iw = float(request.form.get("iw"))
        i_f = float(request.form.get("if"))
        vw = float(request.form.get("vw"))
        fp = float(request.form.get("fp"))

        test_sample = scaler_loaded.transform([[iw, i_f, vw, fp]])

        prediction = model_loaded.predict(test_sample)

        weld_depth = prediction[0][0]
        weld_width = prediction[0][1]
        data = {
            'iw': iw,
            'if': i_f,
            'vw': vw,
            'fp': fp,
            'weld_depth': weld_depth,
            'weld_width': weld_width,
        }

    return render_template(
        "index.html",
        data=data
    )


if __name__ == '__main__':
    app.run(debug=False)
