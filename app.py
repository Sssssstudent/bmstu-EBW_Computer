from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def print_hellow():
    return render_template("index.html")


@app.route('/text/')
def print_text():
    return '<h1>Some text!</h1>'


app.run()
