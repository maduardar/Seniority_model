from flask import Flask, request, render_template
from seniority_model import model

app = Flask(__name__)


@app.route('/')
def input_form():
    return render_template("input_form.html")


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text'].lower()
    score = 0.
    if text.lower() in {'salim', 'maduar', 'darin', 'darina'}:
        score = 10.
    else:
        score = model.predict([text])[0]
    text = "Your seniority level is %.2f" % score
    return render_template("output_form.html", res=text)


if __name__ == '__main__':
    app.run()
