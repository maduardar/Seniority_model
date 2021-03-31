from flask import Flask, request, render_template
from seniority_model import model

app = Flask(__name__)


@app.route('/')
def input_form():
    return render_template("input_form.html")


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text'].lower()
    text = "Your seniority level is %.2f" % model.predict([text])[0]
    return render_template("output_form.html", res=text)


if __name__ == '__main__':
    app.run()
