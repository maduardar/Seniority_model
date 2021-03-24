from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def input_form():
    return render_template("input_form.html")


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text'].lower()
    text = "I guess, you are a " + text
    return render_template("output_form.html", role=text)


if __name__ == '__main__':
    app.run()
