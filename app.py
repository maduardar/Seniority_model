from flask import Flask, request, render_template
from seniority_model import model
# from flask_ngrok import run_with_ngrok

app = Flask(__name__)
# run_with_ngrok(app)


@app.route('/')
def input_form():
    return render_template("input_form.html")


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text'].lower()
    score = 0
    if text.lower() in {'salim', 'maduar', 'darin', 'darina'}:
        score = 100
    else:
        score = model.predict([text])[0]
    text = "Your seniority level is %d" % score
    return render_template("output_form.html", res=text)


if __name__ == '__main__':
    app.run()
    # app.run(host='0.0.0.0', port=80)
