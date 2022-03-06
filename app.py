from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from my_LR import lin_reg
import shutil
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    x_test = request.form['content']

    lr_obj = lin_reg(x_test=x_test)

    lr_obj.input_parser()
    # std_x = lr_obj.input_std_scalar()
    lr_obj.perform()
    return render_template("review.html", score=[lr_obj.r_sq, lr_obj.adj_r_sq, lr_obj.y_pred])

@app.route('/report', methods=['GET', 'POST'])
@cross_origin()
def report():
    cwd = os.getcwd()
    source = cwd + r"\profile.html"
    dest = cwd + r"\templates\profile.html"
    shutil.move(source, dest)
    return render_template("profile.html")

if(__name__== '__main__'):
    app.run(debug=True)