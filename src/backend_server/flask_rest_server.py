
from flask import Flask, render_template, flash, request, make_response, current_app, jsonify
from flask_cors import CORS
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from flask_restful import reqparse, abort, Api, Resource
from flask_caching import Cache
import numpy as np
import pickle  # ew

import keras
from keras.preprocessing import text, sequence
import tensorflow as tf

import copy
import re


# flask boilerplate
DEBUG = True
app = Flask(__name__)
cache = Cache(app, config={"CACHE_TYPE": "simple"})
CORS(app)  # allow cross-domain requests -- allow all by default
api = Api(app)
app.config.from_object(__name__)
app.config["SECRET_KEY"] = "anything"


# for form box queries
class ReusableForm(Form):
    comment = TextField("Comment:", validators=[validators.required()])


model_dir = "../data/models/"
model_names = ["bi_gru_raw_binary_glove.h5", "bi_gru_raw_binary_fasttext.h5"]
print("\n\n\t...Loading models from {:s}".format(model_dir))
models = []
for model_name in model_names:
    # load keras model
    models.append( keras.models.load_model(model_dir + model_name) )
    models[-1]._make_predict_function()
    print("\tLOADED KERAS MODEL: {:s}".format(model_name[:-3]))

# text tokenizer
maxlen = 100
with open(model_dir + "tokenizer_raw_binary.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)
print("\tLOADED TOKENIZER")
print("\tHappy filtering!\n\n")

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument("comment")

graph = tf.get_default_graph()  # for the sake of making model evaluations thread safe


@cache.memoize(timeout=300)  # enable caching this function with 5min memory
def get_prediction(comment):
    """
    Tokenizes comments using trained Keras tokenizer and then returns a toxicity score from trained Keras model.
        comment [String] : text to be scored
    """
    uq_vectorized = tokenizer.texts_to_sequences([comment])
    uq_vectorized = sequence.pad_sequences(uq_vectorized, maxlen=maxlen)
    predictions = []
    for model in models:
        with graph.as_default():
            predictions.append(model.predict(uq_vectorized).flatten()[0])
    prediction = np.mean(predictions)
    prediction = round(prediction.tolist(), 3)
    return prediction


@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}<\pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}<\pre>
    See logs for full stack trace.
    """.format(e), 500


# Route for POST requests:
#   Input is received as a byte-array so we decode into a UTF-8 string and send to the model for prediction.
#   Response must be a JSON object with a "score" field.
@app.route('/api', methods = ['POST'])
def api():
    jsdata = request.data.decode("utf-8")
    prediction = get_prediction(jsdata)
    return jsonify(score=prediction)


# Route for webform that allows user input queries and scores them.
#   This webform is specified in ./templates/toxic_webform.html
@app.route('/', methods=["GET", "POST"])
def predict_toxicity():
    form = ReusableForm(request.form)

    print(form.errors)
    if request.method == "POST":
        comment = request.form["comment"]
        print(comment)

        if form.validate():
            with graph.as_default():
                prediction = get_prediction(comment)
            flash("Toxicity percentage: {}%".format(100*prediction))
        else:
            flash("All the form fields are required.")

    return render_template("toxic_webform.html", form=form)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # use_reloader=False flag makes server load once instead of twice



