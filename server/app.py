#  _         _
# | |       | |
# | |__   __| |___
# | '_ \ / _` / __|
# | |_) | (_| \__ \
# |_.__/ \__,_|___/ recommender-systems

from flask import Flask, request
import os
from utils.model import load_model
from argparse import ArgumentParser

current_dir = os.getcwd()
model_folder = "/".join(current_dir.split("/")
                        [:-1]) if "/server" in current_dir else current_dir
app = Flask(__name__)
path_to_model = f"{model_folder}/ml/models/model.surprisemodel"
model = load_model(path_to_model)

#########################################################
# TODO:
# 1. On the route "/", return a JSON instead of a string with the relevant data
# 2. On the route "/", add "iid" as a parameter that can be provided by the user
# 2. Anticipate some cases where your server should return some error codes:
#       - uid requested does not exist (404)
#       - prediction could not be carried out correctly (500)
#       - route requested does not exist (404)
#    Try to implement those error codes for both routes
# 3. Add additional returned metadata when querying "/metadata"
#########################################################


@app.route("/", methods=["GET"])
def get_recommendation():
    uid = request.args.get('uid', str(196))
    iid = str(302)
    # see: surprise.prediction_algorithms.algo_base.AlgoBase.predict
    pred = model.predict(uid, iid, r_ui=None)
    return f"{pred}"


@app.route("/metadata", methods=["GET"])
def get_metadata():
    # see: surprise.prediction_algorithms.matrix_factorization.SVD
    # pu: The user factors - numpy array of size (n_users, n_factors)
    return {
        "n_users": model.pu.shape[0],
        "n_factors": model.pu.shape[1]
    }