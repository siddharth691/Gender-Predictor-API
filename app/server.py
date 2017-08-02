import os
import logging
import logging.config

import tornado.ioloop
import tornado.web
from tornado.options import options

from sklearn.externals import joblib

from app.settings import MODEL_DIR
from app.settings import MAX_UPDATE_MODEL_TIME
from app.handler import IndexHandler, GenderPredictionHandler, ModelUpdateHandler

from ml_code.update_model import update_model
import time

MODELS = {}


def load_model(pickle_filename):
    return joblib.load(pickle_filename)


def main():

    # Get the Port and Debug mode from command line options or default in settings.py
    options.parse_command_line()


    logger = logging.getLogger('app')
    # Load ML Models
    logger.info("Loading Gender Prediction Model...")
    MODELS["gender"] = load_model(os.path.join(MODEL_DIR, "model.pkl"))

    urls = [
        (r"/$", IndexHandler),
        (r"/api/gender/(?P<action>[a-zA-Z]+)?", GenderPredictionHandler,
            dict(model=MODELS["gender"]))
        (r"/api/gender/update/(?U<action>[a-zA-Z]+)?", ModelUpdateHandler)
    ]

    # Create Tornado application
    application = tornado.web.Application(
        urls,
        debug=options.debug,
        autoreload=options.debug) 

    #Creating a model update periodic callback
    logger.info("Updating the model at datetime: {}".format(time.time()))
    tornado.ioloop.PeriodicCallback(update_model(ml_code_loc='../ml_code/'), MAX_UPDATE_MODEL_TIME).start()

    # Start Server
    logger.info("Starting App on Port: {} with Debug Mode: {}".format(options.port, options.debug))
    application.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


