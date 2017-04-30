import os
import logging
import logging.config

import tornado.ioloop
import tornado.web
from tornado.options import options

from sklearn.externals import joblib

from app.settings import MODEL_DIR
from app.handler import IndexHandler, GenderPredictionHandler


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
    ]

    # Create Tornado application
    application = tornado.web.Application(
        urls,
        debug=options.debug,
        autoreload=options.debug) 

    # Start Server
    logger.info("Starting App on Port: {} with Debug Mode: {}".format(options.port, options.debug))
    application.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


