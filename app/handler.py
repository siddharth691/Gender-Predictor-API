"""
Request Handlers
"""

import tornado.web
import logging
from tornado import concurrent
from tornado import gen
from concurrent.futures import ThreadPoolExecutor

from app.base_handler import BaseApiHandler
from app.settings import MAX_MODEL_THREAD_POOL
from ml_code.FeatureExtraction import FeatureExtraction
logger = logging.getLogger('app')

class IndexHandler(tornado.web.RequestHandler):
    """APP is live"""

    def get(self):
        self.write("This is my first App! App is Live!")

    def head(self):
        self.finish()


class GenderPredictionHandler(BaseApiHandler):

    
    _thread_pool = ThreadPoolExecutor(max_workers=MAX_MODEL_THREAD_POOL)

    def initialize(self, model, *args, **kwargs):
        self.model = model
        super().initialize(*args, **kwargs)

    @concurrent.run_on_executor(executor='_thread_pool')
    def _blocking_predict(self, X):
        results = []
        for featureArray in X:
            target_value = self.model.classify(featureArray)
            if (target_value == 'm'):
                target_value = 'male'
            elif(target_value == 'f'):
                target_value = 'female'
            results.append(target_value)
        return results


    @gen.coroutine
    def predict(self, data):
        if type(data) == dict:
            data = [data]
        fx = FeatureExtraction()
        X = []
        for item in data:
            record  = (fx._nameFeatures(item.get("first_name")))
            logger.info("Responding to name : %s", item.get("first_name"))
            X.append(record)

        results = yield self._blocking_predict(X)
        self.respond(results)
