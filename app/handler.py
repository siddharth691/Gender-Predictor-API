"""
Request Handlers
"""

import tornado.web
import logging
from tornado import concurrent
from tornado import gen
from concurrent.futures import ThreadPoolExecutor

from app.base_handler import BaseApiHandler
from app.exceptions import AuthError
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


class ModelUpdateHandler(BaseApiHandler):
    _thread_pool = ThreadPoolExecutor(max_workers=MAX_MODEL_THREAD_POOL)
    
    @concurrent.run_on_executor(executor='_thread_pool')
    def _blocking_update(self, X):
	
        results = []
        true_pass = open('pass_file.txt', 'r').read()
        
        for nameGender in X:

            if(nameGender[0]!=true_pass):

				logger.info(" Rejecting update request because of authentication error")
        	
				raise AuthError
            
	    	else:

	        	with open('../ml_code/update_data.csv', 'a') as file:

			    file.write(nameGender[1] + ','+ nameGender[2]+'\n')
		    target_show = 'Successfully updated the update data'
			results.append(target_show)
        
        return results


    @gen.coroutine
    def update(self, data):
        if type(data) == dict:
            data = [data]
        X = []


        for item in data:
        	password = item.get("password")
            first_name  = item.get("first_name")
            true_gender = item.get("gender")
            X.append([password, first_name, true_gender])

        results = yield self._blocking_update(X)
        self.respond(results)
