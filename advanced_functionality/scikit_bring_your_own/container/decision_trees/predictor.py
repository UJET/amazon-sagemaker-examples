# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback
import time

import flask

import pandas as pd

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

scoringService = None

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ScoringService(object):
    def __init__(self):
        self.model = self.get_model()
        self.encoders = self.get_encoders()

    def get_model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        print("Loading Model")
        if not hasattr(self, 'model'):
            self.model = self.getModelFromFile()
        return self.model

    def getModelFromFile(self):
        try:
            filePath = os.path.join(model_path, 'boosted-trees-model.pkl')
            with open(filePath, 'rb') as inp:
                print("Model filesize: " + str(os.path.getsize(filePath)))
                return pickle.load(inp)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
        except Exception as e:
            print("Unexpected exception: ", e)
        except:
            print("Unexpected error: ", sys.exc_info()[0])

    def get_encoders(self):
        """Get the model encoders for this instance, loading if not already loaded."""
        print("Loading Encoders")
        if not hasattr(self, 'encoders'):
            self.encoders = self.getEncodersFromFile()
        return self.encoders

    def getEncodersFromFile(self):
        try:
            filePath = os.path.join(model_path, 'boosted-trees-encoders.pkl')
            with open(filePath, 'rb') as inp:
                print("Encoders filesize: " + str(os.path.getsize(filePath)))
                return pickle.load(inp)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
        except Exception as e:
            print("Unexpected exception: ", e)
        except:
            print("Unexpected error:", sys.exc_info()[0])

    def predict(self, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        # Separate input vars from identifier
        iVars = input.iloc[:, 1:]

        # Apply encoders
        # series is a Pandas series
        def resetNovelValuesAndTransform(series, encoder):
            for i, value in series.iteritems():
                if value not in encoder.classes_:
                    series[i] = "<unknown>"
            return encoder.transform(series)

        # catch novel values
        iVars.iloc[:, 7] = resetNovelValuesAndTransform(iVars.iloc[:, 7], self.encoders['callType'])
        iVars.iloc[:, 8] = resetNovelValuesAndTransform(iVars.iloc[:, 8], self.encoders['language'])

        return self.model.predict(iVars)

# The flask app for serving predictions
app = flask.Flask(__name__)

def getScoringService():
    global scoringService
    if scoringService == None:
        scoringService = ScoringService()
    return scoringService

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    scoringService = getScoringService()

    health = scoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    scoringService = getScoringService()

    data = None
    start = time.time()

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO.StringIO(data)
        data = pd.read_csv(s, header=None)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = scoringService.predict(data)

    # Convert from numpy back to CSV
    out = StringIO.StringIO()
    pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    print("Finished in " + str(time.time() - start) + " seconds")

    return flask.Response(response=result, status=200, mimetype='text/csv')
