# Logging
import logging
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cache', type=str, help="provide a local filepath to a folder to be used for caching data")
args = parser.parse_args()

##### ADD CACHE ARGUMENT PARSED THROUGH. CHECK IF FOLDER EXISTS. ELSE CREATE FOLDER

# Log File Config
logging.basicConfig(
     filename="CACHE/latest.log",
     filemode="w+",
     level=logging.INFO, 
     format= '[%(asctime)s] {%(filename)s:%(lineno)d} [%(levelname)s]: %(message)s',
     datefmt='%H:%M:%S',
 )

# Log Console Config
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

# Backend API imports
from data_fetch import *
from planner import *
from components.electrolyser import Electrolyser
from components.renewables import Generator
from nemglo_lite import *

from datetime import datetime
import json
# import validate as inv
from types import SimpleNamespace

# Flask/CORS imports
from flask import Flask, request, jsonify, abort
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)#, resources={r'/*': {'origins': ['https://www.nemglo.org/','http://localhost:3000','http://nemglo-backend.eba-rdn2wca8.ap-southeast-2.elasticbeanstalk.com/','https://main.d33u9p9lbzxx3x.amplifyapp.com']}})
app.config['CORS_HEADERS'] = 'Content-Type'

from flask import abort as fabort, make_response
def abort(status_code, message):
    response = make_response(f'{status_code}\n{message}')
    response.status_code = status_code
    fabort(response)


# API - Market Data inputs [preview]
@app.route('/', methods=['GET'])
def default():
    return "nemglo version beta 0.2"


@app.route('/api/', methods=['GET'])
def default_api():
    return "NEMGLO API"


# API - Market Data inputs [preview]
@app.route('/api/get-market-data', methods=['POST'])
def api_get_market_data():
    try:
        conf = request.json
        return get_market_data(conf)
    except:
        with open('CACHE/latest.log', 'r') as file:
            logfile = file.read()
        err_message = logfile.split('200 -')[-1:]
        abort(500, err_message[0])


# API - Generator Data inputs [preview]. ONLY FOR SINGLE GEN, CANNOT PASS BOTH
@app.route('/api/get-generator-data', methods=['POST'])
def api_get_generator_data():
    try:
        conf = request.json
        return get_generator_data(conf)
    except:
        with open('CACHE/latest.log', 'r') as file:
            logfile = file.read()
        err_message = logfile.split('200 -')[-1:]
        abort(500, err_message[0])


# API - Model Simulation [MAIN]
@app.route('/api/get-data', methods=['POST'])
def api_get_data():
    try:
        conf = request.json
        print(conf)
        return get_operation(conf)
    except:
        with open('CACHE/latest.log', 'r') as file:
            logfile = file.read()
        err_message = logfile.split('200 -')[-1:]
        abort(500, err_message[0])


if __name__ == "__main__":
    print("\n======================================================================\n"+\
          "Access NEMGLO-app (web interface) at: https://www.nemglo.org/simulator \n"+\
          "======================================================================\n")
    app.run(port=5000)
