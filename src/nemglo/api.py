# Backend API imports
from .lite import *
from .defaults import *
# import validate as inv

# Flask/CORS imports
from flask import Flask, request, jsonify, abort
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# Return error log for API request
from flask import abort as fabort, make_response
def abort(status_code, message):
    response = make_response(f'{status_code}\n{message}')
    response.status_code = status_code
    fabort(response)


def throw_error_log():
    with open(LOG_FILEPATH, 'r') as file:
        logfile = file.read()
    err_message = logfile.split('200 -')[-1:]
    abort(500, err_message[0])


# API - Tests [preview]
@app.route('/', methods=['GET'])
def default():
    return "you have reached nemglo"


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
        # with open(LOG_FILEPATH, 'r') as file:
        #     logfile = file.read()
        # err_message = logfile.split('200 -')[-1:]
        # abort(500, err_message[0])
        throw_error_log()


# API - Generator Data inputs [preview]. ONLY FOR SINGLE GEN, CANNOT PASS BOTH
@app.route('/api/get-generator-data', methods=['POST'])
def api_get_generator_data():
    try:
        conf = request.json
        return get_generator_data(conf)
    except:
        # with open(LOG_FILEPATH, 'r') as file:
        #     logfile = file.read()
        # err_message = logfile.split('200 -')[-1:]
        # abort(500, err_message[0])
        throw_error_log()


# API - Model Simulation [MAIN]
@app.route('/api/get-data', methods=['POST'])
def api_get_data():
    try:
        conf = request.json
        print(conf)
        return get_operation(conf)
    except:
        # with open(LOG_FILEPATH, 'r') as file:
        #     logfile = file.read()
        # err_message = logfile.split('200 -')[-1:]
        # abort(500, err_message[0])
        throw_error_log()


def run(port=5000):
    print("\n======================================================================\n"+\
          "Access NEMGLO-app (web interface) at: https://www.nemglo.org/simulator \n"+\
          "======================================================================\n")
    app.run(port=port)
