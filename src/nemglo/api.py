# Backend API imports
from nemglo.lite import *
from nemglo.defaults import *
# import validate as inv
import argparse
import sys

# Flask/CORS imports
from flask import Flask, request, jsonify, abort
from flask_cors import CORS, cross_origin

# Flask
if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    app = Flask(__name__)

# Flask cors
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
        throw_error_log()


# API - Generator Data inputs [preview]. ONLY FOR SINGLE GEN, CANNOT PASS BOTH
@app.route('/api/get-generator-data', methods=['POST'])
def api_get_generator_data():
    try:
        conf = request.json
        return get_generator_data(conf)
    except:
        throw_error_log()


# API - Model Simulation [MAIN]
@app.route('/api/get-data', methods=['POST'])
def api_get_data():
    try:
        conf = request.json
        print(conf)
        return get_operation(conf)
    except:
        throw_error_log()


def run(port=5000):
    print("\n======================================================================\n"+\
          "Access NEMGLO-app (user-interface) see: https://www.nemglo.org/start \n"+\
          "======================================================================\n")
    app.run(port=port)

if __name__=='__main__':
        
    # Check if cache folder provided and valid filepath
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', '-c', type=str, \
        help="provide a local filepath to a folder to be used for caching data")
    args = parser.parse_args()

    # Determine cache folder
    if (args.cache is None):
        logging.info("Default data cache location is: {}.".format(DATA_CACHE.FILEPATH))
    elif (not os.path.exists(args.cache)):
        logging.info("Default data cache location is: {}.".format(DATA_CACHE.FILEPATH))
    else:
        DATA_CACHE.update_path(args.cache)
        logging.info("Updated preffered data cache location to: {}."+ \
            "Note, log files will save to default cache.".format(DATA_CACHE.FILEPATH))

    run()