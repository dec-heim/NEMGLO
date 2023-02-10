import logging
from .defaults import *
import os

# Make default cache location for logging, if not existing
if not os.path.exists("CACHE"):
    os.mkdir("CACHE")

# Log File Config
logging.basicConfig(
     filename=LOG_FILEPATH,
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

# Disable other package loggers (min Warning level)
logging.getLogger("mip").setLevel(logging.WARNING)

# Commence Logging
logger = logging.getLogger(__name__)
logger.info(f"Commence Logging. Saved to file: {LOG_FILEPATH}")

# Must define logging config before remaining package imports
from nemglo.lite import *
from nemglo.api import * 
from nemglo.planning import data_fetch, planner
from nemglo.components import electrolyser, emissions, renewables
from nemglo.defaults import *

from importlib.metadata import version
try:
    __version__ = version("nemglo")
    logging.info("Running NEMGLO version: {}.".format(__version__))
except:
    pass


