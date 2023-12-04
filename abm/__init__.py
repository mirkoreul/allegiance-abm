""" Init
Entry-point for the abm package, sets up a logger and system settings.
"""

# DEPENDENCIES
from abm.constants import *
import abm.model
import random
import numpy

# set random seed
random.seed(RANDOM_SEED)
numpy.random.seed(RANDOM_SEED)

def init_logger(logfile=LOGFILE, path=PATH_LOGS, loglevel=LOGLEVEL):
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)
    logger.propagate = False    # turns off PyCharm's internal logger
    logformatter = logging.Formatter(("%(levelname)s in %(filename)s:%(funcName)s "
                                      "at line %(lineno)d "
                                      # "occured at %(asctime)s"
                                      "\n\t%(message)s"
                                      ))
    logger.debug("Initialized logging at level %s", logger.getEffectiveLevel())
    if logfile:
        filehandler = logging.FileHandler(path+'/log_'+str(datetime.datetime.now())+'.log')
        filehandler.setLevel(loglevel)
        filehandler.setFormatter(logformatter)
        logger.addHandler(filehandler)
        logger.debug("Logging to file: %s", logfile)
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(loglevel)
    streamhandler.setFormatter(logformatter)
    logger.addHandler(streamhandler)

    return logger


def init_models(models, logger):
    """
    Initialize models.

    Parameters
    ----------
    models : Dict<str, str>
        Key-value pairs respectively indicating name and variant of models to initialize.
        Keys correspond to directory names for results storage.
        Model names (keys) must correspond to variable names defined in constants.py.
    logger: logging.Logger

    Returns
    -------
    List<abm.Model>
        Initialized models.
    """
    models_init = []
    for name, variant in models.items():
        models_init.append(
            abm.model.Model(
                variant=variant,
                name=name,
                variant_dir=name,
                path_visuals=PATH_VISUALS,
                path_data=PATH_DATA,
                params_fix=globals()[name.upper()]['PARAMS_FIX'],
                params_sweep=globals()[name.upper()]['PARAMS_SWEEP'],
                save_results=SAVE_RESULTS,
                save_simulations=SAVE_SIMULATIONS,
                vars_track=VARS_TRACK,
                vars_target=VARS_TARGET,
                vars_vis=VARS_VIS,
                logger=logger
            )
        )
    return models_init
