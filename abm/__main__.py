""" Main
Simulate agent-based model variants.
Run this script to produce results for one or multiple model versions, using simulation parameters set in constants.py.
"""

# DEPENDENCIES
import logging
import sys
import abm
from abm.constants import TIMESTAMP, MAINMODELS, SIMODELS, SI_MODELS_RUN
import analysis


def visualize(model, is_supplementary, logger=logging.getLogger()):
    """
    Helper function to create visuals for a completed model.

    Parameters
    ----------
    model : model
        Model to visualize.
    is_supplementary : bool
        Whether model is exclusively for SI.
    logger

    Returns
    -------
    abm.model.Model
        Model with results object removed.
    """
    if not model.save_results:
        logger.warning("Results saving is turned off, will not generate any visuals.")
        return
    elif not model.completed:
        logger.error("Must complete all experiments before visualization")
        return
    else:
        logger.info("Generating visuals...")
        # visuals for discussion in the paper
        if model.name in MAINMODELS and model.variant == "baseline":
            # visuals for main models
            analysis.main(model, vis_baseline=True, vis_extensions=False, vis_si=True)
        elif model.variant == "extension":
            # visuals for extension models
            analysis.main(model, vis_baseline=False, vis_extensions=True, vis_si=False)
        elif is_supplementary:
            # visuals for SI models
            analysis.main(model, vis_baseline=False, vis_extensions=False, vis_si=True)

        # visuals for diagnostics: experiments results figures
        if model.experiments > 3:   # exclude model runs with too few experimental conditions
            analysis.baseline_diagnostics(model)
            if model.variant == "extension":
                analysis.extension_diagnostics(model)

        # visuals for diagnostics: simulations results figures (baseline and extensions only)
        if model.save_simulations and any(model.name == name for name in MAINMODELS.keys()):
            # visualize simulations
            analysis.visualize_simulations(model=model, logger=logger, clean=True)
        logger.info("Visuals for model '%s' completed, clearing results", model.name)
        del model.results
        logger.info("Generation of visuals completed")
        return model


def resume(models):
    """
    Resume model run for given list of models.
    The function does not support parallel processing.

    Note that the following conditions must be met for this to work:
        - Parameters in constants.py must be unchanged from initial run, as they are used to re-initialize the models.
        - Models must have previous experiments or the entire run stored in pickle format.

    Parameters
    ----------
    models : Dict
        Model name-variant pairs, e.g. {'baselinesiqagg': 'baseline'} (see constants.py).

    Returns
    -------
    None
    """
    # initialize logging
    logger = abm.init_logger()
    logger.info("Started PhD ABM resume run at %s", TIMESTAMP)

    # resume model runs
    logger.info("Begin resuming models: %s", ', '.join(name for name in models.keys()))
    failed = []
    for name, variant in models.items():
        # initialize model
        model = abm.init_models(models={name: variant}, logger=logger).pop()
        try:
            completed_model = abm.model.Model.load(model.path_data + '/' + model.variant + ".pkl")
            logger.info("Loaded model %s from file", completed_model.name)
        except(FileNotFoundError, FileExistsError, AttributeError):
            if not isinstance(model, abm.model.Model):
                logger.error("Failed to load a supplied model %s, please check the resume() docs. Moving on...", model)
                continue
            completed_model = model.resume(logger=logger)
        if completed_model is None:
            failed.append(model.name)
            logger.error("Failed to resume model %s, moving on...", model.name)
            del model
            continue
        logger.info("Simulations for model %s completed", completed_model.name)

        # visualize results
        if completed_model.name in MAINMODELS:
            visualize(completed_model, is_supplementary=False, logger=logger)
        else:
            visualize(completed_model, is_supplementary=True, logger=logger)
        del completed_model
    logger.info("PhD ABM resume run completed")


def main(sweep_start=None, sweep_end=None, supplementary_models=SI_MODELS_RUN):
    """
    Run PhD ABM.

    Models to be run are defined as MAINMODELS and SIMODELS in constants.py.
    MAINMODELS are always run, whereas SIMODELS are only run if SI_MODELS_RUN is set to True.

    Parallel processing:
    if both sweep_start and sweep_and are defined, parallel processing is assumed.
    Otherwise, sweep_start and sweep_end are automatically set to the first and last experimental condition.

    Parameters
    ----------
    sweep_start : int
        Optionally specify first experiment to run.
        Must be used with sweep_end.
    sweep_end : int
        Optionally specify last experiment to run.
        Must be used with sweep_start.
    supplementary_models : bool
        True if models for the SI should be included in run.
    Returns
    -------
    List<abm.model.Model>
        Swept models without results.
    """
    # initialize logging
    logger = abm.init_logger()
    logger.info("Started PhD ABM run at %s", TIMESTAMP)

    # initialize models
    models = MAINMODELS
    if supplementary_models:
        models.update(SIMODELS)
    models = abm.init_models(models=models, logger=logger)

    # run models
    logger.info("Running models: %s", ', '.join(m.variant for m in models))
    for modelcount, model in enumerate(models):
        logger.info("Running model %s", model.name)

        # parallel processing: determine number of sweeps
        if sweep_start is None or sweep_end is None:
            sweep_start = 1
            sweep_end = model.experiments
            process_parallel = False
            logger.info("Model %s: running all %d experiments in one process",
                        model.name, sweep_end - sweep_start + 1)
        else:
            process_parallel = True
            if sweep_end > model.experiments:
                logger.error("Model %s: given experiment number %d exceeds maximum experiments to sweep, "
                             "skipping model run", model.name, sweep_end)
                continue
            logger.info("Model %s parallel processing: running subset of experiments: %d through %d",
                        model.name, sweep_start, sweep_end)

        # run model variant
        experiments = model.run_experiments(start=sweep_start,
                                            end=sweep_end,
                                            parallel_process=process_parallel,
                                            logger=logger)
        logger.info("Model %s: simulations for %d experiments complete",
                    model.name, experiments)

        # parallel processing: post-process if last experiment was run
        if not model.experiments == sweep_end:
            logger.info("Model %s: waiting for final experiment to complete...", model.name)
        else:
            logger.info("Model %s parallel processing: final experiment complete, processing results...",
                        model.name)
            while process_parallel and not model.completed:
                missings = model.post_processing(logger=logger)
                if len(missings) > 0:
                    logger.warning("Model %s parallel processing: still waiting on results for experiments: %s",
                                   ', '.join([str(m) for m in missings]))
            else:
                logger.debug("Model %s parallel processing: no need to post-process results, running on a single CPU",
                             model.name)
        logger.info("Model %s (%d/%d), completion status: %r, reached end of experiment %d/%d",
                    model.name, modelcount + 1, len(models), model.completed,
                    sweep_end,
                    model.experiments)
    if all([model.completed for model in models]):
        logger.info("All model simulations completed")
    else:
        logger.debug("Completed run, but some models remain incomplete: %s"
                     "This is expected behavior in case of parallel runs.",
                     [model.name for model in models if not model.completed])

    # visualize results if all models were run
    for completed_model in models:
        if any(completed_model.name == name for name in MAINMODELS.keys()):
            visualize(completed_model, is_supplementary=False)
        else:
            visualize(completed_model, is_supplementary=True)
    return models


if __name__ == '__main__':
    """
    Boilerplate method for execution of the script.
    
    Expects two integers as parameters that define the range of experimental conditions which should be run.
    The first integer defines the first experimental condition to run, the second the last.
    If a model has fewer experimental conditions than indicated, the excess range is ignored.
    
    If no parameter is given, all experimental conditions are run.
    """
    if len(sys.argv) > 1:
        # parse arguments
        params = []
        for i in range(0, len(sys.argv)):
            try:
                params.append(int(sys.argv[i]))
            except ValueError:
                continue
        if len(params) != 2:
            raise ValueError("Expected two integers, got %s. Parameters passed to script must be two integers,"
                             " indicating first and last experiment to run", sys.argv)
        main(sweep_start=params[0], sweep_end=params[1])
    else:
        main()
