""" Model
Topmost logic to run a model variant.
To use, initialize an instance of model, and execute the run() method.
"""

# DEPENDENCIES
import os
import logging
import statistics
import xlsxwriter
import pickle
import pandas
import importlib
from time import sleep
from abm.constants import TIMESTAMP
from abm.simulation import Simulation
import abm.assistant.tracker as tracker


class Model:
    """
    Model variant that simulates experiments, visualizes and stores results.

    Instances of Model refer to a set of parameter combinations and mechanisms
    that are used consistently across experiments.

    Attribute
    ----------
    variant : str
        Name of the model variant.
    path_visuals : str
        Path to directory where visualizations are stored.
        A directory with the name of the model variant is automatically created.
    path_data : str
        Path to directory where raw model results are stored.
    params_fix : Dict
        Key-value pairs for fixed simulation parameters.
    params_sweep : Dict
        Key-value pairs for variable simulation parameters.
        Used to determine experimental conditions.
    save_results : bool
        Whether results should be stored on hard disk.
    vars_vis : Dict
        Complex dictionary used to define how results should be visualized.
        Refer to the documentation in the 'abm.constants' module for details.
    vars_target : List<str>
        Keys of the outcomes that should be compared across experiments.
    variant_dir : str
        Optional directory name that should be used to store results, or None if 'variant' should be used.
        This is useful run the same model variant with different parameters and store the results separately.
    logger : logging.Logger
    """
    def __init__(self, variant, path_visuals, path_data, params_fix, params_sweep,
                 save_results, save_simulations, vars_track, vars_vis, vars_target, variant_dir=None, name=None,
                 logger=logging.getLogger()):
        self.variant = variant
        if "_" in name:
            raise ValueError("Model names must not include special character '_' (reserved for tmp storage)")
        self.name = name
        self.variant_dir = variant if variant_dir is None else variant_dir
        self.timestamp = TIMESTAMP
        self.save_results = save_results
        self.save_simulations = save_simulations
        self.vars_track = vars_track
        self.vars_vis = vars_vis
        self.vars_target = vars_target
        self.results = []
        self.completed = False
        logger.info("Initializing model %s (variant %s)", self.name, self.variant.upper())

        # setup results directories
        if self.save_results:
            for outdir in [path_visuals + '/' + self.variant_dir,
                           path_visuals + '/' + self.variant_dir + '/simulations',
                           path_data + '/' + self.variant_dir]:
                try:
                    os.mkdir(outdir)
                    logger.debug("Output directory created: %s", outdir)
                except FileExistsError:
                    logger.warning("Results directory for variant '%s' already exists: %s", self.name, outdir)
        self.path_visuals = path_visuals + '/' + self.variant_dir
        self.path_visuals_file = self.path_visuals + "/simulations/results_" + self.variant_dir + "_" + TIMESTAMP
        self.path_data = path_data + '/' + self.variant_dir

        # parameters
        def sweep(parameters):
            """
            Generate list with all possible combinations of parameter values for sweeping.

            Parameters
            ----------
            parameters : Dict
                Values must be lists with possible parameter values, keys must be parameter names of type str.

            Returns
            -------
            List<List>
                List of lists: first list with variable names, and other lists containing parameter combinations.
                Parameter values in each combination are in order of names from first list.
            """
            variables = sorted(parameters.items())
            values = [v[1] for v in variables]
            sweeper = [[]]
            for val in values:
                temp = []
                for v in val:
                    for i in sweeper:
                        temp.append(i + [v])
                sweeper = temp
            sweeper.insert(0, [k[0] for k in variables])
            return sweeper

        self.params_fix = params_fix
        logger.info("Fixed parameters %s values: %s",
                    self.params_fix.keys(), self.params_fix.values())
        if params_sweep is not None:
            self.sweeper = sweep(params_sweep)
            self.experiments = len(self.sweeper) - 1
            logger.info("Model sweep with parameters %s, total combinations %d: %s",
                        ' '.join(self.sweeper[0]), self.experiments, self.sweeper[1:])
        else:
            self.sweeper = None
            self.experiments = 1
            logger.info("No parameter combinations to sweep")

    def run_experiments(self, start, end, parallel_process=False, logger=logging.getLogger()):
        """
        Run experiments for model variant and optionally stores results on disk.

        Parameters
        ----------
        start: int
            First experiment to sweep.
        end : int
            Last experiment to sweep.
        parallel_process : bool
            True if this is a parallel processing run.
        logger : logging.Logger

        Returns
        -------
        int
            Number of experiments swept.
        """
        if end > self.experiments:
            raise ValueError("Given last experiment to sweep exceeds number of experimental conditions")
        if start > end:
            raise ValueError("First experiment to sweep must be smaller than last")
        experiments = end - start + 1
        logger.info("Running %d experiments for model '%s' (variant: %s)...",
                    experiments, self.name, self.variant)
        for exp in range(start, end + 1):
            self.simulate(exp=exp, logger=logger)
            if parallel_process:
                pathexp = self.store_experiment(exp=exp, logger=logger)
                logger.info("Parallel processing mode: results for experiment %d temporarily stored at: %s",
                            exp, pathexp)
            else:
                logger.debug("Not using parallel processing, moving to next experiment without temporary storage")
        if not parallel_process:
            # storage
            if self.save_results:
                self.store(logger=logger)
            self.completed = True
        return experiments

    def post_processing(self, waittime=3, timeout=10, logger=logging.getLogger()):
        """
        Combine results from separately run experiments into one results object.

        This method is used for parallel processing only.
        Optionally stores model instance on disk if results saving is turned on in constants.py (recommended).
        Should be run once all experimental conditions have been simulated.
        If the 'self.results' object does not hold results for all experimental conditions expected by this instance,
        they are automatically loaded from Pickle files in the base storage directory.

        Parameters
        ----------
        waittime : int
            Time in seconds to wait between failed attempts to retrieve results for an experimental condition from disk.
        timeout : int
            Number of attempts to retrieve missing results before aborting post-processing.
        logger : logging.Logger

        Returns
        -------
        List<int>
            List of missing experiments that prevented post-processing, or empty list if post-processing succeeded.
        """
        logger.info("Begin post-processing for model %s (variant %s)", self.name, self.variant)

        # wait for experiments to complete and load complete results
        missings = None
        while len(self.results) < self.experiments and timeout >= 0:
            logger.info("Post-processing: current model instance holds results for %d/%d experiments,"
                        " loading results from disk...", len(self.results), self.experiments)
            self.results = []
            missings = []

            for exp in range(1, self.experiments + 1):
                try:
                    logger.info("Post-processing: Retrieving results on experiment %d from disk", exp)
                    self.results.append(self.load_experiment(exp=exp))
                except FileNotFoundError:
                    logger.warning("Post-processing: Results for experiment %d not found, retrying in %d seconds",
                                   exp, waittime)
                    timeout -= 1
                    missings.append(exp)
                    sleep(waittime)
                    continue
        if len(self.results) == self.experiments:
            logger.info("Post-processing completed")
            # cleanup results files for individual experiments
            try:
                logger.debug("Removing auxiliary files for individual experiments...")
                for exp in range(1, self.experiments + 1):
                    os.remove(self.path_data + '/' + self.variant + "_" + str(exp) + ".pkl")
            except FileNotFoundError:
                logger.debug("No auxiliary files to delete, moving on")
            # storage
            self.completed = True
            if self.save_results:
                self.store(logger=logger)
        else:
            logger.warning("Post-processing failed, could not retrieve experiments: %s", str(missings))
        return missings

    def resume(self, logger=logging.getLogger()):
        """
        Resume a previous model run.

        Requires that a valid storage path is defined in 'constants.py'.
        Either returns a previously completed model run, or loads single experiments that were temporarily stored.
        If some but not all experiments a found, the method runs and temporarily stores the missing experiments.

        Parameters
        ----------
        logger

        Returns
        -------
        abm.model.Model
            Completed model or None if no data from previous model runs was found.
        """
        missings = self.post_processing(waittime=0, timeout=0)
        if len(missings) == self.experiments:
            logger.error("Model %s: could not find valid file to resume, aborting resume", self.name)
            return
        while len(missings) > 0:
            logger.info("Model %s: running missing experiments...")
            exp = missings.pop()
            if exp is not None:
                self.run_experiments(start=exp, end=exp, parallel_process=True, logger=logger)
        self.post_processing(waittime=0, timeout=0)
        if self.completed:
            logger.info("Model %s: restored", self.name)
            return self
        else:
            logger.error("Failed to resume model through post-processing of experiment results, retrying...")
            self.resume(logger=logger)

    def simulate(self, exp, logger=logging.getLogger()):
        """
        Simulate experiment for model variant.

        The simulation results are appended to the results attribute of the model instance.

        Parameters
        ----------
        exp : int
            Identifier of the experiment, which is used to determine simulation parameters.
        logger : logging.Logger

        Returns
        -------
        None
        """
        logger.info("Running experiment %d for model %s (variant %s)...", exp, self.name, self.variant)

        # get parameter values for experiment
        if self.sweeper is not None:
            params_sweep = dict.fromkeys(self.sweeper[0])
            for name in self.sweeper[0]:
                params_sweep[name] = self.sweeper[exp][self.sweeper[0].index(name)]
                logger.info("Experiment %d seeded with %s = %s", exp, name, params_sweep[name])
        else:
            params_sweep = {}
            logger.info("Experiment 1 seeded with fixed parameters: ", self.params_fix)

        # run simulations
        sims = []
        for sim in range(1, self.params_fix['S'] + 1):
            logger.debug("Starting simulation %d/%d at: %s", sim, self.params_fix['S'], TIMESTAMP)
            if self.variant.lower() != "baseline":
                try:
                    module_simulation = importlib.import_module('abm.variants.simulation_' + self.variant.lower())
                    class_simulation = getattr(module_simulation, 'Simulation' + self.variant.title())
                    s = class_simulation(logger=logger,
                                         model=self.variant,
                                         experiment=exp,
                                         identifier=sim,
                                         vars_track=self.vars_track,
                                         params={**self.params_fix, **params_sweep})
                    logger.debug("Initialized simulation for variant %s using class %s",
                                 s.model, class_simulation)
                    sims.append(s.run(logger, self.vars_track))
                except ModuleNotFoundError as e:
                    logger.error("No module for model variant %s found. "
                                 "Please make sure that the model variant name, in lowercase, corresponds to "
                                 "a simulation variants module, i.e.: 'abm.variants.simulation_%s'. Original error: %s",
                                 self.variant, self.variant.lower(), e)
            else:
                s = Simulation(logger=logger,
                               model=self.variant,
                               experiment=exp,
                               identifier=sim,
                               vars_track=self.vars_track,
                               params={**self.params_fix, **params_sweep})
                sims.append(s.run(logger, self.vars_track))
            for var in self.vars_target:
                logger.debug("Simulation %d outcome for %s: %f (in percent)",
                             sim, var, round(sims[-1][var].percent(), 2))

        # store experiment results
        for var in self.vars_target:
            logger.info("Experiment %d/%d mean outcome for %s: %f (in percent)",
                        exp,
                        self.experiments,
                        var,
                        round(statistics.mean([s[var].percent() for s in sims]), 2))
        self.results.append(sims)

    def store(self, save_pckl=True, logger=logging.getLogger()):
        """
        Store model results on disk.

        Aggregates all tracked model results by simulation using several hardcoded functions,
        and stores them as Excel and LaTex files.
        The entire model object is stored in serialized (Pickle) format if save_pckl is set to True.

        Parameters
        ----------
        save_pckl : bool
            If True, store entire model object in Pickle format.
        logger : logging.Logger

        Returns
        -------
        None
        """
        logger.info("Saving results for model %s (variant %s)...", self.name, self.variant)

        # store Pickle
        if save_pckl:
            logger.debug("Storing copy of model as Pickle in: %s", self.path_data + self.variant + ".pkl")
            with open(self.path_data + '/' + self.variant + ".pkl", 'wb') as output:
                pickle.dump(self, output, -1)

        # store Excel
        logger.debug("Storing reduced model results as XLSX file in: %s",
                     self.path_data + '/' + self.variant + ".xlsx")
        results = xlsxwriter.Workbook(self.path_data + '/' + self.variant + ".xlsx")
        sheet = results.add_worksheet()
        sheet.write(0, 0, "experiment")
        sheet.write(0, 1, "simulation")
        column = 2
        params = list(key for key in self.sweeper[0])
        for param in [p if not isinstance(p, list) else ''.join(p) for p in params]:
            sheet.write(0, column, param)
            column += 1
        outcomes = list(key for key in self.results[0][0].keys() if
                        self.vars_track[key] and type(self.results[0][0][key]) is tracker.Tracker)
        for var in outcomes:
            sheet.write(0, column, var + '_mean')
            sheet.write(0, column + 1, var + '_min')
            sheet.write(0, column + 2, var + '_max')
            sheet.write(0, column + 3, var + '_first')
            sheet.write(0, column + 4, var + '_second')
            sheet.write(0, column + 5, var + '_last')
            sheet.write(0, column + 6, var + '_sum')
            column = column + 7
        row = 0
        for exp in range(0, self.experiments):
            for sim in range(0, self.params_fix['S']):
                row += 1
                column = 2
                for var, _ in enumerate(self.sweeper[0]):
                    try:
                        sheet.write(row, column, self.sweeper[exp + 1][var])
                    except TypeError:
                        sheet.write(row, column, ', '.join(str(self.sweeper[exp + 1][var])))
                    column += 1
                for var in outcomes:
                    logger.debug("Saving results for variable %s (experiment %d, simulation %d)",
                                 var, exp + 1, sim + 1)
                    sheet.write(row, 0, exp + 1)
                    sheet.write(row, 1, sim + 1)
                    sheet.write(row, column, self.results[exp][sim][var].mean(forceval=True))
                    sheet.write(row, column + 1, self.results[exp][sim][var].min())
                    sheet.write(row, column + 2, self.results[exp][sim][var].max())
                    sheet.write(row, column + 3, self.results[exp][sim][var].first())
                    sheet.write(row, column + 4, self.results[exp][sim][var].second())
                    sheet.write(row, column + 5, self.results[exp][sim][var].last())
                    sheet.write(row, column + 6, self.results[exp][sim][var].sum())
                    column = column + 7
        results.close()

        # store LaTex
        headcols = '& '
        if self.sweeper is not None:
            headcols = headcols + ' & '.join(var for var in self.sweeper[0])
        headcols = headcols + ' & ' + ' & '.join([var for var in self.vars_target]) + r'\\'

        head1 = r'''
        \begin{longtable}{c c c c}
        \caption{Results}\label{tab:results}\\
        \toprule
        '''
        head2 = r'''
        \midrule
        \endfirsthead
        \caption{Results}\\
        \toprule
        '''
        head3 = r'''
        \midrule    	
        \endhead
        '''
        rows = ''
        for e, exp in enumerate(self.results):
            res = []
            for var in self.vars_target:
                res.append(round(statistics.mean([s[var].mean(forceval=True) for s in exp]), 2))
            if self.sweeper is not None:
                cols = ' & '.join(str(self.sweeper[e + 1][self.sweeper[0].index(var)]) for var in self.sweeper[0])
            else:
                cols = ''
            rows += r'\textit{Exp. ' + str(e + 1) + '} & ' \
                    + cols + ' & ' + ' & '.join(str(r) for r in res) + r'\\ \midrule' + '\n'
        bottom = r'''
        \bottomrule
        \caption*{\footnotesize\textit{Note:} NOTE.}\\
        \end{longtable}
        '''
        logger.info("Writing results into LaTex table at: %s",
                    self.path_data + '/' + self.variant + '.tex')
        with open(self.path_data + '/' + self.variant + '.tex', 'w') as f:
            f.write(head1 + headcols + head2 + headcols + head3 + rows + bottom)

        # store LaTex including simulations
        rows = ''
        for e, exp in enumerate(self.results):
            for s in range(0, self.params_fix['S']):
                res = []
                for var in self.vars_target:
                    res.append(round(exp[s][var].mean(forceval=True), 2))
                if self.sweeper is not None:
                    cols = ' & '.join(str(self.sweeper[e + 1][self.sweeper[0].index(var)])
                                      for var in self.sweeper[0])
                else:
                    cols = ''
                rows += r'\textit{Exp. ' + str(e + 1) + r'} & \textit{Sim. ' + str(s + 1) + '} & ' \
                        + cols + ' & ' + ' & '.join(str(r) for r in res) + r'\\ \midrule' + '\n'
        logger.info("Writing results by simulation into LaTex table at: %s",
                    self.path_data + '/' + self.variant + '.tex')
        with open(self.path_data + '/' + self.variant + '_simulations.tex', 'w') as f:
            f.write(head1 + headcols + head2 + headcols + head3 + rows + bottom)

    def store_experiment(self, exp, logger=logging.getLogger()):
        """
        Store results from the latest experiment that was run in Pickle format.

        The experiment results are extracted from the 'self.results' list.
        Storage base path is fixed based on storage directory and model variant.

        Parameters
        ----------
        exp : int
            Number of the last experiment, used as an identifier in the file name.
        logger : logging.Logger

        Returns
        -------
        str
            Relative path to the Pickle file where the experiment was stored.
        """
        path = self.path_data + '/' + self.variant + "_" + str(exp) + ".pkl"
        with open(path, 'wb') as output:
            logger.debug("Saving results for experiment %d to path: %s", exp, path)
            pickle.dump(self.results[-1], output, -1)
        return path

    @staticmethod
    def load(path='storage/baseline/baseline.pkl'):
        """
        Load model variant results from Pickle file.

        Parameters
        ----------
        path : str
            Path to the pickle file (.pkl).

        Returns
        -------
        abm.model.Model or None if loading failed.
        """
        try:
            file = open(path, 'rb')
            try:
                model = pickle.load(file)
                file.close()
            except (pickle.UnpicklingError, EOFError) as _:
                return
        except FileNotFoundError:
            return
        return model

    def load_experiment(self, exp):
        """
        Load experiment results from Pickle file.

        Retrieves a 'self.results' dictionary, as previously stored with 'store_experiment'.
        Base path is fixed based on storage directory and model variant.

        Parameters
        ----------
        exp : int
            Experiment to retrieve.

        Returns
        -------
        Dict<List>
            Results dictionary.
        """
        path = self.path_data + '/' + self.variant + "_" + str(exp) + ".pkl"
        file = open(path, 'rb')
        res = pickle.load(file)
        file.close()
        return res

    def load_excel(self):
        """
        Load aggregated model results from an Excel sheet.

        Creates results if they do not exist.

        Returns
        -------
        pandas.core.frame.DataFrame
            Model results as a dataframe.
        """
        path = self.path_data + '/' + self.variant + '.xlsx'
        if not os.path.exists(path):
            self.store(save_pckl=False)
        return pandas.read_excel(path, engine='openpyxl')

    def load_tracker(self, res, exp, sim):
        """
        Convenience function to extract the tracker for a specific result, simulation.

        Parameters
        ----------
        res : str
            Name of the result, must match Tracker.var.
        exp : int
            Number of the experiment.
        sim : int
            Number of the simulation run.

        Returns
        -------
        abm.assistant.Tracker
            Results tracker.
        """
        if exp not in range(1, self.experiments + 1) or sim not in range(1, self.params_fix['S'] + 1):
            raise KeyError("Tried to load tracker with invalid experiment or simulation number")
        try:
            return self.results[exp - 1][sim - 1][res]
        except KeyError:
            raise KeyError("Unknown results variable, check that 'res' matches a tracker name")
