""" Tracker
Helper class to track, visualize and store results.
"""

# DEPENDENCIES
import matplotlib.pyplot as plt
import itertools
import numpy
import abm.assistant.functions as aux
from abm.constants import IMAGE_FORMAT, IMAGE_DPI, IMAGE_WIDTH, IMAGE_HEIGHT


class Tracker:
    """
    Track, visualize and store model results.

    An instance of Tracker represents results for a single variable of a particular model simulation run.
    The class provides a static method to visualize multiple trackers as a time series.
    To visualize results across simulation runs or experiments, refer to the Model class.

    Attributes
    ----------
    var : str
        Name of the result variable that is tracked.
    exp : int
        Identifier for experiment.
    sim : int
        Identifier for simulation.
    params : Dict
        Fixed and variable experiment parameters.
    model : model
        Name of the model variant.
    values : List
        Result values, where each element represents a result for a generation.
        List may be nested one level maximum.
        Nested values usually imply that results are tracked for each agent (rather than aggregated across agents).

    Methods
    ----------
    mean(forcefal=False)
        Returns mean value of result, or list of mean values if forcefal is False and the result is nested.
    stdev()
        Returns standard deviation of result.
    min()
        Returns minimum value of result.
    max()
        Returns maximum value of result.
    first()
        Returns first value of result.
    last()
        Returns last value of result.
    runs()
        Returns number of generations over which result was tracked.
    append(value)
        Append a single value to results list.
    append_mean(values)
        Append multiple values to (nested) results list.
    visualize_prepare()
        Set standard auxiliary parameters required for plotting.
    visualize_ts(path='', save=False, xlim=None, ylim=None, ylabel=None, xlabel="Generation", suptitle=None,
        title=None, colors=None)
        Visualize a single result as time series.
    visualize_hist(self, path='', save=False, compare=True, ylabel="Frequency", xlabel=None, suptitle=None, title=None,
        bins=10, alpha=0.5, colors=None)
        Visualize a single result as histogram.
    visualize_ts_combine(path='', save=False, xlim=None, ylim=None, ylabel=None, xlabel="Generation", suptitle=None,
        title=None, colors=None, legend=None, **kwargs)
        Visualize multiple Tracker objects in a single time series plot.
    """

    def __init__(self, var, experiment, simulation, params, model):
        self.var = var if isinstance(var, str) else str(var)
        self.exp = experiment
        self.sim = simulation
        self.params = params
        self.model = model
        self.values = []

    @property
    def exp(self):
        return str(self.__exp).zfill(3)

    @exp.setter
    def exp(self, exp):
        self.__exp = int(exp)

    @property
    def sim(self):
        return str(self.__sim).zfill(2) if self.__sim is not None else ''

    @sim.setter
    def sim(self, sim):
        self.__sim = int(sim) if sim is not None else None

    @property
    def title(self):
        title = "Simulation: " + self.sim if self.sim is not '' else None
        for p in self.params.items():
            title = title + ", " if title is not None else ""
            title = title + str(p[0]) + ": " + str(p[1])
        return title

    def __repr__(self):
        if not aux.isnested(self.values):
            return self.var + ": " + ', '.join([str(v) for v in self.values])
        else:
            return self.var + ": " + ', '.join(''.join(str(v)) for v in [l for l in self.values])

    def mean(self, forceval=False):
        if not aux.isnested(self.values):
            return numpy.mean(aux.pop_none(self.values))
        else:
            if forceval:
                return numpy.mean([numpy.mean([v for v in l]) for l in aux.pop_none(self.values)])
            else:
                return [numpy.mean([v for v in l]) for l in aux.pop_none(self.values)]

    def stdev(self):
        if not aux.isnested(self.values):
            return numpy.std(aux.pop_none(self.values))
        else:
            return [numpy.std([v for v in l]) for l in aux.pop_none(self.values)]

    def sum(self):
        if not aux.isnested(self.values):
            return sum(aux.pop_none(self.values))
        else:
            return sum(itertools.chain.from_iterable(aux.pop_none(self.values)))

    def percent(self):
        if not aux.isnested(self.values):
            return sum(aux.pop_none(self.values)) / (self.params['G'] * self.params['N']) * 100
        else:
            return (sum(itertools.chain.from_iterable(aux.pop_none(self.values))))\
                         / (self.params['G'] * self.params['N']) * 100

    def min(self):
        if not aux.isnested(self.values):
            return min(aux.pop_none(self.values))
        else:
            return min(itertools.chain.from_iterable(aux.pop_none(self.values)))

    def max(self):
        if not aux.isnested(self.values):
            return max(aux.pop_none(self.values))
        else:
            return max(itertools.chain.from_iterable(aux.pop_none(self.values)))

    def first(self):
        if not aux.isnested(self.values):
            return self.values[0]
        else:
            return numpy.mean(self.values[0])

    def second(self):
        if not aux.isnested(self.values):
            return self.values[1]
        else:
            return numpy.mean(self.values[1])

    def last(self):
        if not aux.isnested(self.values):
            return self.values[-1]
        else:
            return numpy.mean(self.values[-1])

    def runs(self):
        return len(self.values)

    def append(self, value):
        self.values.append(value)

    def append_mean(self, values):
        self.values.append(numpy.mean([v for v in values]))

    def visualize_prepare(self):
        if aux.isnested(self.values):
            y = self.mean()
        else:
            y = aux.pop_none(self.values)
        x = [i for i in range(0, len(y))]
        stds = self.stdev()
        return [y, x, stds]

    def visualize_ts(self, path='', save=False, xlim=None, ylim=None, ylabel=None, xlabel="Generation",
                     suptitle=None, title=None, colors=None):
        fig = plt.figure(figsize=(IMAGE_WIDTH, IMAGE_HEIGHT), dpi=IMAGE_DPI)
        y, x, stds = self.visualize_prepare()
        if colors is not None:
            plt.plot(x, y, linewidth=2, color=colors[0][0])
            if aux.isnested(self.values):
                plt.fill_between(x, [(v - std) for v, std in zip(y, stds)], [(v + std) for v, std in zip(y, stds)],
                                 alpha=0.5, color=colors[0][1])
        else:
            plt.plot(x, y, linewidth=2)
            if aux.isnested(self.values):
                plt.fill_between(x, [(v - std) for v, std in zip(y, stds)], [(v + std) for v, std in zip(y, stds)],
                                 alpha=0.5)
        plt.xlim(xlim[0], xlim[1]) if xlim is not None else plt.xlim(min(x) - 1, max(x) + 1)
        plt.ylim(ylim[0], ylim[1]) if ylim is not None else plt.ylim(min(y) - 1, max(y) + 1)
        plt.xlabel(xlabel)
        plt.ylabel(self.var if ylabel is None else ylabel)
        plt.suptitle("Experiment " + self.exp if suptitle is None else suptitle)
        plt.title(self.title if title is None else title, size='x-small')
        if save:
            plt.savefig(path + "_" + self.exp + "_" + self.sim + "_" + self.var + "_timeseries." + IMAGE_FORMAT,
                        format=IMAGE_FORMAT, dpi=IMAGE_DPI)
            plt.close(fig)
        else:
            plt.show()

    def visualize_hist(self, path='', save=False, compare=True,
                       ylabel="Frequency", xlabel=None, suptitle=None, title=None,
                       bins=10, alpha=0.5, colors=None):
        fig = plt.figure(figsize=(IMAGE_WIDTH, IMAGE_HEIGHT), dpi=IMAGE_DPI)
        if compare and not aux.isnested(self.values):
            raise TypeError("Visual comparison of two histograms requires Tracker's 'values' to be a nested list")
        else:
            y = self.values
        y = aux.pop_none(y)  # remove potential None elements from values prior to plotting
        if compare:
            if colors is not None:
                plt.hist(y[0], alpha=alpha, label="Initial distribution", bins=bins, color=colors[0])
            else:
                plt.hist(y[0], alpha=alpha, label="Initial distribution", bins=bins)
        if colors is not None:
            plt.hist(y[-1] if aux.isnested(self.values) else y, alpha=alpha, label="Final distribution", bins=bins,
                     color=colors[1])
        else:
            plt.hist(y[-1] if aux.isnested(self.values) else y, alpha=alpha, label="Final distribution", bins=bins)

        plt.xlim(self.min() - 0.01, self.max() + 0.01)
        plt.xlabel(self.var if xlabel is None else xlabel)
        plt.ylabel(ylabel)
        plt.suptitle("Experiment " + self.exp if suptitle is None else suptitle)
        plt.title(self.title if title is None else title, size='x-small')
        plt.legend(bbox_to_anchor=(-0.1, -0.1), loc='upper center', ncol=2, fancybox=True)
        if save:
            plt.savefig(path + "_" + self.exp + "_" + self.sim + "_" + self.var + "_hist." + IMAGE_FORMAT,
                        format=IMAGE_FORMAT, dpi=IMAGE_DPI)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def visualize_ts_combine(path='', save=False, xlim=None, ylim=None, ylabel=None, xlabel="Generation",
                             suptitle=None, title=None, colors=None, legend=None, **kwargs):
        """
        Visualize multiple trackers as time series.

        Parameters
        ----------
        path : str
        save : boolean
        xlim : Tuple(int)
        ylim : Tuple(int)
        ylabel : str
        xlabel : str
        suptitle : str
        title : str
        colors : List(Tuple)
        legend : List(str)
        **kwargs : **Tracker

        Returns
        -------
        None
        """
        tracker = next(iter(kwargs.values()))
        n = len(kwargs)
        i = 0
        fig = plt.figure(figsize=(IMAGE_WIDTH, IMAGE_HEIGHT), dpi=IMAGE_DPI)
        for key, v in kwargs.items():
            y, x, stds = v.visualize_prepare()
            if colors is not None:
                if aux.isnested(v.values):
                    plt.fill_between(x, [(v - std) for v, std in zip(y, stds)], [(v + std) for v, std in zip(y, stds)],
                                     alpha=0.5, color=colors[i][1])
                plt.plot(x, y, linewidth=2, color=colors[i][0])
            else:
                plt.rcParams["axes.prop_cycle"] = plt.cycler("color",
                                                             plt.cm.get_cmap('tab20c')(numpy.linspace(0, 1, n)))
                if aux.isnested(v.values):
                    plt.fill_between(x, [(v - std) for v, std in zip(y, stds)], [(v + std) for v, std in zip(y, stds)],
                                     alpha=0.5)
                plt.plot(x, y, linewidth=2)
            i += 1

        plt.xlim(xlim[0], xlim[1]) if xlim is not None else plt.xlim(0,
                                                                     max([t.runs() for t in kwargs.values()]))
        plt.ylim(ylim[0], ylim[1]) if ylim is not None else plt.ylim(min([t.min() - 1 for t in kwargs.values()]),
                                                                     max([t.max() + 1 for t in kwargs.values()]))
        plt.xlabel(xlabel)
        plt.ylabel(tracker.var if ylabel is None else ylabel)
        plt.suptitle("Experiment " + tracker.exp if suptitle is None else suptitle)
        plt.title(tracker.title if title is None else title, size='x-small')
        if legend is not None:
            plt.legend([l for l in legend],
                       bbox_to_anchor=(-0.1, -0.1), loc='upper center', ncol=2, fancybox=True)
        else:
            plt.legend([t.var for t in kwargs.values()],
                       bbox_to_anchor=(-0.1, -0.1), loc='upper center', ncol=2, fancybox=True)
        if save:
            plt.savefig(path + "_" + tracker.exp + "_" + tracker.sim + "_" + tracker.var
                        + "_timeseries." + IMAGE_FORMAT,
                        format=IMAGE_FORMAT, dpi=IMAGE_DPI)
            plt.close(fig)
        else:
            plt.show()
