"""Analysis
Create custom visuals for analysis.

This script requires data from completed model simulations.
"""

# DEPENDENCIES
import statistics
import math
from operator import itemgetter
import copy
import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.lines
import matplotlib.cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import abm
from abm.constants import *
import abm.assistant.functions as aux
import abm.assistant.tracker as tracker


# HELPER
def set_size(width=489, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/.

    Parameters
    ----------
    width: float or string
            LaTex document width in points.
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


############################################################################################################
# MAIN
############################################################################################################
def visualize_experiments_heatmap(model, data, x, y, z,
                                  z_percent=False, zlegend=('Min', 'Max'), znorm=(0, 1),
                                  fun='mean', labx='x', laby='y', labz='z', cmap='Greys', cmap_seq=True,
                                  save=False):
    """
    Visualize variation of result variable in response to two swept parameters on a heatmap.

    Parameters
    ----------
    model: abm.model.Model
        Model to visualize. Only needed for model parameters, results are imported from data frame.
    data : pandas.core.frame.DataFrame
        Model results dataframe to visualize z.
    x : str
        Name of parameter that varied across experiments. Will be displayed on the x-axis.
         Must be in model.sweeper[0].
    y : str
        Name of parameter that varied across experiments. Will be displayed on the y-axis.
         Must be in model.sweeper[0].
    z : str
        Result corresponding to column name in data that will be used to create the heatmap.
    z_percent : bool
        Whether z should be converted to percentages of N.
    zlegend : List<str>
        Labels for z-axis ticks (one for each in znorm).
    znorm : Tuple<float>
        Minimum, central tendency and maximum value for colorbar normalization.
        For example, set to (-100, 0, 100) for values between -100 and 100.
    fun : str
        Function applied to z by experiment, must be one of 'mean', 'median', 'min', 'max', or 'first'.
    labx : str
        Label for x-axis.
    laby : str
        Label for y-axis.
    labz : str
        Label for heatmap legend (z-axis).
    cmap : str
        Colormap in matplotlib.colors.ListedColormap.
    cmap_seq : bool
        Whether colormap is sequential (True) or divergent (False).
        If set to True (default), the middle value of znorm is ignored.
    save: bool
        Whether to store figure on disk.

    Returns
    -------
    None
    """
    # prepare data
    if fun == 'mean':
        res = data.groupby('experiment')[z].mean()
    elif fun == 'median':
        res = data.groupby('experiment')[z].median()
    elif fun == 'min':
        res = data.groupby('experiment')[z].min()
    elif fun == 'max':
        res = data.groupby('experiment')[z].max()
    elif fun == 'first':
        res = data.groupby('experiment')[z].agg('first')
    else:
        raise ValueError("Illegal option supplied to fun, please check the documentation")
    xloc = model.sweeper[0].index(x)
    yloc = model.sweeper[0].index(y)
    df = pandas.DataFrame({z: res.values,
                           x: [model.sweeper[1:][e][xloc] for e in range(0, model.experiments)],
                           y: [model.sweeper[1:][e][yloc] for e in range(0, model.experiments)]})
    dfplot = df.set_index([y, x])[z].unstack()

    # visualize
    fig, axis = plt.subplots()
    if cmap_seq:
        norm = matplotlib.colors.Normalize(vmin=znorm[0], vmax=znorm[-1])
    else:
        norm = matplotlib.colors.TwoSlopeNorm(vmin=znorm[0], vcenter=znorm[1], vmax=znorm[-1])
    heatmap = axis.pcolor(dfplot, cmap=matplotlib.colormaps[cmap], norm=norm)
    plt.xticks(np.arange(len(dfplot.index)) + 0.5, dfplot.index)
    plt.yticks(np.arange(len(dfplot.columns)) + 0.5, dfplot.columns)
    plt.xlabel(labx)
    plt.ylabel(laby)
    cbar = matplotlib.pyplot.colorbar(heatmap)
    cbar.set_label(labz)
    cbar.ax.get_yaxis().set_ticks(znorm)
    if z_percent:
        zlegend = ['0%', '100%'] if zlegend is None else zlegend
        cbar.ax.set_yticklabels(zlegend)
        cbar.ax.text(0, -105, zlegend[0], ha='left', va='top')
        cbar.ax.text(0, 105, zlegend[-1], ha='left', va='bottom')
    else:
        zlegend = ['Min', 'Max'] if zlegend is None else zlegend
        cbar.ax.set_yticklabels(zlegend)
    if save:
        fig.savefig(model.path_visuals + '/heatmap_' + z + '.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT, dpi=IMAGE_DPI)
        plt.close(fig)
    else:
        plt.show()

def visualize_type_pattern(model, exp, sim=None, ci=False, save=True, path=None, ycolors=None, yline=None):
    """
    Visualize defector types as LOWESS for a selected experiment.

    Except for experiments, defaults are fixed to fit desired visualization in main body of the paper.

    Parameters
    ----------
    model : abm.model.Model
    exp : int
        Selected experiment.
    sim : int
        Selected simulation for the given experiment.
        If None, plot mean value across simulations.
    ci : bool
        Whether to plot confidence intervals around type prevalences.
        Only applies if no simulations are given (sim is None).
    save : bool
    path : str
    ycolors : List<str>
    yline : List<str>

    Returns
    -------
    None
    """
    # prepare figure
    fig, axs = plt.subplots(figsize=set_size(), dpi=IMAGE_DPI)
    axs.set_xticks([0, 0.5 * model.params_fix['G'], model.params_fix['G']])
    axs.set_yticks([0 * model.params_fix['N'], 0.25 * model.params_fix['N'],
                    0.5 * model.params_fix['N'], 0.75 * model.params_fix['N'], model.params_fix['N']])
    color, color_btw = 'black', 'grey'
    plt.xlim(0, model.params_fix['G'])
    plt.ylim(0, model.params_fix['N'])
    x = [i for i in range(1, model.params_fix['G'] + 1)]
    fig.subplots_adjust(bottom=0.3)
    ylegend = ["Conformers (I)", "False Defectors (III)", "Secret Defectors (II)", "Defectors (IV)"]
    ycolors = ['grey', 'red', 'black', 'sienna'] if ycolors is None else ycolors
    yline = ['solid', 'solid', 'solid', 'solid'] if yline is None else yline
    axs.set_ylabel('Number of Agents')
    axs.set_xlabel('Generation')
    plt.suptitle("")
    plt.title("")

    # obtain results
    if sim is None:
        tc, sd, fd, td = [], [], [], []
        tct = [aux.pop_none(model.load_tracker("trueconf", exp, sim).values)
               for sim in range(1, model.params_fix['S'] + 1)]
        sdt = [aux.pop_none(model.load_tracker("secretdef", exp, sim).values)
               for sim in range(1, model.params_fix['S'] + 1)]
        fdt = [aux.pop_none(model.load_tracker("falsedef", exp, sim).values)
               for sim in range(1, model.params_fix['S'] + 1)]
        tdt = [aux.pop_none(model.load_tracker("truedef", exp, sim).values)
               for sim in range(1, model.params_fix['S'] + 1)]

        # aggregate across simulations
        for g in range(0, model.params_fix['G']):
            tc.append(statistics.mean(tct[sim][g] for sim in range(0, model.params_fix['S'])))
            sd.append(statistics.mean(sdt[sim][g] for sim in range(0, model.params_fix['S'])))
            fd.append(statistics.mean(fdt[sim][g] for sim in range(0, model.params_fix['S'])))
            td.append(statistics.mean(tdt[sim][g] for sim in range(0, model.params_fix['S'])))
    else:
        tc = aux.pop_none(model.load_tracker("trueconf", exp, sim).values)
        sd = aux.pop_none(model.load_tracker("secretdef", exp, sim).values)
        fd = aux.pop_none(model.load_tracker("falsedef", exp, sim).values)
        td = aux.pop_none(model.load_tracker("truedef", exp, sim).values)

    # plot
    for t, trcker in enumerate([tc, sd, fd, td]):
        y_sm, y_std = aux.lowess(np.array(x), np.array(trcker))
        axs.plot(x, y_sm, linewidth=2, color=ycolors[t], linestyle=yline[t])
        if sim is None and ci:
            axs.fill_between(x, y_sm - 1.96 * y_std,
                             y_sm + 1.96 * y_std, alpha=0.3, color=color_btw)

    # plot legend
    axs.legend(ylegend, loc='upper center', bbox_to_anchor=(0.5, -0.2),
               ncol=2, fancybox=True)

    # save
    if save:
        path = (model.path_visuals + "/trajectories_idealtypes_"
                + str(exp) + "_" + "." + IMAGE_FORMAT) if path is None else path
        plt.savefig(path,
                    format=IMAGE_FORMAT, dpi=IMAGE_DPI)
        plt.close(fig)
    else:
        plt.show()


def visualize_type_patterns(model, exps, sims=None, ci=False, save=True, path=None,
                            ycolors=None, yline=None,
                            ax0title="(A) Conformity", ax1title="(B) Defection"):
    """
    Visualize defector types as LOWESS for two selected experiments.

    Except for experiments, defaults are fixed to fit desired visualization in main body of the paper.

    Parameters
    ----------
    model : abm.model.Model
    exps : List<int>
        Selected experiments, must be of length 2.
    sims : List<int>
        Selected simulation for each given experiments.
        If None, plot mean value across simulations for each experiment.
    ci : bool
        Whether to plot confidence intervals around type prevalences.
        Only applies if no simulations are given (sim is None).
    save : bool
    path : str
    ycolors : List<str>
    yline : List<str>
    ax0title : str
    ax1title : str

    Returns
    -------
    None
    """
    # prepare figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=set_size(), dpi=IMAGE_DPI,
                            sharey='all')
    for a in axs:
        a.set_xticks([0, 0.5 * model.params_fix['G'], model.params_fix['G']])
        a.set_yticks([0 * model.params_fix['N'], 0.25 * model.params_fix['N'],
                      0.5 * model.params_fix['N'], 0.75 * model.params_fix['N'], model.params_fix['N']])
    color, color_btw = 'black', 'grey'
    plt.xlim(0, model.params_fix['G'])
    plt.ylim(0, model.params_fix['N'])
    x = [i for i in range(1, model.params_fix['G'] + 1)]
    fig.subplots_adjust(bottom=0.3)
    ylegend = ["Conformers (I)", "False Defectors (III)", "Secret Defectors (II)", "Defectors (IV)"]
    ycolors = ['grey', 'red', 'black', 'sienna'] if ycolors is None else ycolors
    yline = ['solid', 'solid', 'solid', 'solid'] if yline is None else yline
    axs[0].set_ylabel('Number of Agents')
    axs[0].set_xlabel('Generation')
    axs[1].set_xlabel('Generation')
    plt.suptitle("")
    plt.title("")
    axs[0].set_title(ax0title)
    axs[1].set_title(ax1title)

    for e, exp in enumerate(exps):
        # obtain results
        if sims is None:
            tc, sd, fd, td = [], [], [], []
            tct = [aux.pop_none(model.load_tracker("trueconf", exp, sim).values)
                   for sim in range(1, model.params_fix['S'] + 1)]
            sdt = [aux.pop_none(model.load_tracker("secretdef", exp, sim).values)
                   for sim in range(1, model.params_fix['S'] + 1)]
            fdt = [aux.pop_none(model.load_tracker("falsedef", exp, sim).values)
                   for sim in range(1, model.params_fix['S'] + 1)]
            tdt = [aux.pop_none(model.load_tracker("truedef", exp, sim).values)
                   for sim in range(1, model.params_fix['S'] + 1)]

            # aggregate across simulations
            for g in range(0, model.params_fix['G']):
                tc.append(statistics.mean(tct[sim][g] for sim in range(0, model.params_fix['S'])))
                sd.append(statistics.mean(sdt[sim][g] for sim in range(0, model.params_fix['S'])))
                fd.append(statistics.mean(fdt[sim][g] for sim in range(0, model.params_fix['S'])))
                td.append(statistics.mean(tdt[sim][g] for sim in range(0, model.params_fix['S'])))
        else:
            tc = aux.pop_none(model.load_tracker("trueconf", exp, sims[e]).values)
            sd = aux.pop_none(model.load_tracker("secretdef", exp, sims[e]).values)
            fd = aux.pop_none(model.load_tracker("falsedef", exp, sims[e]).values)
            td = aux.pop_none(model.load_tracker("truedef", exp, sims[e]).values)

        # plot
        for t, trcker in enumerate([tc, sd, fd, td]):
            y_sm, y_std = aux.lowess(np.array(x), np.array(trcker))
            axs[e].plot(x, y_sm, linewidth=2, color=ycolors[t], linestyle=yline[t])
            if sims is None and ci:
                axs[e].fill_between(x, y_sm - 1.96 * y_std,
                                    y_sm + 1.96 * y_std, alpha=0.5, color=color_btw)

    # plot legend
    axs[1].legend(ylegend, loc='upper center', bbox_to_anchor=(-0.1, -0.2),
                  ncol=2, fancybox=True)

    # save
    if save:
        path = (model.path_visuals + "/trajectories_idealtypes_" + str(exps[0]) + "_"
                + str(exps[1]) + "." + IMAGE_FORMAT) if path is None else path
        plt.savefig(path,
                    format=IMAGE_FORMAT, dpi=IMAGE_DPI)
        plt.close(fig)
    else:
        plt.show()


def visualize_conformity(model, exp, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    'custom blue', ['#f3a582', '#6d0220'], N=100),
                         colorline=True, linewidth=1):
    """Visualize group conformity over generations for all simulations of a given experiment."""
    def draw_simulations(xs, y_vec, z_vec, axs):
        for vs, cs in zip(y_vec, z_vec):
            y_sm, _ = aux.lowess(np.array(xs), np.array(vs), f=1. / 5.)
            if colorline:
                aux.colorline(xs, y_sm, norm(cs), linewidth=linewidth, cmap=cmap, ax=axs)
            else:
                plt.plot(xs, y_sm, linewidth=linewidth, color=cmap(norm(cs)), ax=axs)

    # prepare data
    x = [g for g in range(0, model.params_fix['G'])]
    y, z = [], []

    for sim in range(1, model.params_fix['S'] + 1):
        y.append([dl for dl in model.load_tracker('allegiance', exp, sim).values][1:])
        z.append([None if vl is None else vl for vl in model.load_tracker('labeling', exp, sim).values][1:])
    fig = plt.figure(figsize=set_size(), dpi=IMAGE_DPI)
    fig.gca().set_xlim(0, model.params_fix['G'])
    fig.gca().set_xticks([x for x in range(0, model.params_fix['G'] + 1, int(model.params_fix['G'] / 2))])
    fig.gca().set_ylim(-100, 100)
    fig.gca().set_yticks([-100, 0, 100])
    fig.gca().set_yticklabels(['-100%', '0%', '100%'])
    cmap = matplotlib.cm.get_cmap('cool') if cmap is None else cmap
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    fig.gca().set_xlabel("Generation")
    fig.gca().set_ylabel(r"Group Conformity ($\Delta_\lambda$ %)")
    # colorbar
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=fig.gca(), orientation='horizontal', label='% Labeled Agents')
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.get_xaxis().set_ticks([])
    cbar.ax.text(0, -0.67, '0%', ha='left', va='bottom'),
    cbar.ax.text(50, -0.67, '', ha='center', va='bottom'),
    cbar.ax.text(100, -0.67, '100%', ha='right', va='bottom')
    # draw conformity
    if aux.nest_level(y) > 2:  # [optional] expected nesting levels: [variables >] simulations > values
        for yvec, zvec in zip(y, z):
            draw_simulations(x, y_vec=yvec, z_vec=zvec, axs=fig.gca())
    else:
        draw_simulations(x, y_vec=y, z_vec=z, axs=fig.gca())
    if model.save_results:
        fig.savefig(model.path_visuals + '/simulations_' + 'conformity_' + str(exp) + '.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT, dpi=IMAGE_DPI)
    plt.close(fig)


############################################################################################################
# SI ONLY
############################################################################################################
def visualize_eq_fitness(save=True):
    """Model specification figure to illustrate fitness score equation."""
    plt.figure(figsize=set_size(), dpi=IMAGE_DPI)
    plt.xlim(-1, 1)
    plt.ylim(-0.5, 1)
    plt.xlabel(r"Deviance from Loyalty Expectations ($\delta_i$)")
    plt.ylabel(r"Fitness Score ($f$)")
    colors = ['black', 'sienna', 'red', 'grey']
    linetypes = ['solid', '-.', 'dotted']
    plt.gca().set_aspect('equal')
    plt.grid(False)

    z1 = [1, -1, 0, -1]  # k
    z1_2 = [1, 1, 0.5, 0]  # lambda
    z2 = [0.1, 0.5]  # p-i
    for y in [0.5, 0, -0.5]:
        plt.axhline(y=y, color='grey', linestyle=':', linewidth=1)
    for vl, k in enumerate(z1):
        lam = z1_2[vl]
        plt.axvline(x=lam, color='grey', linestyle=':', linewidth=1)
        plt.axvline(x=lam - 1, color='grey', linestyle=':', linewidth=1)
        xs = np.linspace(lam - 1, lam, 101)
        y = [((x ** 2) / (math.exp(k * x))) - (0 * vl) for x in xs]  # assuming no label
        plt.plot(xs, y, color=colors[vl], linestyle=linetypes[0], linewidth=1,
                 label=r'k = ' + str(k) + r', $\lambda$ = ' + str(lam))
    for d, diff in enumerate(z2):
        xs = np.linspace(-0.5, 0.5, 101)  # assuming lambda = 0.5
        y = [((x ** 2) / (math.exp(0 * x))) - (diff * 1) for x in xs]  # assuming k = 0
        plt.plot(xs, y, color=colors[2], linestyle=linetypes[d + 1], linewidth=1,
                 label=r'$|p_A-i_A| \cdot l$ = ' + str(diff))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
               ncol=3, fancybox=True)
    if save:
        plt.savefig(PATH_VISUALS + '/final_fitness.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT, dpi=IMAGE_DPI, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_shift_heatmap(model, detect_shifts=True, simplify_shifts=False, x='P_AGG', y='LAM', save=False,
                            cmap='tab20',
                            xlab=r'Initial Behavior ($\bar{i} = \bar{p}$)',
                            ylab=r'Loyalty Expectations ($\lambda$)'):
    """
    Visualize relationship between two parameters x and y, and automatically detected allegiance shifts (z).

    Parameters
    ----------
    model : abm.model.Model
    detect_shifts : bool
        True if shifts should be detected and mapped, False if only the final (dominant) outcome should be mapped.
    simplify_shifts : bool
        Whether to simplify shifts from all possible defector types to conformity and defection.
    x : str
    y : str
    save : bool
    cmap : Union(str, List<str>)
    xlab : str
    ylab : str

    Returns
    -------
    None
    """
    # get data frame
    dat = model.load_excel()

    # classify experiments
    exps, expchanges = [], []
    for exp in range(1, model.experiments + 1):
        sims, changecount = [], []
        for sim in range(1, model.params_fix['S'] + 1):
            # get dominant defector types
            tc, sd, fd, td = aux.pop_none(model.load_tracker("trueconf", exp, sim).values), \
                aux.pop_none(model.load_tracker("secretdef", exp, sim).values), \
                aux.pop_none(model.load_tracker("falsedef", exp, sim).values), \
                aux.pop_none(model.load_tracker("truedef", exp, sim).values)
            conformers = [tc[g] + fd[g] for g in range(0, model.params_fix['G'])]
            defectors = [sd[g] + td[g] for g in range(0, model.params_fix['G'])]
            if simplify_shifts:
                trackers = {'Conformity': conformers, 'Defection': defectors}
                at = []
                for c, d in zip(conformers, defectors):
                    types = [c, d]
                    at.append(['Conformity', 'Defection'][types.index(max(types))])
            else:
                trackers = {'I': tc, 'II': sd, 'III': fd, 'IV': td}
                at = []
                for c, s, f, d in zip(tc, sd, fd, td):
                    types = [c, s, f, d]
                    at.append(['I', 'II', 'III', 'IV'][types.index(max(types))])
            dominant = sorted([{a: at.count(a)} for a in set(at)],
                              key=lambda k: [k for k in k.values()], reverse=True)[:2]
            dominanttypes = [k for d in dominant for k in d.keys()]

            # detect unique changes between dominant types, including only
            # (1) changes between the two most dominant types, (2) mean prevalence differs before and after change point
            changes, changespos, changes_all = [], [], []
            for i in range(2, len(at) - 1):
                if (at[i] != at[i - 1]
                        and statistics.mean(trackers[at[i]][:i]) < statistics.mean(trackers[at[i]][i + 1:])
                        and statistics.mean(trackers[at[i - 1]][:i]) > statistics.mean(trackers[at[i - 1]][i + 1:])
                        and at[i] in dominanttypes and at[i - 1] in dominanttypes):
                    changes.append((at[i - 1], at[i]))
                    changespos.append(i)
                # count all changes between types
                if at[i] != at[i - 1]:
                    changes_all.append((at[i - 1], at[i]))
            if len(changes) == 0:
                # dominant type without change
                sims.append(list(dominant[0])[0])
            elif len(set(changes)) == 1:
                # single change
                if detect_shifts:
                    sims.append(changes[0][0] + r'$\rightarrow$' + changes[0][1])  # alternative: consider prior type
                    # sims.append(r'$\rightarrow$' + changes[0][1]) # alternative: consider final type only
                else:
                    sims.append(changes[0][1])  # final type
            else:
                # most prevalent change within simulation
                dom = max(changes, key=changes.count)
                if detect_shifts:
                    sims.append(dom[0] + r'$\rightarrow$' + dom[1])  # alternative: consider prior type
                    # sims.append(r'$\rightarrow$' + dom[1])    # alternative: consider final type only
                else:
                    sims.append(dom[1])  # final type of dominant change
            changecount.append(len(changes_all))
        # most prevalent simulation pattern
        exps.append(max(sims, key=sims.count))
        # total changes
        expchanges.append(max(changecount, key=changecount.count))

    # add classification to data frame with aggregated results (by experiment)
    scenarios = [[exp] * model.params_fix['S'] for exp in exps]
    scenarios = [s for exp in scenarios for s in exp]
    expchanges = [[exp] * model.params_fix['S'] for exp in expchanges]
    expchanges = [s for exp in expchanges for s in exp]
    dat['experiment_scenario'] = scenarios
    dat['experiment_changes'] = expchanges

    # plot dominant shifts
    z = 'experiment_scenario'
    xloc = model.sweeper[0].index(x)
    yloc = model.sweeper[0].index(y)
    mapper = {}
    if simplify_shifts:
        labx, laby, labz = xlab, ylab, 'Allegiance Shift'
        shifts = ['Defection', 'Conformity$\\rightarrow$Defection',
                  'Defection$\\rightarrow$Conformity', 'Conformity']
    else:
        labx, laby, labz = xlab, ylab, 'Defector Type'
        shifts = ['IV', 'III$\\rightarrow$IV', 'II$\\rightarrow$IV', 'I$\\rightarrow$IV',  # all possible shifts
                  'IV$\\rightarrow$III', 'II$\\rightarrow$III', 'I$\\rightarrow$III',
                  'IV$\\rightarrow$II', 'III$\\rightarrow$II', 'I$\\rightarrow$II',
                  'II',
                  'IV$\\rightarrow$I', 'III$\\rightarrow$I', 'II$\\rightarrow$I', 'I',
                  '$\\rightarrow$IV', '$\\rightarrow$I', '$\\rightarrow$II'
                  ]
    for s in set(scenarios):
        if s not in shifts:
            raise Exception("Shift visualization failed: shift not recognized: ", str(s))
    i = 0
    for shift in shifts:
        if shift in scenarios:
            mapper[shift] = i
            i += 1
    res = dat.groupby('experiment')[z].agg('first')
    df = pandas.DataFrame({z: res.values,
                           x: [model.sweeper[1:][e][xloc] for e in range(0, model.experiments)],
                           y: [model.sweeper[1:][e][yloc] for e in range(0, model.experiments)]})
    dfplot = df.set_index([y, x])[z].unstack()
    fig, axis = plt.subplots()
    if isinstance(cmap, list):
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom', cmap, N=len(mapper))
        heatmap = axis.pcolor(dfplot.replace(mapper), cmap=cmap)
    else:
        heatmap = axis.pcolor(dfplot.replace(mapper), cmap=plt.cm.get_cmap(cmap, len(mapper)))
    plt.xticks(np.arange(len(dfplot.index)) + 0.5, dfplot.index)
    plt.yticks(np.arange(len(dfplot.columns)) + 0.5, dfplot.columns)
    plt.xlabel(labx)
    plt.ylabel(laby)
    cbar = fig.colorbar(heatmap, ticks=list(mapper.values()))
    cbar.set_label(labz)
    cbar.set_ticks([cbar.vmin + (cbar.vmax - cbar.vmin) / len(mapper) * (0.5 + i) for i in range(len(mapper))])
    cbar.ax.set_yticklabels(list(mapper.keys()))

    if save:
        fig.savefig(model.path_visuals + '/heatmap_' + z + '.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT, dpi=IMAGE_DPI, bbox_inches='tight')
        plt.close(fig)

    # plot shift frequency
    z = 'experiment_changes'
    xloc = model.sweeper[0].index(x)
    yloc = model.sweeper[0].index(y)
    labx, laby, labz = xlab, ylab, 'Allegiance Shift Count'
    res = dat.groupby('experiment')[z].agg('first')
    df = pandas.DataFrame({z: res.values,
                           x: [model.sweeper[1:][e][xloc] for e in range(0, model.experiments)],
                           y: [model.sweeper[1:][e][yloc] for e in range(0, model.experiments)]})
    dfplot = df.set_index([y, x])[z].unstack()
    fig, axis = plt.subplots()
    plt.xticks(np.arange(len(dfplot.index)) + 0.5, dfplot.index)
    plt.yticks(np.arange(len(dfplot.columns)) + 0.5, dfplot.columns)
    plt.xlabel(labx)
    plt.ylabel(laby)
    norm = matplotlib.colors.Normalize(vmin=min(dat['experiment_changes']), vmax=max(dat['experiment_changes']))
    heatmap = axis.pcolor(dfplot,
                          cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                              'custom', ['white', 'grey', 'firebrick', 'darkred'],
                              N=len(set(dat['experiment_changes']))),
                          norm=norm)
    cbar = fig.colorbar(heatmap, ticks=list(set(dat['experiment_changes'])))
    cbar.set_label(labz)
    cbar.set_ticks([cbar.vmin + (cbar.vmax - cbar.vmin) / len(set(dat['experiment_changes'])) * (0.5 + i)
                    for i in range(len(set(dat['experiment_changes'])))])
    cbar.ax.set_yticklabels(list(set(dat['experiment_changes'])))

    if save:
        fig.savefig(model.path_visuals + '/heatmap_count_' + z + '.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT, dpi=IMAGE_DPI, bbox_inches='tight')
        plt.close(fig)


############################################################################################################
# DIAGNOSTICS
############################################################################################################
def baseline_diagnostics(basemodel):
    """Visualize diagnostics for experiments in baseline."""
    dat_baseline = basemodel.load_excel()
    if basemodel.experiments > 1:
        # average types, misidentification, and ratio r by experimental condition
        visualize_experiments(data=dat_baseline,
                              path_visuals=basemodel.path_visuals)
        for var, name in enumerate(basemodel.sweeper[0]):
            # lowess for defector types by value of parameter var
            visualize_experiments_dist(model=basemodel,
                                       xparam=basemodel.sweeper[0][var],
                                       xlab=r'$\lambda$' if name == 'LAM'
                                       else r'$k$' if name == 'K'
                                       else r'$\bar{p}$' if name == 'P_AGG'
                                       else r'$\bar{i}$' if name == 'I_AGG'
                                       else r'$\bar{q}$' if name == 'Q_AGG'
                                       else r'$\sigma_{q}$' if name == 'Q_SD'
                                       else r'$\sigma_{i}$' if name == 'I_SD'
                                       else name,
                                       yparam=['trueconf', 'falsedef', 'secretdef', 'truedef'],
                                       ylegend=['Conformers (I)', 'False Defectors (III)',
                                                'Secret Defectors (II)', 'Defectors (IV)'],
                                       ycolors=['grey', 'red', 'black', 'sienna'])
            # lowess for labeling by value of parameter var
            visualize_experiments_dist(model=basemodel,
                                       xparam=basemodel.sweeper[0][var],
                                       xlab=r'$\lambda$' if name == 'LAM'
                                       else r'$\bar{p}$' if name == 'P_AGG'
                                       else r'$\bar{i}$' if name == 'I_AGG'
                                       else r'$\bar{q}$' if name == 'Q_AGG'
                                       else r'$\sigma_{q}$' if name == 'Q_SD'
                                       else r'$\sigma_{i}$' if name == 'I_SD'
                                       else name,
                                       yparam=['labeling'],
                                       ylegend=['Labeling'],
                                       ycolors=['black'])


def extension_diagnostics(extensionmodel):
    """Visualize diagnostics for experiments in extension models."""
    exps = [1, int(extensionmodel.experiments / 2), extensionmodel.experiments]
    for e in exps:
        visualize_initial_state(extensionmodel, exp=e, sim=1)
        visualize_loyalty_init_final(extensionmodel, exp=e)


def visualize_initial_state(model, exp, sim):
    """Visualize initial distribution of key parameters for given experiment, simulation"""
    tags_i = model.load_tracker('tags_i', exp, sim).values[0]
    tags_p = model.load_tracker('tags_p', exp, sim).values[0]
    tags_q = model.load_tracker('tags_q', exp, sim).values[0]
    lam = model.params_fix['LAM']
    n = model.params_fix['N']

    dat = pandas.melt(pandas.DataFrame(
        {'id': [a for a in range(0, n)],
         'i': [i for i in tags_i],
         'p': [p for p in tags_p]}),
        id_vars=['id'], var_name='attr', value_name='value')
    f, (ax_box, ax_dist) = matplotlib.pyplot.subplots(2, sharex='none', gridspec_kw={"height_ratios": (.15, .85)},
                                                      figsize=set_size(), constrained_layout=True)
    g = sns.histplot(dat, x='value', hue="attr", palette=['black', 'red'], hue_order=['i', 'p'], alpha=0.7,
                     fill=True, bins=22, binrange=(0, 1), ax=ax_dist)
    sns.boxplot(x=[lam - q for q in tags_q],
                color='black', ax=ax_box, medianprops=dict(color="white", alpha=0.7))
    matplotlib.pyplot.axvline(lam, color='red')
    matplotlib.pyplot.xlim(0, 1)
    matplotlib.pyplot.ylim(0, model.params_fix['N'] / 2)
    ax_box.set_xlabel(r'Tolerance ($\lambda - q$)')
    ax_dist.set_xlabel('Allegiance')
    ax_dist.set_ylabel('Agents')
    ax_dist.set_yticks(np.arange(0, (n + 1) / 2, 100))
    ax_box.set_yticks(list([]))
    for ax in [ax_box, ax_dist]:
        ax.set_xticks([x / 10 for x in range(0, 11)])
        ax.set_xticklabels([str(x / 10) for x in range(0, 11)])
    leg = g.axes.get_legend()
    leg.set_title("")
    g.axes.legend(['Loyalty Expectations', 'Public Allegiance', 'Private Behavior'],
                  loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fancybox=True)
    matplotlib.pyplot.savefig(model.path_visuals + '/distribution_initial_state_'
                              + str(exp) + "_" + str(sim) + "." + IMAGE_FORMAT,
                              format=IMAGE_FORMAT, dpi=IMAGE_DPI)
    plt.close()


def visualize_experiments(data, path_visuals='', result=None,
                          xvar="Experiment", xlabels=None, fun_agg=None, kind_visual='bar',
                          logger=logging.getLogger()):
    """
    Visualize results across experiments.

    This helper method has two modes:
    (1) If no result (res)/aggregation function (fun) is specified, it generates visualizations for
        a custom set of results variables.
    (2) If result and aggregation function are specified, it generates a single visualization for
        a selected result without storing it.
    The second mode is designed for manual, ad-hoc representations of results.

    data : pandas.core.frame.DataFrame
        Model results dataframe to visualize.
    path_visuals : str
        Path to directory where visualizations should be saved.
        If empty string '' (default), results are shown directly instead of stored.
    result : str
        Either a str equal to a column in the results Excel sheet, or None to plot default results variables.
        Must specify fun.
    xvar : str
        Label for x-axis.
    xlabels : List<str>
        Labels for x-axis ticks.
    fun_agg : str
        Function used to aggregate res by experiment. See abm.assistant.functions.plot_attr for options.
        Must specify res.
    kind_visual : str
        Type of plot to visualize res after applying fun.
    logger : logging.Logger

    Returns
    -------
    None
    """

    def plot_attr(dat, col, group='experiment', fun='mean', kind='bar', save=None,
                  xlim=None, ylim=None, ylabel=None, xlabel='', xticks=None,
                  title=None, colors=None, legend=None):
        """
        Aggregate and visualize data.

        Parameters
        ----------
        dat : pandas.core.frame.DataFrame
            Pandas dataframe containing column to plot.
        col : Union[str, List(str)]
            Name of the column to plot.
        group : str
            Name of the column by which results should be grouped.
        fun : str
            Function applied to col after grouping, must be one of 'mean', 'median', 'min', or 'max'.
        kind : str
            The type of graph to plot.
        save : str
            Specify path to output file if the results should be stored (file extension is added automatically).
        xlim : Tuple(int)
        ylim : Tuple(int)
        ylabel : str
        xlabel : str
        xticks : List<str>
        title : str
        colors : List(str)
        legend : List(str)

        Returns
        -------
        None
        """
        if fun is None or type(fun) is not str:
            raise ValueError("fun must be of type str")
        elif fun == 'mean':
            res = dat.groupby(group)[col].mean()
        elif fun == 'median':
            res = dat.groupby(group)[col].median()
        elif fun == 'min':
            res = dat.groupby(group)[col].min()
        elif fun == 'max':
            res = dat.groupby(group)[col].max()
        else:
            raise ValueError("Illegal option supplied to fun, please check the documentation")
        if colors is None:
            colors = 'black'
        if xticks is None:
            xticks = [i for i in range(1, len(res) + 1)]
        ax = res.plot(kind=kind, xlim=xlim, ylim=ylim, title=title, figsize=set_size(),
                      color=colors)
        plt.subplots_adjust(bottom=0.3)
        ax.set(ylabel=ylabel, xlabel=xlabel)
        ax.set_xticklabels(xticks)
        if legend is not None:
            ax.legend(legend,
                      bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2, fancybox=True)
        if save is not None and save != '':
            matplotlib.pyplot.savefig(save + '.' + IMAGE_FORMAT, format=IMAGE_FORMAT, dpi=IMAGE_DPI)
            matplotlib.pyplot.close('all')
        else:
            matplotlib.pyplot.show()

    if result is not None and fun_agg is not None and kind_visual is not None:
        logger.info("Creating custom visualization across experiments...")
        plot_attr(data, col=result, group='experiment', fun=fun_agg, kind=kind_visual, xticks=xlabels)
    elif result is None and fun_agg is None:
        logger.info("Creating default visualizations across experiments...")
        plot_attr(data, col=['falsedef_mean', 'secretdef_mean', 'truedef_mean', 'trueconf_mean'],
                  group='experiment', fun='mean', kind='bar', save=path_visuals + '/average_types',
                  xlabel=xvar, ylabel="Number of Agents", xticks=xlabels,
                  colors=['grey', 'red', 'black', 'sienna'],
                  legend=['Conformers (I)', 'False Defectors (III)', 'Secret Defectors (II)', 'Defectors (IV)'])
        plot_attr(data, col=['falsedef_mean', 'secretdef_mean'],
                  group='experiment', fun='mean', kind='bar', save=path_visuals + '/average_misdef',
                  xlabel=xvar, ylabel="Number of Agents", xticks=xlabels,
                  colors=['red', 'black'],
                  legend=['False Defectors', 'Secret Defectors'])
        plot_attr(data, col='r_mean', group='experiment', fun='mean', kind='bar',
                  save=path_visuals + '/average_r',
                  xlabel=xvar, ylabel=r"\rho", xticks=xlabels,
                  colors=['black'])
    else:
        raise ValueError("Illegal use of method parameters, please check the documentation")


def visualize_experiments_dist(model, xparam=None, xlab=None, yparam=None, ylegend=None, ycolors=None,
                               logger=logging.getLogger()):
    """
    Visualizes and optionally stores results variable across experiments.

    Visualizations are stored in model.path_visuals if model.save_results is set to True.

    Parameters
    ----------
    model : abm.model.Model
    xparam : str
        Name of a parameter that was swept as an experimental condition.
        Must be in 'model.sweeper[0]'. Defaults to the first parameter that is being swept.
        If multiple parameters were swept, results are averaged across experiments for values of xparam.
    xlab : str
        Label for x-axis, defaults to xparam.
    yparam : List<str>
        Name of parameter(s) that should be plotted on the y-axis.
        Must be in model.vars_track['res']. Defaults to defector types.
    ylegend : List<str>
        Labels for parameter(s) in the plot legend.
        Label positions in the list must correspond to position of y-parameters in yparam.
    ycolors : List<str>
        List of colors to plot parameter(s).
        Color positions in the list must correspond to position of y-parameters in yparam.
        Defaults to color from matplotlib's 'tab20c' pool.
    logger : logging.Logger

    Returns
    -------
    None
    """
    logger.info("Visualizing distribution across experiments for swept parameter: %s", xparam)
    if xparam is None:
        var = xparam = model.sweeper[0][0]
    else:
        var = xparam
    index = model.sweeper[0].index(var)

    # get values of variable parameter
    dat = pandas.DataFrame(pandas.Series([x for _, x in sorted(zip(range(1, model.experiments + 1),
                                                                   [e[index] for e in model.sweeper[1:]]))],
                                         name=var))
    if len(model.sweeper[0]) > 1:
        logger.debug("More than one parameter was swept, reducing x-axis parameter to unique values.")
        dat = dat.drop_duplicates()

    # get distribution of y-values across experiments
    if yparam is None:
        yparam = ['falsedef', 'secretdef', 'truedef', 'trueconf']
        ysave = 'types'
    else:
        ysave = yparam[0]
    yvars = []
    for y in yparam:
        if model.vars_track[y] is True:
            yvars.append(y)
    for y in yvars:
        logger.debug("Preparing results for y-axis parameter: %s", y)
        test = model.load_tracker(y, 1, 1)
        if not isinstance(test.last(), int):
            logger.warning("Ratio values in tracker '%s' will be transformed to integer", test.var)
        freq = []
        pdf = []
        for exp in range(1, model.experiments + 1):
            temp = []
            for sim in range(1, model.params_fix['S'] + 1):
                logger.debug("Retrieving results for experiment %d, simulation %d", exp, sim)
                tracked = model.load_tracker(y, exp, sim)
                temp.append(sum(filter(None, tracked.values)))
            freq.append(sum([int(round(e, 0)) for e in temp]))
            pdf.append(freq[-1] / (model.params_fix['N'] * model.params_fix['G'] * model.params_fix['S']))
        if len(model.sweeper[0]) > 1:
            logger.debug("More than one parameter was swept, averaging y-axis parameter across experiments.")
            dat[y + '_freq'], dat[y + '_pdf'], dat[y + '_cdf'] = None, None, None
            for value in dat[var]:
                indices = [i for i, x in enumerate([e[index] for e in model.sweeper[1:]]) if x == value]
                dat.loc[dat[var] == value, [y + '_freq']] = statistics.mean(itemgetter(*indices)(freq))
                dat.loc[dat[var] == value, [y + '_pdf']] = statistics.mean(itemgetter(*indices)(pdf))
            dat[y + '_cdf'] = None
        else:
            dat[y + '_freq'], dat[y + '_pdf'] = freq, pdf
            dat[y + '_cdf'] = dat[y + '_pdf'].cumsum()
        dat.sort_values(by=var, inplace=True)
    if len(dat.columns) < 2:
        logger.error("No usable trackers passed to function, aborting visualization.")
        return

    # smooth
    dat_ip = pandas.DataFrame()
    for y in yvars:
        try:
            f = interp1d(dat[var], dat[y + '_pdf'], kind='cubic')
            x = np.linspace(min([e[index] for e in model.sweeper[1:]]), max([e[index] for e in model.sweeper[1:]]),
                            num=100, endpoint=True)
            dat_ip[var] = x
            dat_ip[y + '_pdf_ip'] = f(x)
        except ValueError as e:
            logger.error("Interpolation failed, plotting original values, error: %s", e)
            dat_ip[var] = dat[var]
            dat_ip[y + '_pdf_ip'] = dat[y + '_pdf']

    # plot lowess
    fig, ax = matplotlib.pyplot.subplots()
    fig.subplots_adjust(bottom=0.3)
    matplotlib.pyplot.rcParams["axes.prop_cycle"] = matplotlib.pyplot.cycler("color",
                                                                             matplotlib.pyplot.cm.get_cmap('tab20c')
                                                                             (np.linspace(0, 1, len(yvars))))
    dat_ip.plot(x=var, y=[y + '_pdf_ip' for y in yvars],
                ax=ax, figsize=set_size(),
                xlim=(min([e[index] for e in model.sweeper[1:]]), max([e[index] for e in model.sweeper[1:]])),
                ylim=(0, 1),
                title="",
                color=ycolors)
    ax.set_xlabel(var) if xlab is None else ax.set_xlabel(xlab)
    ax.set_ylabel('Probability')
    ax.legend(ylegend if ylegend is not None else yparam,
              bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2, fancybox=True)
    if model.save_results:
        fig.savefig(model.path_visuals + '/distribution_' + ysave + '_' + xparam + '_smooth' + '.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT, dpi=IMAGE_DPI)
        matplotlib.pyplot.close(fig)
    else:
        matplotlib.pyplot.show()

    # plot histogram
    fig, ax = matplotlib.pyplot.subplots()
    fig.subplots_adjust(bottom=0.3)
    dat.plot.bar(x=var, y=[y + '_pdf' for y in yvars],
                 ax=ax, figsize=set_size(),
                 xlim=(min([e[index] for e in model.sweeper[1:]]), max([e[index] for e in model.sweeper[1:]])),
                 ylim=(0, 1),
                 title="",
                 color=ycolors)
    ax.set_xlabel(var) if xlab is None else ax.set_xlabel(xlab)
    ax.set_ylabel('Probability')
    ax.legend(ylegend if ylegend is not None else yparam,
              bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2, fancybox=True)
    if model.save_results:
        fig.savefig(model.path_visuals + '/distribution_' + ysave + '_' + xparam + '.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT, dpi=IMAGE_DPI)
        matplotlib.pyplot.close(fig)
    else:
        matplotlib.pyplot.show()


def visualize_simulations(model, clean=False, logger=logging.getLogger()):
    """
        Visualize all simulation results for all experiments.

        Parameters
        ----------
        model : abm.model.Model
        clean : bool
            Whether to delete results from this instance of model after visualizing them.
            Set this option to True to conserve memory, after preserving a copy of the model instance on disk.
        logger : logging.Logger

        Returns
        -------
        None
    """
    logger.info("Creating visualizations for model %s (variant %s)...", model.name, model.variant)
    if model.save_results:
        logger.debug("Saving visualizations for variation %s to: %s", model.variant, model.path_visuals)
    else:
        logger.debug("Visualizing simulation results...")

    for e, exp in enumerate(model.results):
        for var in model.vars_vis:
            # visualize multiple results variables by simulation
            if var['option'] == 'ts_combine':
                if type(var['res']) == list and len(var['res']) < 2:
                    logger.error("Results variables %s misspecified: need two variables for TS combination",
                                 var['res'])
                    pass
                else:
                    for s, sim in enumerate(exp):
                        logger.debug("Visualizing %s for experiment %d, simulation %d as combined time series",
                                     var['res'], e + 1, s + 1)
                        res_sim = {}
                        for i in range(0, len(var['res'])):
                            res_sim['tracker' + str(i)] = sim[var['res'][i]]
                        tracker.Tracker.visualize_ts_combine(path=model.path_visuals_file,
                                                             save=model.save_results,
                                                             ylim=var['ylim'],
                                                             ylabel=var['label'],
                                                             colors=var['colors'],
                                                             **res_sim)

            # visualize histograms by simulation
            elif var['option'] == 'hist':
                for s, sim in enumerate(exp):
                    logger.debug("Visualizing %s for experiment %d, simulation %d as histogram",
                                 var['res'], e + 1, s + 1)
                    sim[var['res']].visualize_hist(model.path_visuals_file,
                                                   save=model.save_results,
                                                   xlabel=var['label'],
                                                   colors=var['colors'])

            # visualize time series for a single variable by experiment
            elif var['option'] == 'ts':
                logger.debug("Visualizing result variable %s for experiment %d across simulations",
                             var['res'], e + 1)
                res_exp = {}
                legend = []
                for sim in range(1, len(exp) + 1):
                    key = 'tracker' + str(sim)
                    res_exp[key] = copy.deepcopy(exp[sim - 1][var['res']])
                    res_exp[key].sim = None
                    legend.append("Simulation " + str(sim))
                tracker.Tracker.visualize_ts_combine(path=model.path_visuals_file,
                                                     save=model.save_results,
                                                     ylim=var['ylim'],
                                                     ylabel=var['label'],
                                                     colors=None,
                                                     legend=legend,
                                                     **res_exp
                                                     )
            else:
                logger.error("Invalid visualization option: %s", str(var['option']))
                pass

        if clean:
            logger.debug("Removing %d simulation results from experiment %d", len(exp), e + 1)
            exp.clear()


def visualize_loyalty_init_final(model, exp, linewidth=1):
    """Visualize loyalty expectations and behavior"""

    def draw_loyalty(xs, pis, pps, tols, vl, axes):
        xs = np.array(xs)
        y_sm, _ = aux.lowess(xs, np.array(pis), f=1. / 5.)
        y2_sm, _ = aux.lowess(xs, np.array(pps), f=1. / 5.)
        y3_sm, _ = aux.lowess(xs, np.array(tols), f=1. / 5.)
        vl = np.array(vl)
        axes.plot(xs, y_sm, linewidth=linewidth, color='black')
        axes.plot(xs, y2_sm, linewidth=linewidth, color='grey', linestyle='dotted')
        axes.axhline(vl[0], linewidth=linewidth, color='red')
        axes.plot(xs, tols, linewidth=linewidth, color='red', linestyle='dotted')
        axes.fill_between(xs[np.where(y_sm < vl)],
                          vl[np.where(y_sm < vl)],
                          y_sm[np.where(y_sm < vl)],
                          color='sienna', alpha=0.5, edgecolor='black')
        axes.fill_between(xs[np.where(y_sm >= vl)],
                          vl[np.where(y_sm >= vl)],
                          y_sm[np.where(y_sm >= vl)],
                          color='grey', alpha=0.5, edgecolor='black')

    # prepare data
    loyalty_x = [a for a in range(0, model.params_fix['N'])]
    if 'LAM' in model.params_fix:
        lam = model.params_fix['LAM']
    else:
        lam = model.sweeper[1:][exp - 1][model.sweeper[0].index('LAM')]
    loyalty_lams = [lam for _ in loyalty_x]
    loyalty_y_first = sorted(model.load_tracker('tags_i', exp, 1).values[0])
    loyalty_y2_first = sorted(model.load_tracker('tags_p', exp, 1).values[0])
    loyalty_y3_first = [lam - q for q in sorted(model.load_tracker('tags_q', exp, 1).values[0])]
    loyalty_y_last = sorted(model.load_tracker('tags_i', exp, 1).values[-1])
    loyalty_y2_last = sorted(model.load_tracker('tags_p', exp, 1).values[-1])
    loyalty_y3_last = [lam - q for q in sorted(model.load_tracker('tags_q', exp, 1).values[-1])]

    # prepare figure
    fig, axs = plt.subplots(1, 2, dpi=IMAGE_DPI)
    fig.subplots_adjust(bottom=0.3, left=0.15)
    for i, ax in enumerate(axs):
        ax.set_xlim(loyalty_x[0], loyalty_x[1])
        ax.set_xticks([0, model.params_fix['N']])
        ax.set_xticklabels(["0%", "100%"])
        ax.set_xlabel("Agents")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.5, 0.8, 1])
        ax.set_yticklabels(["", "", "", "", ""])
        ax.set_ylabel("")
        ax.title.set_text(["Initial", "Final"][i])
    axs[0].set_ylabel("Loyalty")
    axs[0].set_yticklabels(["", "Low", "Moderate", "High", ""])

    # draw data
    draw_loyalty(loyalty_x, loyalty_y_first, loyalty_y2_first, loyalty_y3_first, loyalty_lams, axes=axs[0])
    draw_loyalty(loyalty_x, loyalty_y_last, loyalty_y2_last, loyalty_y3_last, loyalty_lams, axes=axs[1])

    # custom legend
    custom_lines = [Line2D([0], [0], color='red', lw=linewidth, linestyle='solid'),
                    Line2D([0], [0], color='red', lw=linewidth, linestyle='dotted'),
                    Patch(facecolor='sienna', edgecolor='sienna'),
                    Line2D([0], [0], color='black', lw=linewidth, linestyle='solid'),
                    Line2D([0], [0], color='black', lw=linewidth, linestyle='dotted'),
                    Patch(facecolor='grey', edgecolor='grey')]
    fig.legend(custom_lines,
               ['Loyalty Expectations', r'Tolerance ($\lambda - q$)', 'Defection',
                'Private Behavior', 'Perceived Behavior', 'Conformity'],
               loc='lower center', ncol=2, bbox_to_anchor=(0.55, 0)
               )

    if model.save_results:
        fig.savefig(model.path_visuals + '/simulations_' + 'loyalty_' + str(exp) + '.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT, dpi=IMAGE_DPI)
        plt.close(fig)


############################################################################################################
# DEPRECATED
############################################################################################################
def visualize_identification(data, model, x='TAU', y='LAM', z='r_mean',
                             labx=r'Official Loyalty Expetations ($\tau$)',
                             laby=r'Unofficial Loyalty Expectations ($\lambda$)',
                             labz=r'$\rho$ (Mean)', cmap='RdGy'):
    """Visualize over- and under-identification parameter (r) on heatmap (see visualize_experiments_heatmap)."""
    # prepare data frame
    res = data.groupby('experiment')['r_mean'].mean()
    normalizer = matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(data['r_mean']))
    zvals = normalizer(res.values)
    xloc = model.sweeper[0].index(x)
    yloc = model.sweeper[0].index(y)
    df = pandas.DataFrame({z: zvals,
                           x: [model.sweeper[1:][e][xloc] for e in range(0, 121)],
                           y: [model.sweeper[1:][e][yloc] for e in range(0, 121)]})
    dfplot = df.set_index([y, x])[z].unstack()

    # visualize
    fig, axis = matplotlib.pyplot.subplots()
    heatmap = axis.pcolor(dfplot, cmap=matplotlib.pyplot.cm.get_cmap(cmap))
    matplotlib.pyplot.xticks(np.arange(len(dfplot.index)) + 0.5, dfplot.index)
    matplotlib.pyplot.yticks(np.arange(len(dfplot.columns)) + 0.5, dfplot.columns)
    matplotlib.pyplot.xlabel(labx)
    matplotlib.pyplot.ylabel(laby)
    cbar = matplotlib.pyplot.colorbar(heatmap)
    cbar.set_label(labz)
    cbar.ax.get_yaxis().labelpad = -20
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.text(1.5, max(zvals), r'Over-Identify ($\rho > 1$)', ha='left', va='top'),
    cbar.ax.text(1.5, 0.5, r'Identify/No Defection ($\rho = 1$)', ha='left', va='center'),
    cbar.ax.text(1.5, 0, r'Under-Identify ($\rho < 1$)', ha='left', va='bottom')
    fig.savefig(model.path_visuals + '/heatmap_' + z + '.' + IMAGE_FORMAT,
                format=IMAGE_FORMAT, dpi=IMAGE_DPI)


def visualize_pattern_defection(model, exp, sim):
    """Visualize a single defector type pattern."""
    plt.figure(figsize=set_size(), dpi=IMAGE_DPI)
    tc = aux.pop_none(model.load_tracker("trueconf", exp, sim).values)
    fd = aux.pop_none(model.load_tracker("falsedef", exp, sim).values)
    td = aux.pop_none(model.load_tracker("truedef", exp, sim).values)
    sd = aux.pop_none(model.load_tracker("secretdef", exp, sim).values)

    # plot
    x = [i for i in range(1, 1001)]
    plt.xlim(0, 1000)
    plt.ylim(0, 100)
    plt.xlabel('Generation')
    plt.ylabel('Number of Agents')
    ylegend = ["Conformers (I)", "Secret Defectors (II)", "False Defectors (III)", "Defectors (IV)"]
    ycolors = ['grey', 'black', 'red', 'sienna']
    plt.suptitle("")
    plt.title("")
    for t, trcker in enumerate([tc, sd, fd, td]):
        y_sm = lowess(trcker, x, frac=1. / 3.)
        plt.plot(x, y_sm, linewidth=1, color=ycolors[t])
    plt.legend(ylegend, loc='lower center', ncol=2, fancybox=True)
    plt.subplots_adjust(left=0.125, bottom=0.2, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
    plt.show()
    plt.savefig(model.path_visuals + "/results_profile." + IMAGE_FORMAT, format=IMAGE_FORMAT, dpi=IMAGE_DPI)
    plt.close()


def visualize_simulations_all(model, save, y, z, xlim, ylim, zlim, ylabel, zlabel, zlegend, ysave,
                              yticks=(-100, 0, 100), yticklabels=('-100%', '0%', '100%'),
                              ztextypos=(-1, 0, 1), ztextxpos=(0, 0, 0),
                              cmap=matplotlib.cm.get_cmap('cool'), colorline=True, linewidth=0.15):
    """
    Visualize y and z over generations x as time series with one line per model simulation.

    Parameters
    ----------
    model : abm.model.Model
    save : bool
    y : List<List<float>>
        Nested values to plot over x.
        Nesting depth must be either two (list of values for each simulation),
        or three (list of values for each simulation for each variable to plot).
    z : List<List<float>>
        Nested values to color lines, with one list of values per simulation.
    xlim
    ylim
    zlim
    ylabel
    yticks
    yticklabels
    zlabel
    zlegend: List<str>
        List of length 3 with labels for minimum (-1), middle (0), and maximum (1) value on normalized z-axis scale.
    ztextypos
    ztextxpos
    ysave
    cmap
    colorline
    linewidth

    Returns
    -------
    None

    """
    def draw_lines(xs, y_vec, z_vec):
        for vs, cs in zip(y_vec, z_vec):
            y_sm, _ = aux.lowess(np.array(xs), np.array(vs), f=1. / 5.)
            if colorline:
                aux.colorline(xs, y_sm, norm(cs), linewidth=linewidth, cmap=cmap)
            else:
                plt.plot(xs, y_sm, linewidth=linewidth, color=cmap(norm(cs)))

    fig = plt.figure(figsize=set_size(), dpi=IMAGE_DPI)
    plt.xlim(xlim[0], xlim[1])
    fig.gca().set_xticks([x for x in range(0, model.params_fix['G'] + 1, 500)])
    plt.ylim(ylim[0], ylim[1])
    fig.gca().set_yticks(list(yticks))
    fig.gca().set_yticklabels(list(yticklabels))
    cmap = matplotlib.cm.get_cmap('cool') if cmap is None else cmap
    norm = matplotlib.colors.Normalize(vmin=zlim[0], vmax=zlim[1])
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    x = [i for i in range(0, xlim[1] + 1)]
    if aux.nest_level(y) > 2:  # [optional] expected nesting levels: [variables >] simulations > values
        for yvec, zvec in zip(y, z):
            draw_lines(x, y_vec=yvec, z_vec=zvec)
    else:
        draw_lines(x, y_vec=y, z_vec=z)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, orientation='vertical', label=zlabel)
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.text(ztextxpos[0], ztextypos[0], zlegend[0], ha='left', va='top'),
    cbar.ax.text(ztextxpos[1], ztextypos[1], zlegend[1], ha='left', va='center'),
    cbar.ax.text(ztextxpos[2], ztextypos[2], zlegend[2], ha='left', va='bottom')
    plt.tight_layout()
    if save:
        fig.savefig(model.path_visuals + '/all_simulations_' + ysave + '.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT, dpi=IMAGE_DPI)
        plt.close(fig)


def visualize_conformity_loyalty(model, exp, combined=False,
                                 cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                                     'custom blue', ['#f3a582', 'firebrick'], N=100),
                                 colorline=True, linewidth=0.15):
    """
    Custom plot combining two types of figures:
        - group conformity over generations by simulation,
        - loyalty at beginning and end of simulation (optional)

    Parameters
    ----------
    model : abm.model.Model
    exp : int
        Experiment to plot.
    combined : bool
        Whether to combine the plots into a single figure.
    cmap
        Colormap to use in case y-axis lines should be colored by value.
    colorline
        Whether y-axis simulation lines should be colored by value.
    linewidth

    Returns
    -------
    None
    """

    def draw_simulations(xs, y_vec, z_vec, axs):
        for vs, cs in zip(y_vec, z_vec):
            y_sm, _ = aux.lowess(np.array(xs), np.array(vs), f=1. / 5.)
            if colorline:
                aux.colorline(xs, y_sm, norm(cs), linewidth=linewidth, cmap=cmap, ax=axs)
            else:
                plt.plot(xs, y_sm, linewidth=linewidth, color=cmap(norm(cs)), ax=axs)

    def draw_loyalty(xs, pis, pps, tols, vl, axs):
        xs = np.array(xs)
        y_sm, _ = aux.lowess(xs, np.array(pis), f=1. / 5.)
        y2_sm, _ = aux.lowess(xs, np.array(pps), f=1. / 5.)
        y3_sm, _ = aux.lowess(xs, np.array(tols), f=1. / 5.)
        vl = np.array(vl)
        axs.plot(xs, y_sm, linewidth=linewidth, color='black')
        axs.plot(xs, y2_sm, linewidth=linewidth, color='grey', linestyle='dotted')
        axs.axhline(vl[0], linewidth=linewidth, color='red')
        axs.plot(xs, tols, linewidth=linewidth, color='red', linestyle='dotted')
        axs.fill_between(xs[np.where(y_sm < vl)],
                         vl[np.where(y_sm < vl)],
                         y_sm[np.where(y_sm < vl)],
                         color='sienna', alpha=0.5, edgecolor='black')
        axs.fill_between(xs[np.where(y_sm >= vl)],
                         vl[np.where(y_sm >= vl)],
                         y_sm[np.where(y_sm >= vl)],
                         color='grey', alpha=0.5, edgecolor='black')

    # prepare data
    x = [g for g in range(0, model.params_fix['G'])]
    y, z = [], []
    for sim in range(1, model.params_fix['S'] + 1):
        y.append([100 * dl / (model.params_fix['N'])
                  for dl in model.load_tracker('allegiance', exp, sim).values][1:])
        z.append([None if vl is None else vl for vl in model.load_tracker('labeling', exp, sim).values][1:])
    loyalty_x = [a for a in range(0, model.params_fix['N'])]
    if 'LAM' in model.params_fix:
        lam = model.params_fix['LAM']
    else:
        lam = model.sweeper[1:][exp - 1][model.sweeper[0].index('LAM')]
    loyalty_lams = [lam for _ in loyalty_x]
    loyalty_y_first = sorted(model.load_tracker('tags_i', exp, 1).values[0])
    loyalty_y2_first = sorted(model.load_tracker('tags_p', exp, 1).values[0])
    loyalty_y3_first = [lam - q for q in sorted(model.load_tracker('tags_q', exp, 1).values[0])]
    loyalty_y_last = sorted(model.load_tracker('tags_i', exp, 1).values[-1])
    loyalty_y2_last = sorted(model.load_tracker('tags_p', exp, 1).values[-1])
    loyalty_y3_last = [lam - q for q in sorted(model.load_tracker('tags_q', exp, 1).values[-1])]

    # prepare figure
    if combined:
        fig = plt.figure(figsize=set_size(subplots=(2, 2)), dpi=IMAGE_DPI, constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
        # loyalty top
        ax_top_left = fig.add_subplot(gs[0, 0])
        ax_top_right = fig.add_subplot(gs[0, 1])
        for i, ax in enumerate([ax_top_left, ax_top_right]):
            ax.set_xlim(loyalty_x[0], loyalty_x[1])
            ax.set_xticks([0, model.params_fix['N']])
            ax.set_xticklabels(["0%", "100%"])
            ax.set_xlabel("")
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.2, 0.5, 0.8, 1])
            ax.set_yticklabels(["", "Low", "Moderate", "High", ""])
            ax.set_ylabel("")
            ax.title.set_text(["Initial", "Final"][i])
        ax_top_left.set_ylabel("")
        # conformity bottom
        ax_bottom = fig.add_subplot(gs[1, :])
        ax_bottom.set_xlim(0, model.params_fix['G'])
        ax_bottom.set_xticks([x for x in range(0, model.params_fix['G'] + 1, int(model.params_fix['G'] / 2))])
        ax_bottom.set_ylim(-100, 100)
        ax_bottom.set_yticks([-100, 0, 100])
        ax_bottom.set_yticklabels(['-100%', '0%', '100%'])
        cmap = matplotlib.cm.get_cmap('cool') if cmap is None else cmap
        norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
        ax_bottom.set_xlabel("Generation")
        ax_bottom.set_ylabel(r"Group Conformity ($\Delta_\lambda$ %)")
        # colorbar
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax_bottom, orientation='horizontal', label='% Labeled Agents')
        cbar.ax.get_yaxis().set_ticks([])
        cbar.ax.get_xaxis().set_ticks([])
        cbar.ax.text(-5, -5, '0%', ha='left', va='bottom'),
        cbar.ax.text(0, -5, '', ha='center', va='bottom'),
        cbar.ax.text(110, -5, '100%', ha='right', va='bottom')

        # draw data
        if aux.nest_level(y) > 2:  # [optional] expected nesting levels: [variables >] simulations > values
            for yvec, zvec in zip(y, z):
                draw_simulations(x, y_vec=yvec, z_vec=zvec, axs=ax_bottom)
        else:
            draw_simulations(x, y_vec=y, z_vec=z, axs=ax_bottom)
        draw_loyalty(loyalty_x, loyalty_y_first, loyalty_y2_first, loyalty_y3_first, loyalty_lams, axs=ax_top_left)
        draw_loyalty(loyalty_x, loyalty_y_last, loyalty_y2_last, loyalty_y3_last, loyalty_lams, axs=ax_top_right)

        # custom legend
        custom_lines = [Line2D([0], [0], color='red', lw=linewidth, linestyle='solid'),
                        Line2D([0], [0], color='red', lw=linewidth, linestyle='dotted'),
                        Patch(facecolor='sienna', edgecolor='sienna'),
                        Line2D([0], [0], color='black', lw=linewidth, linestyle='solid'),
                        Line2D([0], [0], color='black', lw=linewidth, linestyle='dotted'),
                        Patch(facecolor='grey', edgecolor='grey')]
        ax_bottom.legend(custom_lines,
                         ['Loyalty Expectations', r'Tolerance ($\lambda - q$)', 'Defection',
                          'Private Behavior', 'Perceived Behavior', 'Conformity'],
                         loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, fancybox=True,
                         )
        if model.save_results:
            fig.savefig(model.path_visuals + '/simulations_' + 'conformity_loyalty_' + str(exp) + '.' + IMAGE_FORMAT,
                        format=IMAGE_FORMAT, dpi=IMAGE_DPI)
            plt.close(fig)
    else:
        # conformity
        fig = plt.figure(dpi=IMAGE_DPI, constrained_layout=True)
        fig.gca().set_xlim(0, model.params_fix['G'])
        fig.gca().set_xticks([x for x in range(0, model.params_fix['G'] + 1, int(model.params_fix['G'] / 2))])
        fig.gca().set_ylim(-100, 100)
        fig.gca().set_yticks([-100, 0, 100])
        fig.gca().set_yticklabels(['-100%', '0%', '100%'])
        cmap = matplotlib.cm.get_cmap('cool') if cmap is None else cmap
        norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
        fig.gca().set_xlabel("Generation")
        fig.gca().set_ylabel(r"Group Conformity ($\Delta_\lambda$ %)")
        # colorbar
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=fig.gca(), orientation='horizontal', label='% Labeled Agents')
        cbar.ax.get_yaxis().set_ticks([])
        cbar.ax.get_xaxis().set_ticks([])
        cbar.ax.text(-5, -5, '0%', ha='left', va='bottom'),
        cbar.ax.text(0, -5, '', ha='center', va='bottom'),
        cbar.ax.text(110, -5, '100%', ha='right', va='bottom')
        # draw conformity
        if aux.nest_level(y) > 2:  # [optional] expected nesting levels: [variables >] simulations > values
            for yvec, zvec in zip(y, z):
                draw_simulations(x, y_vec=yvec, z_vec=zvec, axs=fig.gca())
        else:
            draw_simulations(x, y_vec=y, z_vec=z, axs=fig.gca())
        if model.save_results:
            fig.savefig(model.path_visuals + '/simulations_' + 'conformity_' + str(exp) + '.' + IMAGE_FORMAT,
                        format=IMAGE_FORMAT, dpi=IMAGE_DPI)
            plt.close(fig)
        plt.close(fig)

        # loyalty
        fig = plt.figure(dpi=IMAGE_DPI, constrained_layout=True)
        fig.gca().set_xlim(loyalty_x[0], loyalty_x[1])
        fig.gca().set_xticks([0, model.params_fix['N']])
        fig.gca().set_xticklabels(["0%", "100%"])
        fig.gca().set_xlabel("")
        fig.gca().set_ylim(0, 1)
        fig.gca().set_yticks([0, 0.2, 0.5, 0.8, 1])
        fig.gca().set_yticklabels(["", "Low", "Moderate", "High", ""])
        fig.gca().set_ylabel("")
        fig.gca().title.set_text("")
        fig.gca().set_ylabel("")
        draw_loyalty(loyalty_x, loyalty_y_first, loyalty_y2_first, loyalty_y3_first, loyalty_lams, axs=fig.gca())

        # custom legend
        custom_lines = [Line2D([0], [0], color='red', lw=linewidth, linestyle='solid'),
                        Line2D([0], [0], color='red', lw=linewidth, linestyle='dotted'),
                        Patch(facecolor='sienna', edgecolor='sienna'),
                        Line2D([0], [0], color='black', lw=linewidth, linestyle='solid'),
                        Line2D([0], [0], color='black', lw=linewidth, linestyle='dotted'),
                        Patch(facecolor='grey', edgecolor='grey')]
        fig.gca().legend(custom_lines,
                         ['Loyalty Expectations', r'Tolerance ($\lambda - q$)', 'Defection',
                          'Private Behavior', 'Perceived Behavior', 'Conformity'],
                         loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, fancybox=True,
                         )
        if model.save_results:
            fig.savefig(model.path_visuals + '/simulations_' + 'loyalty_' + str(exp) + '.' + IMAGE_FORMAT,
                        format=IMAGE_FORMAT, dpi=IMAGE_DPI)
            plt.close(fig)


############################################################################################################
# RUN TO VISUALIZE RESULTS FOR GIVEN MODEL
############################################################################################################
def main(model, vis_baseline=True, vis_extensions=True, vis_si=True):
    # load aggregated data
    dat = model.load_excel()

    # baseline figures
    if vis_baseline:
        if model.name == "baseline":
            # heatmap final allegiance
            visualize_experiments_heatmap(model=model, data=dat, x='P_AGG', y='LAM', z='allegiance_last',
                                          labx=r'Initial Behavior ($\bar{i}=\bar{p}$)',
                                          laby=r'Loyalty Expectations ($\lambda$)',
                                          labz=r'Group Conformity ($\Delta_\lambda$)',
                                          z_percent=True,
                                          znorm=(-100, 0, 100),
                                          zlegend=["100% \n Defection",
                                                   "Conformity\n = \n Defection",
                                                   "100%\n Conformity"],
                                          cmap='RdGy',
                                          cmap_seq=False,
                                          save=model.save_results)

        if model.name == "baselinek":
            # selected allegiance shift patterns (defector types over generations)
            visualize_type_patterns(model=model, exps=[21, 1], sims=[1, 1], save=model.save_results)
            visualize_type_patterns(model=model, exps=[21, 1], sims=[1, 1], save=model.save_results,
                                    yline=['solid', 'dotted', 'dashed', 'dashdot'],
                                    path=model.path_visuals + "/trajectories_idealtypes_21_1" + "." + IMAGE_FORMAT
                                    )
            return

        # baseline SI
        if vis_si:
            # fitness
            visualize_eq_fitness(model.save_results)

            # misidentification
            visualize_experiments_heatmap(model=model, data=dat, x='P_AGG', y='LAM', z='r_mean',
                                          labx=r'Initial Behavior ($\bar{i}=\bar{p}$)',
                                          laby=r'Loyalty Expectations ($\lambda$)',
                                          labz=r'Identification ($\rho$)',
                                          zlegend=[r'$\rho = 0$', r'$\rho = 1$', r'$\rho \geq 2$'],
                                          znorm=(0, 1, 2),
                                          cmap='RdGy', cmap_seq=False, save=model.save_results)

    # baseline SI alternative parameter specifications and variants
    elif vis_si:
        # defector type (final) by misperception
        if model.name in ["baselinesipi"]:
            visualize_shift_heatmap(model, detect_shifts=False, simplify_shifts=False,
                                    x='P_AGG', y='I_AGG', save=model.save_results,
                                    cmap=['sienna', 'black', 'grey'],
                                    xlab=r'Perceived Behavior ($\bar{p}$)',
                                    ylab=r'Private Behavior ($\bar{i}$)')

    # extension
    if vis_extensions:
        exps = [1, int(model.experiments / 2), model.experiments]
        for e in exps:
            visualize_type_pattern(model, exp=e,
                                   yline=['solid', 'dotted', 'dashed', 'dashdot'],
                                   path=model.path_visuals
                                        + "/trajectories_idealtypes_" + str(e) + "." + IMAGE_FORMAT)
            visualize_conformity(model, exp=e)


if __name__ == '__main__':
    """Boilerplate method to reproduce main figures with stored model results."""
    baseline = abm.model.Model.load('./storage/baseline/baseline.pkl')
    # Figure 1
    main(model=baseline, vis_baseline=True, vis_extensions=False, vis_si=False)
    del baseline

    # Figure 2
    baselinek = abm.model.Model.load('./storage/baselinek/baseline.pkl')
    main(model=baselinek, vis_baseline=True, vis_extensions=False, vis_si=False)
    del baselinek

    # Figures 3 and 4
    gdr = abm.model.Model.load('./storage/extensiongdr/extension.pkl')  # GDR extension
    opt = abm.model.Model.load('./storage/extensionopt/extension.pkl')  # OPT extension
    for m in [gdr, opt]:
        main(model=m, vis_baseline=False, vis_extensions=True, vis_si=False)
