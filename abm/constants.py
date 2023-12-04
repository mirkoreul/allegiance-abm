""" Constants
Configure system settings and model parameters.

Model Names
----------
Must not include any underscores '_', as these are reserved for temporary storage files.

Model Parameters
----------
Several parameters must be given in order to start a model run.
For some parameters, if no value is provided, the model will use the value of an alternative parameter.
This allows for sweeps where two or more parameters change simultaneously across experimental conditions.
Specifically, the following parameters should be provided:
- 'LAM' (else equal to: 1. 'I_AGG', 2. 'P_AGG')
- 'K' (else equal to: 1. 'LAM', 2. 'I_AGG', 3. 'P_AGG')
- 'I_AGG'/'P_AGG' (else equal to: 1. 'P_AGG'/'I_AGG', 2. 'LAM')
- 'I_SD'/'P_SD' (else equal to: 'P_SD'/'I_SD')
- 'N', 'S', 'G', 'T', 'P', 'M', 'Q_AGG', 'Q_SD' (no alternative)
See simulation.seed_environment for details.

Variable Trackers
----------
The dictionaries only define which variables should be tracked and how they should be visualized.
Variable specifications are given in simulation.store_results.
Visualizations are implemented in assistant.tracker.visualize* methods (basic diagnostics)
and in analysis.py (paper visuals).
Note that results variables specified in VARS_TARGET and VARS_VIS must be set to True in VARS_TRACK.
"""

# DEPENDENCIES
import datetime
import logging

# SYSTEM SETTINGS
TIMESTAMP = str(datetime.datetime.now())
LOGLEVEL = logging.INFO  # set to logging.DEBUG to diagnose model steps or package functionality
LOGFILE = False  # set to True if logfile should be stored on disk
SAVE_RESULTS = True  # set to True if results should be stored on disk
SAVE_SIMULATIONS = False  # set to True to visualize simulation diagnostics
SI_MODELS_RUN = False  # set to True to produce SI results
PATH_VISUALS = './graphics'  # path to save results visualizations
PATH_DATA = './storage'  # path to save results data, required for temporary storage during parallel processing
PATH_LOGS = './logs'  # path to save logs
RANDOM_SEED = 3273344404
IMAGE_FORMAT = 'pdf'  # figure output format
IMAGE_DPI = 600  # figure output DPI
IMAGE_WIDTH = 20  # figure width in inch
IMAGE_HEIGHT = 12  # figure width in inch
IMAGE_FONTSIZE = 16  # figure text label fontsize

# MODEL PARAMETER DEFAULTS
N = 1000  # number of agents
S = 30  # simulation runs per experiment
G = 100  # time - duration: runs per simulation
T = 10  # time - affected population: percentage of agents who label/adapt
P = 3  # time - interaction frequency: agent labeling pairings per run
M = 0.1  # evolutionary randomness: probability to mutate tolerance, tag
K = 0  # incentives
Q_AGG = 0.1  # mean tolerance
Q_SD = 0.1  # standard deviation of tolerance
I_SD = 0.1  # standard deviation of (private, public) allegiance tags
LAM = 0.5  # loyalty expectation
UNITINTERVAL = [x / 10.00 for x in range(0, 10 + 1)]  # default parameter range from 0 to 1 (0.1 steps)

# MODEL VARIANTS
# MODELS MAIN PAPER
BASELINE = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'T': T, 'M': M, 'P': P, 'Q_AGG': Q_AGG, 'Q_SD': Q_SD, 'I_SD': I_SD
},
    'PARAMS_SWEEP': {  # variable parameters (sweep), LAM=I_AGG, K=LAM (see simulation.seed_environment)
        'LAM': UNITINTERVAL, 'P_AGG': UNITINTERVAL
    }
}
BASELINEK = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'T': T, 'M': M, 'P': P, 'LAM': LAM, 'Q_AGG': Q_AGG, 'Q_SD': Q_SD, 'I_SD': I_SD
},
    'PARAMS_SWEEP': {  # variable parameters (sweep)
        'K': [x / 10.00 for x in range(-10, 10 + 1)]
    }
}
# EXTENSIONS
EXTENSIONGDR = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': 1800, 'T': 0.5, 'M': M, 'P': 1,
    'LAM': 0.7,  # loyalty expectations (all groups)
    'I_AGG': 0.8,  # private allegiance (majority group)
    'P_SHIFT': -0.1,  # difference private + public allegiance (all agents)
    'Q_AGG': 0,
    'Q_SD': 0.01,
    'I_SD': 0.05,
    'P_SD': 0.1,
    'DIST': [(0.95, 0.04, 0.01),
             ({'i_agg': 0.8},
              {'i_agg': 0.5},
              {'i_agg': 0.2})
             ]
},
    'PARAMS_SWEEP': {'K': [x / 10 for x in range(-10, 11)]}
}
EXTENSIONOPT = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': 400, 'T': 1, 'M': M, 'P': P,
    'LAM': 0.6,  # loyalty expectations (all groups)
    'I_AGG': 0.7,  # private allegiance (majority group)
    'P_SHIFT': 0.1,  # difference private + public allegiance (all agents)
    'Q_AGG': 0.1,
    'Q_SD': 0.05,
    'I_SD': 0.1,
    'P_SD': 0.2,
    'DIST': [(0.59, 0.2, 0.2, 0.01),
             ({'i_agg': 0.7, 'p_shift': 0.1},
              {'i_agg': 0.5, 'p_shift': 0.5},
              {'i_agg': 1.0, 'p_shift': -0.5},
              {'i_agg': 0.2, 'p_shift': 0})
             ],
},
    'PARAMS_SWEEP': {'K': [x / 10 for x in range(-10, 11)]}
}
MAINMODELS = {  # models that should always be run (to reproduce main paper sections, create visuals for diagnostics)
    'baseline': 'baseline',   # lambda, i=p
    'baselinek': 'baseline',  # k given i=p=lambda=0.5
    'extensiongdr': 'extension',  # GDR
    'extensionopt': 'extension'   # OPT
}

# MODELS SI
EXTENSIONCUSTOM = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': 100, 'T': 10, 'M': M, 'P': P,
    'LAM': LAM,  # loyalty expectations (all groups)
    'I_AGG': LAM,  # private allegiance (majority group)
    'P_SHIFT': 0,  # difference private + public allegiance (all agents)
    'Q_AGG': 0,
    'Q_SD': 0.001,
    'I_SD': 0.2,
    'P_SD': 0.2
},
    'PARAMS_SWEEP': {'K': [[-1, 1], -1, 1]}
}
BASELINESICUSTOM = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'T': T, 'M': M, 'P': P, 'Q_AGG': Q_AGG, 'Q_SD': Q_SD, 'I_SD': I_SD, 'K': K
},
    'PARAMS_SWEEP': {  # variable parameters (sweep)
        'P_AGG': [0.3, 0.5, 0.7], 'LAM': [0.7, 0.5, 0.3], 'I_AGG': [0.3, 0.5, 0.7],
    }
}
BASELINESIPI = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'T': T, 'M': M, 'P': P, 'LAM': LAM, 'Q_AGG': Q_AGG, 'Q_SD': Q_SD, 'I_SD': I_SD, 'K': K
},
    'PARAMS_SWEEP': {  # variable parameters (sweep)
        'I_AGG': UNITINTERVAL, 'P_AGG': UNITINTERVAL
    }
}
BASELINESIPILAM = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'T': T, 'M': M, 'P': P, 'Q_AGG': Q_AGG, 'Q_SD': Q_SD, 'I_SD': I_SD, 'K': K
},
    'PARAMS_SWEEP': {  # variable parameters (sweep), LAM=I_AGG (see simulation.seed_environment)
        'I_AGG': UNITINTERVAL, 'P_AGG': UNITINTERVAL
    }
}
BASELINESIPLAM = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'T': T, 'M': M, 'P': P, 'Q_AGG': Q_AGG, 'Q_SD': Q_SD, 'I_SD': I_SD, 'I_AGG': LAM, 'K': K
},
    'PARAMS_SWEEP': {  # variable parameters (sweep)
        'P_AGG': UNITINTERVAL, 'LAM': UNITINTERVAL
    }
}
BASELINESIKLAM = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'T': T, 'M': M, 'P': P, 'Q_AGG': Q_AGG, 'Q_SD': Q_SD, 'I_SD': I_SD
},
    'PARAMS_SWEEP': {  # variable parameters (sweep)
        'K': [-1, -0.5, 0, 0.5, 1], 'LAM': UNITINTERVAL
    }
}
BASELINESIQAGG = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'T': T, 'M': M, 'P': P, 'LAM': LAM, 'Q_SD': Q_SD, 'I_SD': I_SD, 'K': K
},
    'PARAMS_SWEEP': {  # variable parameters (sweep)
        'Q_AGG': [0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
}
BASELINESIQSD = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'T': T, 'M': M, 'P': P, 'LAM': LAM, 'Q_AGG': Q_AGG, 'I_SD': I_SD, 'K': K
},
    'PARAMS_SWEEP': {  # variable parameters (sweep)
        'Q_SD': [0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
}
BASELINESIISD = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'T': T, 'M': M, 'P': P, 'LAM': LAM, 'Q_AGG': Q_AGG, 'Q_SD': Q_SD, 'K': K
},
    'PARAMS_SWEEP': {  # variable parameters (sweep)
        'I_SD': [0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
}
BASELINESIP = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'T': T, 'M': M, 'LAM': LAM, 'Q_AGG': Q_AGG, 'Q_SD': Q_SD, 'I_SD': I_SD, 'K': K
},
    'PARAMS_SWEEP': {  # variable parameters (sweep)
        'P': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
}
BASELINESIT = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'P': P, 'M': M, 'LAM': LAM, 'Q_AGG': Q_AGG, 'Q_SD': Q_SD, 'I_SD': I_SD, 'K': K
},
    'PARAMS_SWEEP': {  # variable parameters (sweep)
        'T': [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }
}
BASELINESIM = {'PARAMS_FIX': {  # fixed parameters
    'N': N, 'S': S, 'G': G, 'T': T, 'P': P, 'LAM': LAM, 'Q_AGG': Q_AGG, 'Q_SD': Q_SD, 'I_SD': I_SD, 'K': K
},
    'PARAMS_SWEEP': {  # variable parameters (sweep)
        'M': UNITINTERVAL
    }
}
SIMODELS = {    # models that should be run only for SI
    # note: format is {name: variant}
    'extensioncustom': 'extension',  # multiple k in same environment (extension class, baseline params)
    'baselinesicustom': 'baseline',  # relationship lambda, i, p
    'baselinesipi': 'baseline',  # relationship i, p given lambda=0.5
    'baselinesipilam': 'baseline',  # relationship i, p given lambda=i
    'baselinesiplam': 'baseline',  # relationship lambda, p given i=lambda
    'baselinesiklam': 'baseline',  # stepwise k (.5) given i=p=lambda
    'baselinesiqagg': 'baseline',  # q given i=p=lambda=0.5
    'baselinesiisd': 'baseline',  # i_sd given i=p=lambda=0.5
    'baselinesiqsd': 'baseline',  # q_sd given i=p=lambda=0.5
    'baselinesip': 'baseline',  # P interaction frequency given i=p=lambda=0.5
    'baselinesit': 'baseline',  # T affected by loyalty trials given i=p=lambda=0.5
    'baselinesim': 'baseline',  # M probability to mutate given i=p=lambda=0.5
}

# TRACKER PARAMETERS
VARS_TRACK = {'tags_i': True,
              'tags_p': True,
              'tags_error': True,
              'tags_q': True,
              'labeling': True,
              'tolerated': True,
              'r': True,
              'allegiance': True,
              'cohesion': True,
              'truedef': True,
              'falsedef': True,
              'secretdef': True,
              'trueconf': True}
VARS_TARGET = ['trueconf', 'secretdef', 'falsedef', 'truedef', 'r', 'labeling', 'allegiance']  # store as Excel/LaTex
VARS_VIS = [  # visualize for analysis and store on disk if SAVE_SIMULATIONS is True
    {'res': 'tags_i', 'option': 'ts', 'ylim': (0, 1), 'label': 'Private Behavior',
     'colors': [('black', 'grey')]},
    {'res': 'tags_p', 'option': 'ts', 'ylim': (0, 1), 'label': 'Public Behavior',
     'colors': [('black', 'grey')]},
    {'res': 'tags_error', 'option': 'ts', 'ylim': (-1, 1), 'label': 'Deception (Mean p - i)',
     'colors': [('black', 'grey')]},
    {'res': 'tags_q', 'option': 'ts', 'ylim': (0, 1), 'label': 'Tolerance (Mean)',
     'colors': [('black', 'grey')]},
    {'res': 'tags_q', 'option': 'hist', 'label': 'Tolerance',
     'colors': ['black', 'grey']},
    {'res': ['tags_i', 'tags_p'], 'option': 'ts_combine', 'ylim': (0, 1),
     'label': 'Public/Private Behavior', 'colors': [('black', 'grey'), ('red', 'sienna')]},
    {'res': 'labeling', 'option': 'ts', 'ylim': (0, 100), 'label': 'Labeled Agents (%)',
     'colors': [('black', 'grey')]},
    {'res': 'tolerated', 'option': 'ts', 'ylim': (0, 100), 'label': 'Tolerated Agents (%)',
     'colors': [('black', 'grey')]},
    {'res': 'r', 'option': 'ts', 'label': 'Misidentification (Labeled over Private Defectors)', 'ylim': (0, N),
     'colors': [('black', 'grey')]},
    {'res': 'allegiance', 'option': 'ts', 'label': r'Group Conformity ($\Delta_\lambda$)', 'ylim': (0, N),
     'colors': [('black', 'grey')]},
    {'res': 'cohesion', 'option': 'ts', 'label': r'Cohesion ($\sigma_i)$', 'ylim': (0, 1),
     'colors': [('black', 'grey')]},
    {'res': ['truedef', 'falsedef', 'secretdef', 'trueconf'], 'option': 'ts_combine', 'ylim': None,
     'label': 'Defector Types',
     'colors': [('sienna', None), ('red', None), ('black', None), ('grey', None)]},
    {'res': 'truedef', 'option': 'ts', 'ylim': (0, N), 'label': 'Defectors',
     'colors': [('black', 'grey')]},
    {'res': 'falsedef', 'option': 'ts', 'ylim': (0, N), 'label': 'False Defectors',
     'colors': [('black', 'grey')]},
    {'res': 'secretdef', 'option': 'ts', 'ylim': (0, N), 'label': 'Secret Defectors',
     'colors': [('black', 'grey')]},
    {'res': 'trueconf', 'option': 'ts', 'ylim': (0, N), 'label': 'Conformers',
     'colors': [('black', 'grey')]}
]
