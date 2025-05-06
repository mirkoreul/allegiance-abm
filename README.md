# Overview
Reproduce results for the paper: 
[`How Loyalty Trials Shape Allegiance to Political Order`](https://doi.org/10.1177/00220027231222004).

# Reproduce Results
## Linux
Execute the following steps in a terminal:
1. Clone this repository: `git clone https://github.com/mirkoreul/allegiance-abm.git ABM`
2. Navigate to the cloned repository: `cd ./ABM`
3. Setup and enter a Python virtual environment: 
   - `python3 -m venv venv`
   - `source ./venv/bin/activate`
4. Install Python dependencies and abm package: `pip install .`
5. (Recommended) Install [GNU Parallel]([https://www.gnu.org/software/parallel/]): `sudo apt-get install parallel`
6. Run the model: 
   - (Recommended) With parallel processing, for a full parameter sweep: `./bin/run-parallel 1 121`
   - Without parallel processing: `python3 -m 'abm'`

System and parameter settings can be modified in `abm/constants.py`.
By default all results given in the paper are reproduced, excluding the online appendix.

# Package Structure
- abm/\_\_init__.py: entry-point to initialize logger and models.
- abm/\_\_main__.py: run to reproduce results for all model variants.
- abm/constants.py: system and model parameter settings.
- abm/model.py: topmost logic to run models and store results.
- abm/simulation.py: core logic where model mechanisms are executed.
- abm/environment.py: spatial topography and meta-agent.
- abm/agent.py: agent characteristics and behavior.
- abm/assistant/functions.py: helper functions.
- abm/assistant/tracker.py: helper class to track and visualize outcomes.
- abm/variants/environment_extension.py: ABM spatial topography and meta-agent for contextualized models.
- abm/variants/simulation_extension.py: ABM simulation for contextualized models.
- bin/run-parallel: wrapper for GNU Parallel, run to reproduce results using all CPU cores.

# Citation
```
@article{Reul2023,
  title = {How {{Loyalty Trials Shape Allegiance}} to {{Political Order}}},
  author = {Reul, Mirko and Bhavnani, Ravi},
  year = {2023},
  journal = {Journal of Conflict Resolution},
  volume = {69},
  number = {1},
  pages = {178--206},
  doi = {10.1177/00220027231222004},
}
```

