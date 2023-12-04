""" Simulation
Core PhD ABM logic that sets up initial system state and runs simulations.
Should be called from an instance of Model.
"""

# DEPENDENCIES
import statistics
import abm.environment as environment
import abm.agent as agent
import abm.assistant.tracker as tracker


class Simulation:
    """
    Setup and simulate model.

    Attribute
    ----------
    logger : logging.Logger
    model : str
        Name of the model variant.
    experiment : int
        Identifier of the experiment.
    identifier : int
        Identifier of the simulation.
    vars_track : Dict
        Key-value pairs indicating which outcomes should be tracked.
    params : Dict
        Key-value pairs of model parameters.
    """

    def __init__(self, logger, model, experiment, identifier, vars_track, params):
        self.model = model
        self.experiment = experiment
        self.identifier = identifier
        self.params = params
        self.env = None
        logger.info("Model %s: initializing simulation/experiment %d/%d", self.model, self.identifier, self.experiment)

        # environment and results
        self.results = dict.fromkeys((key for key in vars_track.keys()))
        self.res_truedef = None
        self.res_falsedef = None
        self.res_secretdef = None
        self.res_trueconf = None
        self.res_shiftlabeling_intensity = 0
        self.res_shiftadapt_intensity = 0
        self.seed_environment(logger=logger)
        self.init_results(logger=logger)
        self.store_results(vars_track)

    def init_results(self, logger):
        """Initialize simulation results trackers."""
        for k in self.results.keys():
            self.results[k] = tracker.Tracker(var=k.capitalize(),
                                              experiment=self.experiment,
                                              simulation=self.identifier,
                                              params=self.params,
                                              model=self.model)
            logger.debug("Initialized key %s", str(k))

    def store_results(self, vars_track):
        """Store simulation results."""
        if vars_track['tags_i']:
            self.results['tags_i'].append([a.i for a in self.env.get_sorted()])
        if vars_track['tags_p']:
            self.results['tags_p'].append([a.p for a in self.env.get_sorted()])
        if vars_track['tags_error']:
            self.results['tags_error'].append(statistics.mean([a.p - a.i for a in self.env.get_sorted()]))
        if vars_track['tags_q']:
            self.results['tags_q'].append([a.q for a in self.env.get_sorted()])
        if vars_track['labeling']:
            res_labels = (sum([a.labeled for a in self.env.get_sorted()])
                          * 100) / self.params['N']
            self.results['labeling'].append(res_labels)
        if vars_track['tolerated']:
            res_tolerated = (sum([a.tolerated for a in self.env.get_sorted()])
                             * 100) / self.params['N']
            self.results['tolerated'].append(res_tolerated)
        if vars_track['truedef']:
            self.results['truedef'].append(self.res_truedef)
        if vars_track['falsedef']:
            self.results['falsedef'].append(self.res_falsedef)
        if vars_track['secretdef']:
            self.results['secretdef'].append(self.res_secretdef)
        if vars_track['trueconf']:
            self.results['trueconf'].append(self.res_trueconf)
        if vars_track['r']:
            if all([res is not None for res in [self.res_falsedef, self.res_secretdef, self.res_truedef]]):
                self.results['r'].append(((self.res_falsedef + self.res_truedef + 1) /
                                          (self.res_secretdef + self.res_truedef + 1)))
            else:
                self.results['r'].append(None)
        if vars_track['allegiance']:
            delta_lam = sum([(a.i - self.env.lam) / self.env.N * 100 for a in self.env.get_sorted()])
            self.results['allegiance'].append(delta_lam)
        if vars_track['cohesion']:
            self.results['cohesion'].append(1 - self.results['tags_i'].stdev()[-1])

    def seed_environment(self, logger):
        """Seed initial environment with agents."""
        self.env = environment.Environment(n=self.params['N'],
                                           lam=self.params['LAM'] if 'LAM' in self.params.keys()
                                           else self.params['I_AGG'] if 'I_AGG' in self.params.keys()
                                           else self.params['P_AGG'],
                                           k=self.params['K'] if 'K' in self.params.keys()
                                           else self.params['LAM'] if 'LAM' in self.params.keys()
                                           else self.params['I_AGG'] if 'I_AGG' in self.params.keys()
                                           else self.params['P_AGG'],
                                           i_agg=self.params['I_AGG'] if 'I_AGG' in self.params.keys()
                                           else self.params['P_AGG'] if 'P_AGG' in self.params.keys()
                                           else self.params['LAM'],
                                           i_sd=self.params['I_SD'] if 'I_SD' in self.params.keys()
                                           else self.params['P_SD'],
                                           p_agg=self.params['P_AGG'] if 'P_AGG' in self.params.keys()
                                           else self.params['I_AGG'] if 'I_AGG' in self.params.keys()
                                           else self.params['LAM'],
                                           p_sd=self.params['P_SD'] if 'P_SD' in self.params.keys()
                                           else self.params['I_SD'],
                                           q_agg=self.params['Q_AGG'],
                                           q_sd=self.params['Q_SD'])
        self.env.seed_agents()
        logger.debug("Initialized %d agents", len(self.env.agents))

    def mechanism_labeling(self, logger, run, agents):
        """Let agents label each other and update public allegiances."""
        for agent_b in agents:
            # interacting agent
            agent_a = self.env.get_interaction(agent=agent_b, agents=agents, nonlinear=False)
            logger.debug("Run %d: agent %d with q=%f, p=%f has agent %d as interaction partner",
                         run, agent_b.identifier, agent_a.q, agent_b.p, agent_a.identifier)

            # label
            delta = agent.Agent.label(agent_b=agent_b, agent_a=agent_a, lam=self.env.lam)
            if agent_a.labeled:
                logger.debug("Run %d: agent %d labeled agent %d with p = %f for delta %f",
                             run, agent_b.identifier, agent_a.identifier, agent_a.p, delta)
            else:
                logger.debug("Run %d: agent %d did not label agent %d with p = %f for delta %f",
                             run, agent_b.identifier, agent_a.identifier, agent_a.p, delta)
            agent_a.update_p(delta=delta)
            logger.debug("Run %d: agent %d updated public allegiance to %f", run, agent_a.identifier, agent_a.p)

    def mechanism_types(self, logger, run, agent_a):
        """Subject agents to loyalty trials to determine defector types."""
        if agent_a.labeled:
            if agent_a.label_true(lam=self.env.lam):
                self.res_truedef += 1
                logger.debug("Run %d: agent %d is a true defector with public tag %f and private tag %f, label %r",
                             run, agent_a.identifier, agent_a.p, agent_a.i, agent_a.labeled)
            else:
                self.res_falsedef += 1
                logger.debug("Run %d: agent %d is a false defector with public tag %f and private tag %f, label %r",
                             run, agent_a.identifier, agent_a.p, agent_a.i, agent_a.labeled)
        else:
            if agent_a.label_true(lam=self.env.lam):
                self.res_secretdef += 1
                logger.debug("Run %d: agent %d is a secret defector with public tag %f and private tag %f, label %r",
                             run, agent_a.identifier, agent_a.p, agent_a.i, agent_a.labeled)
            else:
                self.res_trueconf += 1
                logger.debug("Run %d: agent %d is a true conformer with public tag %f and private tag %f, label %r",
                             run, agent_a.identifier, agent_a.p, agent_a.i, agent_a.labeled)

    def mechanism_shift(self, logger, run, agent_a):
        """Update private allegiances."""
        pre_i = agent_a.i
        agent_a.update_i(lam=self.env.lam)
        logger.debug("Run %d: agent %d shifted private allegiance from %f to %f",
                     run, agent_a.identifier, pre_i, agent_a.i)
        self.res_shiftlabeling_intensity += agent_a.i - pre_i

    def mechanism_feedback(self, logger, run, agents):
        """Update meta-agent in environment class."""
        logger.debug("Run %d: begin feedback mechanism with tolerance %f", run, statistics.mean([a.q for a in agents]))
        change = self.env.feedback(falsedef=self.res_falsedef, truedef=self.res_truedef, agents=agents)
        logger.debug("Run %d: Tolerance updated by %f to: %f", run, change, statistics.mean([a.q for a in agents]))

    def mechanism_adaptation(self, logger, run, agent_a, agents):
        """Calculate payoffs and let agents adapt."""
        # interacting agent
        agent_b = self.env.get_interaction(agent=agent_a, agents=agents)

        # update payoffs
        for a in [agent_a, agent_b]:
            a.update_payoffs(lam=self.env.lam, k=self.env.k)
        logger.debug("Run %d: agent %d with i=%f,labeled=%r selected agent %d with i=%f,labeled=%r"
                     " as interaction partner. Their payoffs are: %f, %f",
                     run, agent_a.identifier, agent_a.i, agent_a.labeled,
                     agent_b.identifier, agent_b.i, agent_b.labeled,
                     agent_a.f, agent_b.f)

        # adapt
        prev_i = [agent_a.i, agent_b.i]
        a = agent.Agent.adapt(agent_a, agent_b)
        if a is not None:
            if agent_a.identifier == a.identifier:
                self.res_shiftadapt_intensity += a.i - prev_i[0]
            else:
                self.res_shiftadapt_intensity += a.i - prev_i[1]
            logger.debug("Run %d: agent %d adapted to i=%f, p=%f, q=%f", run, a.identifier, a.i, a.p, a.q)
        else:
            logger.debug("Run %d: agents %d and %d have identical payoffs, skipping adaptation",
                         run, agent_a.identifier, agent_b.identifier)

    def mechanism_mutation(self, logger, run, agent_a):
        """Mutate agents."""
        pre_mutation = [agent_a.i, agent_a.p, agent_a.q]
        mutated = agent_a.mutate(m=self.params['M'],
                                 i_agg=self.env.i_agg, i_sd=self.env.i_sd,
                                 p_agg=self.env.p_agg, p_sd=self.env.p_sd,
                                 q_agg=self.env.q_agg, q_sd=self.env.q_sd)
        post_mutation = [agent_a.i, agent_a.p, agent_a.q]
        for pos, m in enumerate(mutated):
            if m:
                logger.debug("Run %d: agent %d mutated %s from %f to %f",
                             run, agent_a.identifier, ['i', 'p', 'q'][pos], pre_mutation[pos], post_mutation[pos])

    def run(self, logger, vars_track):
        """Run initialized simulation."""
        # run simulation
        logger.debug("Running %d iterations...", self.params['G'])
        for run in range(1, self.params['G'] + 1):
            logger.debug("Begin iteration: %d", run)
            # reset run trackers
            self.res_truedef = 0
            self.res_falsedef = 0
            self.res_secretdef = 0
            self.res_trueconf = 0
            self.res_shiftlabeling_intensity = 0
            self.res_shiftadapt_intensity = 0

            # loyalty trials
            agents = self.env.get_random(percent=self.params['T'])
            logger.debug("Run %d: loyalty trials with %d agents: %s",
                         run, len(agents), [a.identifier for a in agents])
            for interaction in range(1, self.params['P'] + 1):
                logger.debug("Run %d: begin agent interaction round %d/%d", run, interaction, self.params['P'])
                self.mechanism_labeling(logger=logger, run=run, agents=agents)
            # defector types
            logger.debug("Run %d: determining defector types", run)
            for agent_a in self.env.get_sorted():
                self.mechanism_types(logger=logger, run=run, agent_a=agent_a)

            # private allegiance updates
            agents_labeled = [a for a in agents if a.labeled]
            if len(agents_labeled) > 0:
                logger.debug("Run %d: updating private allegiance for %d/%d labeled agents",
                             run, len(agents_labeled), len(agents))
                for agent_a in agents_labeled:
                    self.mechanism_shift(logger=logger, run=run, agent_a=agent_a)
            else:
                logger.debug("Run %d: no agents were labeled, skipping private allegiance updates", run)

            # feedback
            self.mechanism_feedback(logger=logger, run=run, agents=agents)

            # adaptation and mutation
            agents = self.env.get_random(percent=self.params['T'])
            logger.debug("Run %d: adaptation and mutation of agents: %s", run, [a.identifier for a in agents])
            for agent_a in agents:
                self.mechanism_adaptation(logger=logger, run=run, agent_a=agent_a, agents=agents)
            for agent_a in agents:
                self.mechanism_mutation(logger=logger, run=run, agent_a=agent_a)

            # store results of run
            self.store_results(vars_track)
            logger.debug("Completed iteration %d/%d\n"
                         "Mean agent characteristics:\n"
                         "       Previous i, p, q: %f %f %f\n"
                         "       Updated i, p, q: %f %f %f\n"
                         "Types:\n"
                         "       True Conformers: %d\n"
                         "       Secret Defectors: %d\n"
                         "       False Defectors: %d\n"
                         "       True Defectors: %d\n"
                         "Cumulative private allegiance change by mechanism:\n"
                         "       Labeling: %f"
                         "       Adaptation: %f",
                         run, self.params['G'],
                         statistics.mean(self.results['tags_i'].values[-2]),
                         statistics.mean(self.results['tags_p'].values[-2]),
                         statistics.mean(self.results['tags_q'].values[-2]),
                         statistics.mean(self.results['tags_i'].values[-1]),
                         statistics.mean(self.results['tags_p'].values[-1]),
                         statistics.mean(self.results['tags_q'].values[-1]),
                         self.env.N - sum([self.res_secretdef, self.res_falsedef, self.res_truedef]),
                         self.res_secretdef, self.res_falsedef, self.res_truedef,
                         self.res_shiftlabeling_intensity, self.res_shiftadapt_intensity)
        logger.info("Completed simulation.\n"
                    "Defector types (percent):\n"
                    "       True Conformers: %f\n"
                    "       Secret Defectors: %f\n"
                    "       False Defectors: %f\n"
                    "       True Defectors: %f",
                    self.results['trueconf'].percent(), self.results['secretdef'].percent(),
                    self.results['falsedef'].percent(), self.results['truedef'].percent())
        return self.results
