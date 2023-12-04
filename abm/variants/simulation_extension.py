""" Simulation
Core PhD ABM logic that sets up initial system state and runs simulations.
Should be called from an instance of Model.

This class defines changes to the baseline model setup.
Refer to the superclass for the baseline model.
"""

# DEPENDENCIES
import abm.simulation
import abm.variants.environment_extension as environment


class SimulationExtension(abm.simulation.Simulation):
    def seed_environment(self, logger):
        """Seed initial environment with agents."""
        self.env = environment.EnvironmentExtension(n=self.params['N'],
                                                    lam=self.params['LAM'],
                                                    k=self.params['K'],
                                                    i_agg=self.params['I_AGG'],
                                                    i_sd=self.params['I_SD'],
                                                    p_agg=None,
                                                    p_sd=self.params['P_SD'],
                                                    q_agg=self.params['Q_AGG'],
                                                    q_sd=self.params['Q_SD'],
                                                    p_shift=self.params['P_SHIFT']
                                                    )
        self.env.seed_agents(agent_dist=self.params['DIST']) if 'DIST' in self.params.keys() else self.env.seed_agents()
        logger.debug("Initialized %d agents", self.env.N)

    def mechanism_mutation(self, logger, run, agent_a):
        """Mutate agents."""
        pre_mutation = [agent_a.i, agent_a.p, agent_a.q]
        mutated = agent_a.mutate(m=self.params['M'],
                                 i_agg=self.env.i_agg, i_sd=self.env.i_sd,
                                 p_agg=self.env.i_agg + self.env.p_shift, p_sd=self.env.p_sd,
                                 q_agg=self.env.q_agg, q_sd=self.env.q_sd)
        post_mutation = [agent_a.i, agent_a.p, agent_a.q]
        for pos, m in enumerate(mutated):
            if m:
                logger.debug("Run %d: agent %d mutated %s from %f to %f",
                             run, agent_a.identifier, ['i', 'p', 'q'][pos], pre_mutation[pos], post_mutation[pos])
