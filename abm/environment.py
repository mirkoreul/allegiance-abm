""" Environment
Specification of spatial topography and meta-agent.
"""

# DEPENDENCIES
import random
import numpy.random
import abm.agent
import abm.assistant.functions as aux


class Environment:
    """
    Spatial topography populated with agents and meta-agent.

    Attribute
    ----------
    n : int
        Number of agents.
    lam : float
        Loyalty expectations.
    i_agg : float
        Group-level interactions with outgroup.
    i_sd : float
        Group-level standard deviation of interactions with outgroup.
    p_agg : float
        Group-level (public) interactions with outgroup.
    p_sd : float
        Group-level (public) standard deviation of interactions with outgroup.
    q_agg : float
        Group-level tolerance threshold.
    q_sd : float
        Group-level standard deviation of tolerance threshold.
    """
    def __init__(self, n, lam, k, i_agg, i_sd, p_agg, p_sd, q_agg, q_sd):
        self.lam = lam
        self.k = k
        self.N = n
        self.agents = []
        self.i_agg = i_agg
        self.i_sd = i_sd
        self.p_agg = p_agg
        self.p_sd = p_sd
        self.q_agg = q_agg
        self.q_sd = q_sd

    @property
    def lam(self):
        return self.__lam

    @lam.setter
    def lam(self, lam):
        self.__lam = lam
        self.__lam = 0 if self.__lam < 0 else self.__lam
        self.__lam = 1 if self.__lam > 1 else self.__lam

    def seed_agents(self):
        """Populate environment with agents."""
        for i in range(0, self.N):
            self.agents.append(abm.agent.Agent(
                identifier=i,
                i_agg=self.i_agg,
                i_sd=self.i_sd,
                p_agg=self.p_agg,
                p_sd=self.p_sd,
                q_agg=self.q_agg,
                q_sd=self.q_sd
            ))

    def feedback(self, falsedef, truedef, agents):
        """Update tolerance for agents in environment."""
        change = ((falsedef - truedef) * (1 / self.N))
        if change != 0:
            for a in agents:
                a.update_q(change)
        return change

    def get_sorted(self):
        """Get list of agents sorted by identifier. Does not modify original list of agents."""
        return sorted(self.agents, key=lambda a: a.identifier)

    def get_random(self, percent=None):
        """
        Shuffle agents in environment and return as list.

        The percent parameter optionally subsets the list of agents.
        """
        random.shuffle(self.agents)
        if percent is None:
            return self.agents
        else:
            slce = max(round((percent / 100) * self.N), 2)
            return self.agents[:slce]

    def get_interaction(self, agent, agents=None, nonlinear=False, n=None):
        """Randomly select one interacting agent who is not the given agent, optionally with nonlinear probability."""
        if agents is None:
            others = [a for a in self.agents if a.identifier is not agent.identifier]
        else:
            others = [a for a in agents if a.identifier is not agent.identifier]
        if nonlinear:
            probabilities = aux.softmax([self.k*(self.lam-a.p) for a in others])
        else:
            probabilities = [1/len(others) for _ in others]
        return numpy.random.choice(others, size=n, p=probabilities, replace=True)
