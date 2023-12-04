""" Agent
Specification of agent characteristics and behavior.
"""

# DEPENDENCIES
import random
import math


class Agent:
    """
    Agent characteristics and behavior.

    Specify either all group-level parameters with suffix 'agg' to draw initial agent characteristics
    from a normal distribution, or specify all initial characteristics i, p, q.

    Attribute
    ----------
    identifier : int
        Unique ID of the agent.
    q_agg : float
        Group-level tolerance threshold.
        Used to draw initial value from distribution.
    q_sd : float
        Group-level standard deviation of tolerance threshold.
        Used to draw initial value from distribution.
    i_agg : float
        Group-level interactions with outgroup.
        Used to draw initial value from distribution.
    i_sd : float
        Group-level standard deviation of interactions with outgroup.
        Used to draw initial value from distribution.
    p_agg : float
        Group-level (public) interactions with outgroup.
        Used to draw initial value from distribution.
    p_sd : float
        Group-level (public) standard deviation of interactions with outgroup.
        Used to draw initial value from distribution.
    i : float
        Private allegiance, interaction of agent with outgroup.
    p : float
        Public perception of agent's allegiance.
    q : float
        Agent tolerance threshold.
    f : float
        Agent fitness score.
    labeled : int or bool
        Whether the agent is labeled.
    labeled_stick : int or bool
        Whether an agent label stuck in a generation.
    """

    def __init__(self, identifier,
                 i_agg=None, i_sd=None, p_agg=None, p_sd=None, q_agg=None, q_sd=None,
                 i=None, p=None, q=None):
        self.identifier = identifier
        self.f = None
        self.labeled = 0
        self.tolerated = 0
        self.labeled_stick = None
        if any([param is None for param in [i, p, q]]):
            self.i, self.p, self.q = self.seed_agent_properties(i_agg, i_sd, p_agg, p_sd, q_agg, q_sd)
        else:
            self.i, self.p, self.q = i, p, q

    def __repr__(self):
        return 'Agent ' + str(self.identifier) + ': ' + str(round(self.i, 2))

    @property
    def i(self):
        return self.__i

    @i.setter
    def i(self, i):
        if i is not None:
            self.__i = i
            self.__i = 0 if self.i < 0 else self.__i
            self.__i = 1 if self.i > 1 else self.__i

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, p):
        if p is not None:
            self.__p = p
            self.__p = 0 if self.p < 0 else self.__p
            self.__p = 1 if self.p > 1 else self.__p

    @property
    def q(self):
        return self.__q

    @q.setter
    def q(self, q):
        if q is not None:
            self.__q = q
            self.__q = 0 if self.__q < 0 else self.__q
            self.__q = 1 if self.__q > 1 else self.__q

    @property
    def f(self):
        return self.__f

    @f.setter
    def f(self, f):
        self.__f = f

    def label_true(self, lam):
        """Determines if a label sticks for a generation."""
        if self.i < lam:
            self.labeled_stick = 1
            return True
        else:
            self.labeled_stick = 0
            return False

    def update_p(self, delta):
        """Updates public allegiance."""
        self.p = self.p - (self.p * delta)
        return self.p

    def update_i(self, lam):
        """Updates private allegiance."""
        delta = lam - self.i
        self.i = self.i - (self.i * delta)
        return self.i

    def update_q(self, change):
        """Updates tolerance."""
        self.q += change
        return self.q

    def update_payoffs(self, lam, k):
        """Update agent payoff value f."""
        delta = lam - self.i
        self.f = (delta ** 2) / (math.exp(k * delta)) - abs(self.p - self.i) * self.labeled

    def mutate(self, m, i_agg, i_sd, p_agg, p_sd, q_agg, q_sd):
        """Mutate agent characteristics."""
        mutated = [False, False, False]
        if random.uniform(0, 1) < m:
            self.i = self.seed_agent_properties(i_agg=i_agg, i_sd=i_sd,
                                                p_agg=None, p_sd=None,
                                                q_agg=None, q_sd=None)[0]
            mutated[0] = True
        if random.uniform(0, 1) < m:
            self.p = self.seed_agent_properties(i_agg=None, i_sd=None,
                                                p_agg=p_agg, p_sd=p_sd,
                                                q_agg=None, q_sd=None)[1]
            mutated[1] = True
        if random.uniform(0, 1) < m:
            self.q += self.seed_agent_properties(i_agg=None, i_sd=None,
                                                 p_agg=None, p_sd=None,
                                                 q_agg=q_agg, q_sd=q_sd)[2]
            mutated[2] = True
        return mutated

    @staticmethod
    def seed_agent_properties(i_agg, i_sd, p_agg, p_sd, q_agg, q_sd):
        """
        Helper method to draw agent properties from a distribution.
        Returns new properties in List<float> format in order [i, p, q], or NoneType if property is not given.
        """
        attributes = [None, None, None]
        if i_agg is not None and i_sd is not None:
            attributes[0] = random.normalvariate(i_agg, i_sd)
        if p_agg is not None and p_sd is not None:
            attributes[1] = random.normalvariate(p_agg, p_sd)
        if q_agg is not None and q_sd is not None:
            attributes[2] = random.normalvariate(q_agg, q_sd)
        return attributes

    @staticmethod
    def label(agent_b, agent_a, lam):
        """Determine if labeling should occur, and let agent_a label agent_b."""
        delta = lam - agent_a.p
        if delta > agent_b.q:
            agent_a.labeled = 1
            agent_a.tolerated = 0
        else:
            agent_a.labeled = 0
            if agent_a.p < lam:
                agent_a.tolerated = 1
        return delta

    @staticmethod
    def adapt(agent_a, agent_b):
        """Let agent with lower payoffs adopt the characteristics of the other agent."""
        if agent_a.f < agent_b.f:
            agent_a.i = agent_b.i
            agent_a.p = agent_b.p
            agent_a.q = agent_b.q
            return agent_a
        elif agent_a.f > agent_b.f:
            agent_b.i = agent_a.i
            agent_b.p = agent_a.p
            agent_b.q = agent_a.q
            return agent_b
        else:
            return None
