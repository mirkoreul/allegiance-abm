""" Environment
"""

# DEPENDENCIES
import random
import numpy
import abm.environment
import abm.agent


class EnvironmentExtension(abm.environment.Environment):
    """
    Environment extension for more specific applications of the model.

    Attribute
    ----------
    p_shift: float
        Agent-level difference between private and public allegiance (i + p_shift).
    """

    def __init__(self, n, lam, k, i_agg, i_sd, p_agg, p_sd, q_agg, q_sd, p_shift):
        super().__init__(n, lam, k, i_agg, i_sd, p_agg, p_sd, q_agg, q_sd)
        self.p_shift = p_shift

    @property
    def k(self):
        if isinstance(self.__k, list):
            return random.choice(self.__k)
        else:
            return self.__k

    @k.setter
    def k(self, k):
        self.__k = k

    def seed_agents(self, agent_dist=None):
        """
        Seed environment with agents, optionally with a specific distribution for parameter i.

        Parameters are drawn from the normal distribution.
        The 'p' attribute is seeded differently than in the baseline:
        First, we draw the 'i' attribute. Second, the mean for 'p' is based on the sum of the drawn 'i' and 'p_shift'.
        This ensures that values for 'p' have, on average, the desired distance to 'i' at the agent-level.

        Parameters
        ----------
        agent_dist : Tuple<Tuple<float>, Tuple(Dict<str, float>)
            Nested tuple with length 2, with outer elements prescribing:
                1) proportions of agents to seed
                2) corresponding to order of agent proportions, key-value pairs defining the parameters (keys)
                that should be drawn from a normal distribution with a given mean (values).
            The parameter defined in the inner dictionary key may be either of: 'i_agg', 'p_shift', 'q_agg'.
            All undefined agent parameters follow the global definition.
            This option allows for an overall non-normal distribution of agent allegiance in the (0, 1)-range.
            Example:
                (
                    (0.5, 0.2), # agent proportions
                    ({'i_agg': 0.2, 'p_shift': 0.2, 'q_agg': 0.1}, # 50% of agents have mean i=0.2, p=0.2+0.2, q=0.1
                     {'i_agg: 0.4}) # 20% of agents have mean i=0.4, and other are parameters seeded with global values
                )

        Returns
        -------
        None
        """
        if agent_dist is None:
            for i in range(0, self.N):
                tagi = numpy.random.normal(loc=self.i_agg, scale=self.i_sd, size=1)[0]
                self.agents.append(abm.agent.Agent(
                    identifier=i,
                    i=tagi,
                    p=numpy.random.normal(loc=tagi + self.p_shift, scale=self.p_sd, size=1)[0],
                    q=numpy.random.normal(loc=self.q_agg, scale=self.q_sd, size=1)[0]
                ))
        else:
            if not sum(agent_dist[0]) == 1:
                raise ValueError("Misspecified agent distribution: proportions of agents must sum to 1")
            for prop, dist in enumerate(agent_dist[1]):
                for i in range(0, int(round(agent_dist[0][prop]*self.N))):
                    if 'i_agg' in dist.keys():
                        tagi = numpy.random.normal(loc=dist['i_agg'], scale=self.i_sd, size=1)[0]
                    else:
                        tagi = numpy.random.normal(loc=self.i_agg, scale=self.i_sd, size=1)[0]
                    if 'p_shift' in dist.keys():
                        tagp = numpy.random.normal(loc=tagi + dist['p_shift'], scale=self.p_sd, size=1)[0]
                    else:
                        tagp = numpy.random.normal(loc=tagi + self.p_shift, scale=self.p_sd, size=1)[0]
                    if 'q_agg' in dist.keys():
                        tagq = numpy.random.normal(loc=dist['q_agg'], scale=self.q_sd, size=1)[0]
                    else:
                        tagq = numpy.random.normal(loc=self.q_agg, scale=self.q_sd, size=1)[0]
                    self.agents.append(abm.agent.Agent(
                        identifier=i,
                        i=tagi,
                        p=tagp,
                        q=tagq
                    ))
