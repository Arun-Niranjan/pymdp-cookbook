"""
In this example, we show how to encode long temporal range dependencies.

The definition of a Markov Decision Process is that everything about
the current state can be inferred from the previous one.

However there are plenty of problems where there is a sparse signal
that warns of something coming in the future. Encoding this in a pymdp
Agent is non-trivial, but this example will show how it could be done.

We start by considering a binary event our agent is looking forward to.
Let's say it's payday, and we want our agent to be able to "count down"
the days.

We can transform the long range temporal dependency into a short range
factorial dependency, by having a set of states that correspond to the
"countdown"

TODO:
    - modify observation and state spaces to:
        - the value to represent the delay (in discrete timesteps) for the event to happen
        - the transition of the value to always decrease by 1 "unit" per timestep
    - check with Conor etc. about generalised free energy here.
      It would be really nice to have an observation belief vector here

"""

from rich import print
from pymdp.agent import Agent
from pymdp.distribution import compile_model
import numpy as np

model_specification = {
    "controls": {
        "null": {"elements": ["null"]},
        # "save": {"elements": ["false", "true"]},
    },
    "states": {
        "time_to_payday": {
            "elements": [
                "0",
                "1",
                "2",
                "3",
                "unknown",
            ],
            "depends_on": ["time_to_payday"],
            "controlled_by": ["null"],
        },
        "savings_level": {
            "elements": [
                "low",
                "high",
            ],
            "depends_on": ["savings_level"],
            "controlled_by": ["save"],
        },
    },
    "observations": {
        "paid": {
            "elements": ["false", "true"],
            "depends_on": ["time_to_payday"],
        },
        "savings_level": {
            "elements": ["low", "high"],
            "depends_on": ["savings_level"],
        },
        # "time_to_payday": {
        #     "elements": [
        #         "0",
        #         "1",
        #         "2",
        #         "3",
        #     ],
        #     "depends_on": ["time_to_payday"],
        # }
    }
}

model = compile_model(model_specification)

# Perfect observation-state mapping
model.A["paid"]["true", "0"] = 1.0
model.A["paid"]["false"][1:] = 1.0

model.A["savings_level"].data = np.eye(
    len(model_specification["states"]["savings_level"]["elements"]),
    len(model_specification["observations"]["savings_level"]["elements"]),
)

# model.A["time_to_payday"].data = np.eye(
#     len(model_specification["states"]["time_to_payday"]["elements"]),
#     len(model_specification["observations"]["time_to_payday"]["elements"]),
# )

# We now set up the temporal state dynamics
# Initialise B tensor (hidden state transitions)
# B[f][s_t+1, s_t, u_t] stores the probability:
    #  for factor f (hidden state variable)
    #   of hidden state level s_t+1
    #   given:
    #       hidden state level s_t
    #       action u_t
model.B["time_to_payday"]["unknown", "unknown", "null"] = 1.0
model.B["time_to_payday"]["unknown", "0", "null"] = 1.0
model.B["time_to_payday"]["0", "1", "null"] = 1.0
model.B["time_to_payday"]["1", "2", "null"] = 1.0
model.B["time_to_payday"]["2", "3", "null"] = 1.0

# Set a flat prior on preference of getting paid
model.C["paid"]["false"] = 0.5
model.C["paid"]["true"] = 0.5
print(model)

# Instantiate a pymdp Agent from the generative model
agent = Agent(
    **model,
    policy_len=5,
    # inductive_depth=5,
    use_inductive=True,
    apply_batch=True,
    learn_A=False,
    learn_B=False,
    learn_D=False,
)

empirical_prior = [np.array([[0.0, 0.0, 0.1, 0.8, 0.1]])]
qs = agent.infer_states(observations=[np.array([[0]])],
    empirical_prior=empirical_prior,
)
q_pi, G = agent.infer_policies(qs)
