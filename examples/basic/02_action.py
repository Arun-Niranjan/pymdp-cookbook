"""
This example demonstrates the combination of perception and action.

Here we set up a generative model for an Active Inference Thermostat.
The thermostat:
    - observes the temperature of the room and perfectly infers the real temperature (as a hidden state).
    - has controls for a radiator which can increase the temperature of the room by a varying degree.
    - has a prior preference for a middling temperature, but would rather be too hot than too cold.
"""

from pymdp.agent import Agent
from pymdp.distribution import compile_model
import numpy as np

model_specification = {
    "controls": { # These are things the Agent *believes* it can do to affect the environment
        "radiator": {"elements": ["off", "on"]},
    },
    "states": {
        "temperature": {
            "elements": ["low", "medium", "high"],
            "depends_on": ["temperature"],
            "controlled_by": ["radiator"],
        },
    },
    "observations": {
        "temperature": {
            "elements": ["low", "medium", "high"],
            "depends_on": ["temperature"]
        }
    }
}

model = compile_model(model_specification)

# Let's give our thermostat a perfect thermometer (i.e the likelihood matrix is an Identity matrix)
model.A["temperature"].data = np.eye( # Pull out the A matrix for the hidden state "temperature"
    len(model_specification["states"]["temperature"]["elements"]), # Number of hidden state *values*
    len(model_specification["observations"]["temperature"]["elements"]), # Number of observation *values*
)

# Let's define how our thermostat believes temperature will evolve from one timestep to the next
# We'll set it up so:
    # if the radiator is off, the temperature will drop by one "unit" per timestep
    # if the radiator is on, the temperature will increase by one "unit" per timestep
    # subject to limits on the lowest and highest values for temperature
# Initialise B tensor (hidden state transitions)
# B[f][s_t+1, s_t, u_t] stores the probability:
    #  for factor f (hidden state variable)
    #   of hidden state level s_t+1
    #   given:
    #       hidden state level s_t
    #       action u_t
model.B["temperature"]["low",       "low",      "off"]  = 1.0
model.B["temperature"]["medium",    "low",      "on"]   = 1.0

model.B["temperature"]["low",       "medium",   "off"]  = 1.0
model.B["temperature"]["high",      "medium",   "on"]   = 1.0

model.B["temperature"]["medium",    "high",     "off"]  = 1.0
model.B["temperature"]["high",      "high",     "on"]   = 1.0

# This is easier to visualise by selecting the factor and action,
# and then looking at the state mapping
# >>> print(model.B["temperature"][:,:,"off"])
# [[1. 1. 0.]
#  [0. 0. 1.]
#  [0. 0. 0.]]
# >>> print(model.B["temperature"][:,:,"on"])
# [[0. 0. 0.]
#  [1. 0. 0.]
#  [0. 1. 1.]]
# For a properly normalised B tensor, each of the columns in the matrix above
# should sum to 1.0. Why? Because a column corresponds to B[f][:,s_t,u_t]

# C tensor is observation preference mapping (reward prior)
model.C["temperature"]["low"] = 0.1
model.C["temperature"]["medium"] = 0.7
model.C["temperature"]["high"] = 0.2

# D tensor defaults to flat prior on hidden states, so we don't need to set that
# >>> print(model.D["temperature"].data)
# [0.33333333 0.33333333 0.33333333]

# Instantiate a pymdp Agent from the generative model
agent = Agent(
    **model,
    apply_batch=True,
    learn_A=False,
    learn_B=False,
    learn_D=False,
)

# Now let's set up our first perception/action iteration
observation = np.array([[0]]) # Observe the room is cold
qs = agent.infer_states(
    observations=[observation],
    empirical_prior=agent.D,
)
# >>> print(qs)
# [Array([[[1.00000e+00, 2.22045e-16, 2.22045e-16]]], dtype=float32)]

# Now let's calculate the expected free energy G and the posterior distribution over policies (q_pi)
q_pi, G = agent.infer_policies(qs)
# >>> print(q_pi) # q_pi is just a softmax over G
# [[0.3543437 0.6456563]]

# We can now sample an action from q_pi
action = agent.sample_action(q_pi)
# >>> print(action)
# [[1]] # corresponds to turning the radiator on
