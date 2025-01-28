"""
In this example, we will set up an action-perception loop.
This requires that we set up an `environment` for our Agent to interact.
We will use a similar thermostat model we used in the action example, but extend it to:
    - have a finer grained understanding of temperature
    - have policies (sequences of more than one action) to demonstrate probabilistic planning

The environment must take an action as input and return an observation.
It does *not* need to use the same generative model as the Agent!
In fact we should distinguish between the environment having a generative *process*
and the Agent has a generative *model* _of_ that process.
"""

from dataclasses import dataclass
from pymdp.agent import Agent
from pymdp.distribution import compile_model
import numpy as np

model_specification = {
    "controls": { # These are things the Agent *believes* it can do to affect the environment
        "radiator": {"elements": ["off", "on"]},
    },
    "states": {
        "temperature": {
            "elements": ["very_low", "low", "medium", "high", "very_high"],
            "depends_on": ["temperature"],
            "controlled_by": ["radiator"],
        },
    },
    "observations": {
        "temperature": {
            "elements": ["very_low", "low", "medium", "high", "very_high"],
            "depends_on": ["temperature"]
        }
    }
}

model = compile_model(model_specification)

# Perfect thermometer
model.A["temperature"].data = np.eye( # Pull out the A matrix for the hidden state "temperature"
    len(model_specification["states"]["temperature"]["elements"]), # Number of hidden state *values*
    len(model_specification["observations"]["temperature"]["elements"]), # Number of observation *values*
)

# Initialise B tensor (hidden state transitions)
# B[f][s_t+1, s_t, u_t] stores the probability:
    #  for factor f (hidden state variable)
    #   of hidden state level s_t+1
    #   given:
    #       hidden state level s_t
    #       action u_t

model.B["temperature"]["very_low",  "very_low",     "off"]  = 1.0
model.B["temperature"]["low",       "very_low",     "on"]   = 1.0

model.B["temperature"]["very_low",  "low",          "off"]  = 1.0
model.B["temperature"]["medium",    "low",          "on"]   = 1.0

model.B["temperature"]["low",       "medium",       "off"]  = 1.0
model.B["temperature"]["high",      "medium",       "on"]   = 1.0

model.B["temperature"]["medium",    "high",         "off"]  = 1.0
model.B["temperature"]["very_high", "high",         "on"]   = 1.0

model.B["temperature"]["high",      "very_high",    "off"]  = 1.0
model.B["temperature"]["very_high", "very_high",    "on"]   = 1.0

# C tensor is observation preference mapping (reward prior)
# In this case, our thermostat prefers "high"
model.C["temperature"]["very_low"]  = 0.1
model.C["temperature"]["low"]       = 0.1
model.C["temperature"]["medium"]    = 0.2
model.C["temperature"]["high"]      = 0.5
model.C["temperature"]["very_high"] = 0.1

# Instantiate a pymdp Agent from the generative model
agent = Agent(
    **model,
    policy_len=2,
    apply_batch=True,
    learn_A=False,
    learn_B=False,
    learn_D=False,
)

"""
Here we'll define the environment i.e. the room the thermostat is trying to regulate.
"""

@dataclass
class RoomEnv:
    current_temp: int

    def get_observation(self, action: int | None) -> int:
        """
        get_observation uses the environment current state and agent action
        to update its own state and return an observation.

        The internal *process* can be completely different to the agent, but the
        plumbing must be the same i.e. the observation and action space must match.
        """
        match action:
            case 1: # If the radiator is on, heat up the room
                next_index = min(4, self.current_temp + 1)
            case _: # If it's off, the room cools down
                next_index = max(0, self.current_temp - 1)
        self.current_temp = next_index
        return next_index


# Let's initialise the environment and set up the action-perception loop
env = RoomEnv(current_temp=1)
action = None

# Set the time horizon
T = 10
for t in range(0, T):
    observation = env.get_observation(action=action)
    # Perception
    qs = agent.infer_states(
        observations=[np.array([[observation]])],
        empirical_prior=agent.D,
    )
    # Action
    q_pi, G = agent.infer_policies(qs)
    action = int(agent.sample_action(q_pi)[0][0])
    print(f"t: {
        str(t).ljust(2)
    } | Observed temperature: {
        model_specification["observations"]["temperature"]["elements"][observation].ljust(10)
    } | Acted with radiator: {
        model_specification["controls"]["radiator"]["elements"][action].ljust(10)
    }")
