"""
This example demonstrates perception, which is the simplest part of active inference.
Here we set up a generative model to answer the classic question of a clinical diagnostic test:

    "If I test positive for a disease, what is the probability I actually have it?"

There are two things to consider:
- the sensitivity and specificity of the test (false positive and false negative rates)
- the overall prevalence of the disease

We can represent this model using the following factor graph:

[1] -> x -> [2] -> y

where:
- x is the value of whether you have the disease or not ("sick" or "healthy"), a hidden state
- y is the observed value of the test ("positive" or "negative")
- [1] represents P(x) (the prior over hidden states) - which is the D vector
- [2] represents P(y|x) (the likelihood of observations given a state) - which is the A matrix

This example graph is shown in Figure 4.2 in the Parr Active Inference Textbook
DOI: https://doi.org/10.7551/mitpress/12441.001.0001

"""

from rich import print
from pymdp.agent import Agent
from pymdp.distribution import compile_model
import numpy as np

# We can specify a model structure and compile it with the pymdp compile_model helper function
# This creates all the tensor shapes we need
model_specification = {
    "controls": { # If our Agent could control the environment, we'd add that here
        "none": {"elements": ["none"]} # No real actions here, but required for model compilation
    },
    "states": { # Each hidden state is named in this dictionary
        "disease_state": { # The first hidden state *variable* (also referred to as a "factor")
            "elements": [ # Possible values the first hidden state variable can take
                "sick", # The first hidden state *value*
                "healthy", # The second hidden state *value*
                # If we had more values this state could have, they would go here
            ],
            "depends_on": ["disease_state"], # Key and value required for model compilation but we won't actually use this
            "controlled_by": ["none"], # Key and value required for model compilation but we won't actually use this
        }
        # If we had more hidden state *variables*, they would go here
    },
    "observations": { # Each observation type is a modality, and are keys in this dictionary
        "test_observation":{ # The first observation *modality*
            "elements": [ # Options the first observation modality can take
                "positive", # The first value this observation modality can take
                "negative", # The second value this observation modality can take
                # If we had more values this modality could have, they would go here
            ],
            "depends_on": ["disease_state"] # The hidden state *variable* that causes this observation
        }, # If we had more observation modalities, they would go here
    },
}

model = compile_model(model_specification)

# We now have a an empty structure that we need to populate with our priors
# Let's start with the likelihood tensor A.
# A is indexed by A[{OBSERVATION_MODALITY}][{OBSERVATION_VALUE}, {HIDDEN_STATE_VALUE}]
model.A["test_observation"]["positive", "sick"]     = 0.90 # True  Positive Rate
model.A["test_observation"]["positive", "healthy"]  = 0.05 # False Positive Rate
model.A["test_observation"]["negative", "sick"]     = 0.10 # False Negative Rate
model.A["test_observation"]["negative", "healthy"]  = 0.95 # True  Negative Rate

# How prevalent is the disease? This is our prior belief of being sick without any observations.
# Initialise the prior beliefs about the hidden states - the D vector
# D is indexed by D[{HIDDEN_STATE_VARIABLE}][{HIDDEN_STATE_VALUE}]
model.D["disease_state"]["sick"]    = 0.01 # P(X="sick")    - probability of the first hidden state *value*
model.D["disease_state"]["healthy"] = 0.99 # P(X="healthy") - probability of the first hidden state *value*

# Instantiate a pymdp Agent from the generative model
agent = Agent(
    **model,
    apply_batch=True, # Required for indexing to work
    learn_A=False, # We don't want our agent to learn anything from serial observations
    learn_B=False, # We don't want our agent to learn anything from serial observations
    learn_D=False, # We don't want our agent to learn anything from serial observations
)

print("======================================================")
print("Likelihood Tensor:")
print(agent.A)
print("======================================================")

print("Hidden State Prior Tensor:")
print(agent.D)
print("======================================================")

print("Generative Model dimensions:")
print("observable modalities: %s"%(agent.num_modalities))
print("observable values    : %s"%(agent.num_obs))
print("state variables      : %s"%(agent.num_factors))
print("state values         : %s"%(agent.num_states))
print("======================================================")

# Generate example observations
positive = np.array([[0]])
negative = np.array([[1]])
print("Generating observations:")
print("%s: %s"%("test_observation", model_specification["observations"]["test_observation"]["elements"][0]))
print("observation_vector: %s"%(positive))

print("%s: %s"%("test_observation", model_specification["observations"]["test_observation"]["elements"][1]))
print("observation_vector: %s"%(negative))
print("======================================================")

print("Inferring disease state given a positive test result:")
qs = agent.infer_states(
    observations=[positive],
    empirical_prior=agent.D,
)
print("Belief that I am sick given a positive test: %.2f"%(qs[0][0][0][0]))
print("All marginal posterior beliefs about hidden states:")
print(qs)
print("======================================================")

print("Inferring disease state given a negative test result:")
qs = agent.infer_states(
    observations=[negative],
    empirical_prior=agent.D,
)
print("Belief that I am sick given a negative test: %.2f"%(qs[0][0][0][0]))
print("All marginal posterior beliefs about hidden states:")
print(qs)
print("======================================================")
