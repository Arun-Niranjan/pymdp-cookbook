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
import pymdp
import numpy as np
import copy

model_labels = {
    "observations": { # Each observation type is a modality, and are keys in this dictionary
        "test_observation": [ # The first modality containing values in a list
            "positive", # the first value this observation modality can take
            "negative", # the second value this observation modality can take
            # If we had more values this modality could have, they would go here
        ],
        # If we had more observation modalities, they would go here
    },
    "states": { # Each hidden state is named in this dictionary
        "disease_state": [ # The first hidden state *variable* containing possible values as a list
            "sick",     # the first hidden state *value*
            "healthy",  # the second hidden state *value*
            # If we had more values this state could have, they would go here
        ],
        # If we had more hidden state *variables*, they would go here
    },
}

# How prevalent is the disease? This is our prior belief of being sick without any observations.

# Initialise the prior beliefs about the hidden states - the D vector
D = np.array(
    [
        [ # First state variable (disease state), as a list of values of probabilities
            0.01,  # P(X="sick") - probability of the first hidden state *value*
            0.99,  # P(x="healthy") - probability of the second hidden state *value*
        ],
        # If we had more hidden state *variables* they would go here
    ],
)

# Initialise the A matrix (likelihood matrix that maps from hidden states to observations)
A_stub = pymdp.utils.create_A_matrix_stub(model_labels)
A_stub.loc[("test_observation","positive"),("sick")]    = 0.90
A_stub.loc[("test_observation","positive"),("healthy")] = 0.05 # False positive rate
A_stub.loc[("test_observation","negative"),("sick")]    = 0.10 # False negative rate
A_stub.loc[("test_observation","negative"),("healthy")] = 0.95

print("======================================")
print("Likelihood Matrix:")
print(A_stub)
print("======================================")

A = pymdp.utils.convert_A_stub_to_ndarray(A_stub, model_labels)

num_obs, _, n_states, _ = pymdp.utils.get_model_dimensions_from_labels(model_labels)

print("Model dimensions:")
print("observables: %s"%(num_obs))
print("states: %s"%(n_states))
print("======================================")

# Generate some example observations
observation = pymdp.utils.obj_array_zeros(num_obs)
positive, negative = copy.deepcopy(observation), copy.deepcopy(observation)
positive[0][0] = 1.0
negative[0][1] = 1.0

print("Generating observations:")
print("%s: %s"%("test_observation", model_labels["observations"]["test_observation"][0]))
print("observation_vector: %s"%(positive))

print("%s: %s"%("test_observation", model_labels["observations"]["test_observation"][1]))
print("observation_vector: %s"%(negative))
print("======================================")

# Run inference over hidden states using fixed point iteration
print("Inferring disease state given a positive test result:")
qs = pymdp.algos.run_vanilla_fpi(A, positive, num_obs, n_states, prior=D)

print("Belief that I am sick given a positive test: %.2f"%(qs[0][0]))
print("All marginal posterior beliefs about hidden states:")
print(qs)
print("======================================")

print("Inferring disease state given a negative test result:")
qs = pymdp.algos.run_vanilla_fpi(A, negative, num_obs, n_states, prior=D)

print("Belief that I am sick given a negative test: %.2f"%(qs[0][0]))
print("All marginal posterior beliefs about hidden states:")
print(qs)
print("======================================")