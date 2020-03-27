""" usage example """

#imports
import numpy as np
from viterbi_algorithm.HiddenModel import HiddenModel
from viterbi_algorithm.ObservedModel import ObservedModel

#possible states for a hidden model
hidden_states = ['a', 'b', 'c']

#probability of being in the beginning
#a: 0.1
#b: 0.2
#c: 0.7
initial_probabilities = np.array([0.1, 0.2, 0.7])

#probability of
#a -> a: 0.2; a -> b: 0.2; a-> c: 0.6
#b -> a: 0.3; b -> b: 0.3; b-> c: 0.4
#c -> a: 0.1; c -> b: 0.5; c-> c: 0.4
transition_probabilities = np.array([
    [0.2, 0.2, 0.6],
    [0.3, 0.3, 0.4],
    [0.1, 0.5, 0.4]
])

#create hidden model
hidden_model = HiddenModel(initial_probabilities, transition_probabilities)

#possible states for a observed model
observed_states = ['A', 'B']

number_of_states = len(observed_states)

#sequence of state indices of the observed model
#ABAAABB -> [0, 1, 0, 0, 0, 1, 1]
observed_sequence = np.array([0, 1, 0, 0, 0, 1, 1])

#create observed model
observed_model = ObservedModel(number_of_states, observed_sequence)

#probability of
#a -> A: 0.9; a -> B: 0.1
#b -> A: 0.2; b -> B: 0.8
#c -> A: 0.5; c -> B: 0.5
compliance_probabilities = np.array([
    [0.9, 0.1],
    [0.2, 0.8],
    [0.5, 0.5]
])

#connect models and print status
if hidden_model.connect_observed_model(observed_model, compliance_probabilities):
    print('connected successfully')
else:
    print('error in connecting models')
print('----------------------')

#calculate and print status
if hidden_model.calculate_path():
    print('sequence computed successfully')
    print('----------------------')
    for state in hidden_model.path:
        print(hidden_states[state],end='')
    print()
else:
    print('error')