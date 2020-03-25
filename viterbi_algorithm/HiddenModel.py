import numpy as np

class HiddenModel:

    def __init__(self, number_of_states, initial_probabilities, transition_probabilities):
        self.number_of_states = number_of_states
        self.initial_probabilities = initial_probabilities
        self.transition_probabilities = transition_probabilities

    def check_probabilities(self):
        if self.initial_probabilities.shape != (self.number_of_states,):
            return False
        if self.transition_probabilities.shape != (self.number_of_states, self.number_of_states):
            return False
        return True
    
    def connect_observed_model(self, observed_model, compliance_probabilities):
        if compliance_probabilities.shape != (self.number_of_states, observed_model.number_of_states):
            return False
        self.observed_model = observed_model
        self.compliance_probabilities = compliance_probabilities
        self.sequence_length = len(observed_model.sequence)
        return True

    def calculate_path(self):
        if self.observed_model == None:
            return False
        self.states = np.zeros(self.number_of_states)
        s = self.states
        
        total_probabilities = np.zeros((self.sequence_length, self.number_of_states))
        total_probabilities_path = np.zeros((self.sequence_length, self.number_of_states), dtype=np.uint16)
        
        first_observed_state = self.observed_model.sequence[0]
        for i in range(self.number_of_states):
            total_probabilities[0][i] = self.compliance_probabilities[i][first_observed_state] * self.initial_probabilities[i]

        for t in range(1, self.sequence_length):
            for k in range(self.number_of_states):
                observed_state = self.observed_model.sequence[t]
                intermediate_probabilities = np.zeros(self.number_of_states)
                for i in range(self.number_of_states):
                    intermediate_probabilities[i] = self.transition_probabilities[i][k] * total_probabilities[t-1][i]
                total_probabilities_path[t][k] = intermediate_probabilities.argmax()
                total_probabilities[t][k] = intermediate_probabilities[total_probabilities_path[t][k]] * self.compliance_probabilities[k][observed_state]

        path = np.zeros((self.sequence_length), dtype=np.uint16)
        path[-1] = total_probabilities[-1].argmax()

        for t in range(self.sequence_length-2, 0, -1):
            path[t] = total_probabilities_path[t+1][path[t+1]]
        
        self.path = path
        return True
        
        


