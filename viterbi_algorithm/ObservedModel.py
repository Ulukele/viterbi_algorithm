
class ObservedModel:

    def __init__(self, number_of_states, sequence):
        self.number_of_states = number_of_states
        self.sequence = sequence
    
    def check_sequence(self):
        for state in self.sequence:
            if state < 0 or state >= number_of_states:
                return False
        return True    