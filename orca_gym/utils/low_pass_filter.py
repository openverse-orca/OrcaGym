import numpy as np

class LowPassFilter:
    def __init__(self, alpha, initial_output):
        self.alpha = alpha
        self.output = initial_output

    def apply(self, input):
        self.output = self.alpha * input + (1 - self.alpha) * self.output
        return self.output