"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


from ClassSignal import OutputSignal


class Experiments:
    def __init__(self, systems, signals):
        self.systems = systems
        self.signals = signals
        self.number_steps = signals[0].number_steps
        self.state_dimension = systems[0].state_dimension
        self.output_dimension = systems[0].output_dimension
        self.input_dimension = systems[0].input_dimension
        self.number_experiments = len(signals)
        self.frequency = systems[0].frequency
        self.outputs = []
        self.inputs = signals
        for i in range(self.number_experiments):
            self.outputs.append(OutputSignal(signals[i], systems[i], 'Output ' + signals[i].name))
