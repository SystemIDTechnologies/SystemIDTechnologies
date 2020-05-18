"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


from ClassesGeneral.ClassSignal import OutputSignal


class Experiments:
    def __init__(self, systems, input_signals):
        self.systems = systems
        self.input_signals = input_signals
        self.number_steps = input_signals[0].number_steps
        self.state_dimension = systems[0].state_dimension
        self.output_dimension = systems[0].output_dimension
        self.input_dimension = systems[0].input_dimension
        self.number_experiments = len(input_signals)
        self.frequency = systems[0].frequency
        self.output_signals = []
        for i in range(self.number_experiments):
            self.output_signals.append(OutputSignal(input_signals[i], systems[i], 'Output ' + input_signals[i].name))
