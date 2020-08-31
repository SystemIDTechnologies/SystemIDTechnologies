![](Images/logo.png)

# System Identification Technologies
The special purpose of system identification corresponds to identifying a mathematical model describing a relationship between the input and output of a real system. Recent developments in the field of system identification have provided effective and accurate analytical tools to solve challenging problems in system engineering and virtually in any application concerning models of dynamic systems. This project aims to present the fundamental algorithms of time-domain system identification for linear time-invariant systems.



## Installation
Directly download the zip file or clone the project:
```bash
git clone https://github.com/SystemIDTechnologies/SystemIDTechnologies.git
```

## Usage
Five systems are currently implemented: one, two, three mass spring damper systems, a simulated automobile system and a customized system. For simplicity, each system has his own file: `Main_1Mass.py`, `Main_2Mass.py`, `Main_3Mass.py`, `Main_Automobile.py`, `Main_Custom.py`. Each file is highly customable. Schematic descriptions of the systems are in the Images folder.

## Example
Let's study the Two Mass Spring Damper System.
<p align="center">
  <img src="Images/Mass2.png" width="500">
</p>

### 1. Initial Condition, Input Signal and Measurement Noise
After libraries imports, one must choose an initial condition type, an input signal, the frequency and total length of the input signal as well as a signal to noise ratio for measurement noise (if `True`):

```python
## Initial Condition type, Input Signal parameters and Noise
initialCondition = 'Random'
inputSignalName = 'White Noise'
frequency = 5
total_time = 50
noise = True
snr = 1e8
```

By default, the initial condition is set to be random and the input signal is a white noise of frequency 5 for a total of 50 seconds. Some white noise is added to the measurements with a snr of 1e8.
Initial condition can be random (`'Random'`) or zero (`'Zero'`) or customized later. Input signals can be:
* `'Impulse'`
* `'White Noise'`
* `'Triangle'`
* `'Sinusoidal'`
* `'Combination'`

All signals characteristics can be customized.

### 2. Define the Dynamics
One must choose the characteristics of the system such as the values of the time step (`dt`), the masses (`mass1` and `mass2`), the spring constants (`spring_constant1` and `spring_constant2`), the damping coefficients (`damping_coefficient1` and `damping_coefficient2`), the force coefficients (`force_coefficient1` and `force_coefficient2`) and the measurements (`measurements1` and `measurements2`). Measurements vectors can contain `position`, `velocity` and/or `acceleration`.

```python
## Define the dynamics
dt = 0.1
mass1 = 1
mass2 = 1.8
spring_constant1 = 1
spring_constant2 = 3
damping_coefficient1 = 0.3
damping_coefficient2 = 0.4
force_coefficient1 = 1
force_coefficient2 = 2
measurements1 = ['position', 'velocity', 'acceleration']
measurements2 = ['position', 'velocity', 'acceleration']
Dynamics = TwoMassSpringDamperDynamics(dt, mass1, mass2, spring_constant1, spring_constant2, damping_coefficient1, damping_coefficient2, force_coefficient1, force_coefficient2, measurements1, measurements2)
```


### 3. Initial Condition
This section defines the initial condition. If `'Custom'` has been chosen, it is time to enter a value (np.array of system's dimension). Otherwise, no action is expected.

```python
## Initial Condition of the Linear System
if initialCondition == 'Zero':
    initial_condition = np.zeros(Dynamics.state_dimension)
if initialCondition == 'Random':
    initial_condition = np.random.randn(Dynamics.state_dimension)
if initialCondition == 'Custom':
    initial_condition = np.array([1, 1, 0.1, -0.1])
initial_states = [(initial_condition, 0)]
```


### 4. Define the corresponding Linear System
This section defines the corresponding linear system. If the input signal is zero (initial condition repsonse), one must choose the number of experiments to perform (defalut `number_experiments = 1`).

```python
## Define the corresponding Linear System
if inputSignalName == 'Zero':
    number_experiments = 1
    systems = []
    initial_states = []
    for i in range(number_experiments):
        init_state = [(1*np.random.randn(Dynamics.state_dimension), 0)]
        initial_states.append(init_state)
        if i == 0:
            Sys = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, init_state, 'Two Mass Spring Damper System', Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D)
            systems.append(Sys)
        else:
            systems.append(LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, init_state, 'Two Mass Spring Damper System', Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D))
else:
    Sys = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, initial_states, 'Two Mass Spring Damper System', Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D)
```


### 5. Input Signal parameters
This section allows to tune the parameters of the input signal depending on what has been chosen previously. For an impulse, one can choose the magnitude of the impulse (`magnitude_impulse`). For a white noise, one can choose the mean (`mean`) and the covariance (`covariance`). For a triangle input, one can choose the magnitude of the peak (`magnitude_peak`). For a sinusoidal input, one can choose the magnitude and the frequency of the sinusoid (`magnitude_sinusoid` and `frequency_sinusoid`). For a combination input, one can choose the maximum of the ramp of the linear portion (`maximum_ramp`), the exponential decay rate of the exponential portion (`exponential_decay_rate`) and the covariance of the white noise portion (`exponential_decay_rate`).

```python
## Parameters of the Input Signal - From input signal parameters
number_steps = total_time * frequency + 1
if inputSignalName == 'Impulse':
    magnitude_impulse = 1 * np.ones(Dynamics.input_dimension)
if inputSignalName == 'White Noise':
    mean = np.zeros(Dynamics.input_dimension)
    covariance = 1 * np.eye(Dynamics.input_dimension)
if inputSignalName == 'Triangle':
    magnitude_peak = np.random.randn(Dynamics.input_dimension)
if inputSignalName == 'Sinusoidal':
    magnitude_sinusoid = np.random.randn(Dynamics.input_dimension)
    frequency_sinusoid = 1 * np.random.randn(Dynamics.input_dimension)
if inputSignalName == 'Combination':
    maximum_ramp = 1 * np.ones(Dynamics.input_dimension)
    exponential_decay_rate = -0.1 * np.ones(Dynamics.input_dimension)
    covariance = 1 * np.eye(Dynamics.input_dimension)
  ```
  
  
  ### 6. Define the Input Signal
  This section creates the input signal depending on what has been chosen before. No action is needed.
  
  ```python
  ## Define the Input Signal
if inputSignalName == 'Impulse':
    S1 = Signal(total_time, frequency, Dynamics.input_dimension, inputSignalName, magnitude_impulse=magnitude_impulse)
if inputSignalName == 'White Noise':
    S1 = Signal(total_time, frequency, Dynamics.input_dimension, inputSignalName, mean=mean, covariance=covariance)
if inputSignalName == 'Triangle':
    S1 = Signal(total_time, frequency, Dynamics.input_dimension, inputSignalName, magnitude_peak=magnitude_peak)
if inputSignalName == 'Sinusoidal':
    S1 = Signal(total_time, frequency, Dynamics.input_dimension, inputSignalName, magnitude_sinusoid=magnitude_sinusoid, frequency_sinusoid=frequency_sinusoid)
if inputSignalName == 'Combination':
    S1 = Signal(total_time, frequency, Dynamics.input_dimension, inputSignalName, maximum_ramp=maximum_ramp, covariance=covariance, exponential_decay_rate=exponential_decay_rate, )
if inputSignalName == 'Zero':
    S1 = Signal(total_time, frequency, Dynamics.input_dimension, inputSignalName)
    input_signals = []
    for i in range(number_experiments):
        input_signals.append(S1)
 ```
 
 
### 7. Define the Output Signal
This section creates the output signal that is the propagation of the input signal through the linear system. No action is needed.

```python
## Define the Output Signal
if inputSignalName == 'Zero':
    Exp = Experiments(systems, input_signals)
    S2_ini = Exp.output_signals[0]
else:
    S2_ini = OutputSignal(S1, Sys, 'Output Signal')
```


### 8. Add Noise
This section adds measurement noise previously defined. No action is needed.

```python
## Add some noise
if noise:
    if inputSignalName == 'Zero':
        for i in range(number_experiments):
            output_signal = Exp.output_signals[i]
            mean_noise = np.zeros(Dynamics.output_dimension)
            covariance_noise = np.eye(Dynamics.output_dimension) * np.mean(output_signal.data ** 2) / snr
            print(covariance_noise)
            SNoise = Signal(total_time, frequency, Dynamics.output_dimension, 'Noise', mean=mean_noise, covariance=covariance_noise)
            Exp.output_signals[i] = add2Signals(output_signal, SNoise)
        S2 = Exp.output_signals[0]
    else:
        mean_noise = np.zeros(Dynamics.output_dimension)
        covariance_noise = np.eye(Dynamics.output_dimension) * np.mean(S2_ini.data**2) / snr
        SNoise = Signal(total_time, frequency, Dynamics.output_dimension, 'Noise', mean=mean_noise, covariance=covariance_noise)
        S2 = add2Signals(S2_ini, SNoise)
else:
    if inputSignalName == 'Zero':
        S2 = Exp.output_signals[0]
    else:
        S2 = S2_ini
```


### 9. Markov Parameters
This section calculates Markov parameters. No action is needed.

```python
## Calculate Markov Parameters and Identified system
if inputSignalName == 'Zero':
    markov_parameters_true = getInitialConditionResponseMarkovParameters(Dynamics.A, Dynamics.C, number_steps)
    ERA1 = ERAFromInitialConditionResponse(Exp.output_signals, Dynamics.state_dimension, Dynamics.input_dimension)
    markov_parameters = getInitialConditionResponseMarkovParameters(ERA1.A, ERA1.C, number_steps)
elif inputSignalName == 'Impulse':
    markov_parameters_true = getMarkovParameters(Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D, number_steps)
    markov_parameters = OKID(S1, S2).markov_parameters
    ERA1 = ERA(markov_parameters, Dynamics.state_dimension)
else:
    markov_parameters_true = getMarkovParameters(Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D, number_steps)
    if initialCondition == 'Zero':
        OKID1 = OKIDObserver(S1, S2)
        markov_parameters = OKID1.markov_parameters
        ERA1 = ERA(markov_parameters, Dynamics.state_dimension)
    else:
        OKID1 = OKIDObserverWithInitialCondition(S1, S2)
        markov_parameters = OKID1.markov_parameters
        ERA1 = ERA(markov_parameters, Dynamics.state_dimension)
        x0 = identificationInitialCondition(S1, S2, ERA1.A, ERA1.B, ERA1.C, ERA1.D, 0)
```


### 10. Identified System
This section creates the identified system. No action is needed.

```python
## Define Identified System
if inputSignalName == 'Zero':
    SysID = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, [(ERA1.x0[:, 0], 0)], 'Identified System', ERA1.A, ERA1.B, ERA1.C, ERA1.D)
elif inputSignalName == 'Impulse':
    SysID = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, initial_states, 'Identified System', ERA1.A, ERA1.B, ERA1.C, ERA1.D)
else:
    if initialCondition == 'Zero':
        SysID = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, initial_states, 'Identified System', ERA1.A, ERA1.B, ERA1.C, ERA1.D)
    else:
        SysID = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, [(x0[:, 0], 0)], 'Identified System', ERA1.A, ERA1.B, ERA1.C, ERA1.D)
```


### 11. Identified Output Signal
This section defines the identified output signal that is to compare to the output signal. No action is needed.

```python
## Define the Identified Output Signal
if inputSignalName == 'Zero':
    S2ID = OutputSignal(input_signals[0], SysID, 'Identified Output Signal')
else:
    S2ID = OutputSignal(S1, SysID, 'Identified Output Signal')
```
    
    
### 12. Plotting
This section plots the graphs for analysis. No action is needed

```python
## Plotting
if noise:
    plotSignals([[S1], [S2_ini, S2, S2ID], [subtract2Signals(S2, S2ID)]], 1)
else:
    plotSignals([[S1], [S2, S2ID], [subtract2Signals(S2, S2ID)]], 1)
plotEigenValues([Sys, SysID], 2)
plotSingularValues([ERA1], ['IdentifiedSystem'], 3)
plotMarkovParameters2(markov_parameters, markov_parameters_true, 'OKID', 'True', 4)
```
