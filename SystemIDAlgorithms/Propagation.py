import numpy as np


def propagation(signal, system):

    # Get general parameters
    state_dimension = system.state_dimension
    output_dimension = system.output_dimension
    dt = system.dt
    x0 = system.x0
    initial_states = system.initial_states
    number_initial_states = len(initial_states)
    number_steps = signal.number_steps

    # Get input signal data
    u = signal.data

    # Propagate depending on System type
    system_type = system.system_type

    if system_type == 'linear':
        (A, B, C, D) = system.A, system.B, system.C, system.D

        x = np.zeros([state_dimension, number_steps + 1])
        x[:, 0] = x0
        y = np.zeros([output_dimension, number_steps])
        count_init_states = 1
        for i in range(number_steps):
            y[:, i] = np.matmul(C(i * dt), x[:, i]) + np.matmul(D(i*dt), u[:, i])
            if number_initial_states > 1:
                if i + 1 == initial_states[count_init_states][1]:
                    x[:, i + 1] = initial_states[count_init_states][0]
                    if count_init_states < number_initial_states - 1:
                        count_init_states += 1
                else:
                    x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])
            else:
                x[:, i + 1] = np.matmul(A(i * dt), x[:, i]) + np.matmul(B(i * dt), u[:, i])

    if system_type == 'nonlinear':
        (F, G) = system.F, system.G

        x = np.zeros([state_dimension, number_steps + 1])
        x[:, 0] = x0
        y = np.zeros([output_dimension, number_steps])
        count_init_states = 1
        for i in range(number_steps):
            y[:, i] = G(x[:, i], u[:, i], i * dt)
            if number_initial_states > 1:
                if i + 1 == initial_states[count_init_states][1]:
                    x[:, i + 1] = initial_states[count_init_states][0]
                    if count_init_states < number_initial_states - 1:
                        count_init_states += 1
                else:
                    x[:, i + 1] = F(x[:, i], u[:, i], i * dt)
            else:
                x[:, i + 1] = F(x[:, i], u[:, i], i * dt)

    return (y, x)

# propagation_type = kwargs.get('propagation_type', None)

# # Propagation
# if propagation_type == 'channel':
#     y = np.zeros([output_dimension, input_dimension, number_steps])
#     for ch in range(input_dimension):
#         x = np.zeros([state_dimension, number_steps + 1])
#         x[:, 0] = x0[0]
#         uu = np.zeros([input_dimension, number_steps])
#         uu[ch, 0] = u[ch, 0]
#         for i in range(number_steps):
#             y[:, ch, i] = np.matmul(C(i*dt), x[:, i]) + np.matmul(D(i * dt), uu[:, i])
#             x[:, i + 1] = np.matmul(A(i*dt), x[:, i]) + np.matmul(B(i * dt), uu[:, i])

