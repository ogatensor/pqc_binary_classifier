from math import pi, sqrt

# Define state |0>
initial_state = [1,0] 

# Redefine the quantum circuit
qc = QuantumCircuit(1)

# Initialise the 0th qubit in the state `initial_state`
qc.initialize(initial_state, 0) 

# Rotate the state by 60%
qc.ry(prob_to_angle(0.6), 0)

# Tell Qiskit how to simulate our circuit
backend = Aer.get_backend('statevector_simulator')

# execute the qc
results = execute(qc,backend).result().get_counts()

# plot the results
plot_histogram(results)