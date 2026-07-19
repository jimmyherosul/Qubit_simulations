# ----------------------------- Import libraries -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from qutip import Bloch, basis, expect, sigmax, sigmay, sigmaz, ket2dm

# To check the version of qutip
#import qutip
#print(qutip.__version__)

# ----------------------------- Bloch sphere representation -----------------------------
b = Bloch()

b.add_points([0, 0, 1], 'm')        # Add a point on |0> (the north pole)
b.add_points([0, 0, -1], 'm')       # Add a point on |1> (the south pole)
b.add_points([1, 0, 0], 'm')        # Add a point on |+> (the x-axis equator)
b.add_points([-1, 0, 0], 'm')       # Add a point on |-> (the x-axis equator)
b.add_points([0, 1, 0], 'm')        # Add a point on |+i> (the y-axis equator)
b.add_points([0, -1, 0], 'm')       # Add a point on |-i> (the y-axis equator)

# ----------------------------- Qubit state -----------------------------
#qubit_state = basis(2, 0)                                      # Add the qubit state pointing toward |0>
#qubit_state = basis(2, 1)                                      # Add the qubit state pointing toward |1>
#qubit_state = (basis(2, 0) + (1+0j)*basis(2, 1)).unit()        # Add the normalized qubit state pointing toward |+>
#qubit_state = (basis(2, 0) + (-1+0j)*basis(2, 1)).unit()       # Add the normalized qubit state pointing toward |->
#qubit_state = (basis(2, 0) + (0+1j)*basis(2, 1)).unit()        # Add the normalized qubit state pointing toward |+i>
qubit_state = (basis(2, 0) + (0-1j)*basis(2, 1)).unit()        # Add the normalized qubit state pointing toward |-i>

b.add_states(qubit_state)       # Plotting the qubit state on the Bloch sphere

# ----------------------------- Bloch coordinates -----------------------------
# Extracting coordinates of the Bloch vector from the qubit state
x = expect(sigmax(), qubit_state)
y = expect(sigmay(), qubit_state)
z = expect(sigmaz(), qubit_state)

print("\nBloch coordinates:", [x, y, z])
print("|r| =", np.sqrt(x**2 + y**2 + z**2), "\n")       # If |r| = 1, the qubit state is pure (on surface of Bloch sphere). If |r| < 1, the qubit state is mixed (inside Bloch sphere).

# ----------------------------- State probabilities -----------------------------
qubit_rho = ket2dm(qubit_state)         # Convert the qubit state to a density matrix
prob_0 = qubit_rho[0, 0].real           # qubit_rho_{0,0} diagonal density matrix element - Probability of measuring the qubit in state |0>
prob_1 = qubit_rho[1, 1].real           # qubit_rho_{1,1} diagonal density matrix element - Probability of measuring the qubit in state |1>

print("P_0 =", prob_0)
print("P_1 =", prob_1, "\n")

b.show()
plt.show()