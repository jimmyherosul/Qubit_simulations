# ----------------------------- Import libraries -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from qutip import Bloch, Qobj, basis, expect, sigmax, sigmay, sigmaz, ket2dm

# To check the version of qutip
#import qutip
#print(qutip.__version__)


# ----------------------------- Parameters -----------------------------
# Delta - Detuning (i.e. Unperturbed half-energy difference between the two basis states) in eV
# Delta_eff - Effective Detuning in rotating frame in eV
# V_mag - Coupling strength between the two basis states |0> and |1> in eV
# Omega_d - External Drive frequency in rad/s 
# Omega_R - Driven Rabi frequency in rad/s
# Omega_0 - Unperturbed Effective Detuning frequency (i.e. Natural transition/resonant frequency of the two-level system) in rad/s
# t_duration - Pulse duration for the gate in s
# phi - Phase of the coupling term V between the two basis states |0> and |1> in rad

# where Delta_eff (and by extension Omega_d), V_mag, t_duration, and phi are tuning parameters for implementing logic gates.

# Throughout this simulation, energy is in eV, time is in s and phase is in rad. Using the reduced Planck constant, the (angular) frequency is therefore in rad/s, unless otherwise specified.
# Note that for a frequency range of up to 10GHz (6.3x10^10 rad/s), the corresponding energy range is up to approx 41ueV, which is typical for superconducting qubits.

hbar = 6.582119569e-16        # REDUCED Planck constant in eV*s
E_mean = 0                    # For simplicity, the mean unperturbed energy of the two basis states |0> and |1> is set to zero
Delta = 100e-6                 # For clearer distinction between evolution in lab and rotating frames, we let the instrinsic detuning of the two-level system be 100ueV


# ----------------------------- Generating tuning parameters for gate operation -----------------------------
def tuning_parameters(gate_name, hbar, Delta_eff, Delta, V_I_mag, V_Q_mag):

    # All I/Q Coupling and Detuning signals are generated as a square-wave pulse over t_duration
    V_mag = np.sqrt(V_I_mag**2 + V_Q_mag**2)
    Omega_d = (2/hbar)*(Delta - Delta_eff)
    Omega_0 = 2*Delta/hbar 
    Omega_eff = Omega_0 - Omega_d
    Omega_R = np.sqrt(Omega_eff**2 + (2*V_mag/hbar)**2) 

    # For X gate: Delta_eff = 0, V_I_mag > 0, V_Q_mag = 0, and t_duration such that (Omega_R/2)*t_duration = pi/2
    if gate_name == 'X':
        t_duration = (np.pi/2)*(2/Omega_R)

    # For Y gate: Delta_eff = 0, V_I_mag = 0, V_Q_mag > 0, and t_duration such that (Omega_R/2)*t_duration = pi/2
    elif gate_name == 'Y':      
        t_duration = (np.pi/2)*(2/Omega_R)

    # For Z gate: Delta_eff > 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_R/2)*t_duration = pi/2
    elif gate_name == 'Z':      
        t_duration = (np.pi/2)*(2/Omega_R)

    # For S gate: Delta_eff < 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_R/2)*t_duration = pi/4
    elif gate_name == 'S':      
        t_duration = (np.pi/4)*(2/Omega_R)

    # For T gate: Delta_eff < 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_R/2)*t_duration = pi/8
    elif gate_name == 'T':      
        t_duration = (np.pi/8)*(2/Omega_R)

    # For H gate: Delta_eff < 0, V_I_mag = Delta_eff or 0, V_Q_mag = 0 or Delta_eff, and t_duration such that (Omega_R/2)*t_duration = pi/8
    elif gate_name == 'H':      
        t_duration = (np.pi/2)*(2/Omega_R)

    # For I gate: Delta_eff, V_I_mag, and V_Q_mag can be set to any values as long as they are NOT ALL ZERO, since t_duration = 0
    elif gate_name == 'I':   
        t_duration = 0*(2/Omega_R)

    t_points = np.linspace(0, t_duration, 100) 
    Delta_eff_signal = np.full_like(t_points, Delta_eff)
    V_I_amplitude = np.full_like(t_points, V_I_mag)
    V_Q_amplitude = np.full_like(t_points, V_Q_mag)
        
    # The laboratory-frame Coupling signal rotates at the drive frequency and are obtained from |V|e^{i*(Omega_d*t-phi)}.
    V_signal_lab = V_I_mag*np.cos(Omega_d*t_points) + V_Q_mag*np.sin(Omega_d*t_points)          # Only laboratory-frame Coupling signal we are applying through IQ modulation
    V_signal_lab_90 = V_Q_mag*np.cos(Omega_d*t_points) - V_I_mag*np.sin(Omega_d*t_points)       # Corresponding Coupling signal, phase shifted by 90degrees

    return {
        "Omega_R": Omega_R, 
        "Omega_0": Omega_0,
        "Omega_eff": Omega_eff,
        "Omega_d": Omega_d, 
        "t_points": t_points, 
        "Delta_eff_signal": Delta_eff_signal, 
        "V_I_amplitude": V_I_amplitude, 
        "V_Q_amplitude": V_Q_amplitude, 
        "V_signal_lab": V_signal_lab, 
        "V_signal_lab_90": V_signal_lab_90
        }


# ----------------------------- Unitary time-evolution operator for sinusoidally-driven two-level system in stationary/laboratory and rotating frames of reference ----------------------------- 
def unitary_operator_rot(t, hbar, E_mean, V_I_amplitude, V_Q_amplitude, Omega_R, Omega_eff):
    U = np.exp(-1j*E_mean*t/hbar) * np.array([[np.cos(Omega_R*t/2) + 1j*(Omega_eff/Omega_R)*np.sin(Omega_R*t/2),
                                            -1j*(V_I_amplitude - 1j*V_Q_amplitude)*(2/(hbar*Omega_R))*np.sin(Omega_R*t/2)],
                                            [-1j*(V_I_amplitude + 1j*V_Q_amplitude)*(2/(hbar*Omega_R))*np.sin(Omega_R*t/2),
                                            np.cos(Omega_R*t/2) - 1j*(Omega_eff/Omega_R)*np.sin(Omega_R*t/2)]])

    return U

def unitary_operator_lab(t, hbar, E_mean, V_I_amplitude, V_Q_amplitude, V_signal_lab, V_signal_lab_90, Omega_R, Omega_eff, Omega_d):
    U = np.exp(-1j*E_mean*t/hbar) * np.array([[(np.cos(Omega_d*t) + 1j*np.sin(Omega_d*t))*(np.cos(Omega_R*t/2) + 1j*(Omega_eff/Omega_R)*np.sin(Omega_R*t/2)),
                                            -1j*(V_signal_lab - 1j*V_signal_lab_90)*(2/(hbar*Omega_R))*np.sin(Omega_R*t/2)], 
                                            [-1j*(V_I_amplitude + 1j*V_Q_amplitude)*(2/(hbar*Omega_R))*np.sin(Omega_R*t/2),
                                            np.cos(Omega_R*t/2) - 1j*(Omega_eff/Omega_R)*np.sin(Omega_R*t/2)]])
    return U


# ----------------------------- Time evolution of Qubit state -----------------------------
# REMINDER:
# For X gate: Delta_eff = 0, V_I_mag > 0, V_Q_mag = 0, and t_duration such that (Omega_R/2)*t_duration = pi/2
# For Y gate: Delta_eff = 0, V_I_mag = 0, V_Q_mag > 0, and t_duration such that (Omega_R/2)*t_duration = pi/2
# For Z gate: Delta_eff > 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_R/2)*t_duration = pi/2
# For S gate: Delta_eff < 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_R/2)*t_duration = pi/4
# For T gate: Delta_eff < 0, V_I_mag = 0, V_Q_mag = 0, and t_duration such that (Omega_R/2)*t_duration = pi/8
# For H gate: Delta_eff < 0, V_I_mag = Delta_eff or 0, V_Q_mag = 0 or Delta_eff, and t_duration such that (Omega_R/2)*t_duration = pi/8
# For I gate: Delta_eff, V_I_mag, and V_Q_mag can be set to any values as long as they are NOT ALL ZERO, since t_duration = 0

# Hadamard similarity transformations: HXH=Z, HYH=-Y, HZH=X

# Select a sequence of single-qubit gates and set the appropriate detuning and I/Q coupling strengths
gate_sequence = [
    #{"gate": "H", "Delta_eff": -10e-6,  "V_I_mag": -10e-6, "V_Q_mag": 10e-9},
    #{"gate": "H", "Delta_eff": -10e-6,  "V_I_mag": -10e-6, "V_Q_mag": 10e-9},
    {"gate": "X", "Delta_eff": 10e-9, "V_I_mag": 10e-6, "V_Q_mag": 10e-9},
    #{"gate": "X", "Delta_eff": 10e-9, "V_I_mag": 10e-6, "V_Q_mag": 10e-9},
    {"gate": "Y", "Delta_eff": 10e-9, "V_I_mag": 10e-9, "V_Q_mag": 10e-6},
    #{"gate": "Z", "Delta_eff": 10e-6, "V_I_mag": 10e-9, "V_Q_mag": 10e-9},
    #{"gate": "S", "Delta_eff": -10e-6, "V_I_mag": 10e-9, "V_Q_mag": 10e-9},
    #{"gate": "T", "Delta_eff": -10e-6, "V_I_mag": 10e-9, "V_Q_mag": 10e-9},
    #{"gate": "H", "Delta_eff": -10e-6,  "V_I_mag": -10e-6, "V_Q_mag": 10e-9},
    {"gate": "H", "Delta_eff": -10e-6,  "V_I_mag": -10e-6, "V_Q_mag": 10e-9},
]

# Select an initial state
initial_state = basis(2, 0)                                      # initial state |0>
#initial_state = basis(2, 1)                                      # initial state |1>
#initial_state = (basis(2, 0) + (1+0j)*basis(2, 1)).unit()        # initial state |+>
#initial_state = (basis(2, 0) + (-1+0j)*basis(2, 1)).unit()       # initial state |->
#initial_state = (basis(2, 0) + (0+1j)*basis(2, 1)).unit()        # initial state |+i>
#initial_state = (basis(2, 0) + (0-1j)*basis(2, 1)).unit()        # initial state |-i>

# Compute the time-evolved qubit states for the full gate sequence
def gate_operation(initial_state, gate_sequence):
    t_accumulated = 0
    input_state = initial_state
    input_state_rot = initial_state
    
    t_points_sequence = []
    Delta_eff_signal_sequence = []
    V_signal_lab_sequence = []
    V_signal_lab_90_sequence = []
    V_I_amplitude_sequence = []
    V_Q_amplitude_sequence = []
    Omega_R_sequence = []
    Omega_0_sequence = []
    Omega_eff_sequence = []
    Omega_d_sequence = []
    qubit_states_lab = []
    qubit_states_rot = []

    gate_info = []

    for gate_settings in gate_sequence:         # For each gate in gate_sequence
        gate = gate_settings["gate"]
        Delta_eff = gate_settings["Delta_eff"]
        V_I_mag = gate_settings["V_I_mag"]
        V_Q_mag = gate_settings["V_Q_mag"]

        # Generate tuning parameters (i.e. time duration, I/Q coupling signals and detuning signal) for the current gate.
        # The time points for each gate in tuning_parameters function always start from t = 0.
        gate_param = tuning_parameters(gate, hbar, Delta_eff, Delta, V_I_mag, V_Q_mag)

        t_current_gate = gate_param["t_points"]

        # Apply the corresponding unitary time-evolution operator to the state produced by the previous gate
        U_gate_lab = unitary_operator_lab(
            t_current_gate, 
            hbar, 
            E_mean, 
            gate_param["V_I_amplitude"], 
            gate_param["V_Q_amplitude"],
            gate_param["V_signal_lab"], 
            gate_param["V_signal_lab_90"], 
            gate_param["Omega_R"], 
            gate_param["Omega_eff"],
            gate_param["Omega_d"]
            )

        # Apply the rotating-frame unitary time-evolution operator independently so that it can be replaced later
        U_gate_rot = unitary_operator_rot(
            t_current_gate,
            hbar,
            E_mean,
            gate_param["V_I_amplitude"],
            gate_param["V_Q_amplitude"],
            gate_param["Omega_R"],
            gate_param["Omega_eff"]
            )
        
        
        for i in range(len(t_current_gate)):
            state_t_lab = Qobj(U_gate_lab[:, :, i]) * input_state
            state_t_rot = Qobj(U_gate_rot[:, :, i]) * input_state_rot

            # Shift the time duration of the current gate by accumulated duration of previous gate
            t_sequence = t_accumulated + t_current_gate[i]

            # Store the resulting qubit state and all tuning parameters from each time point
            qubit_states_lab.append(state_t_lab)
            qubit_states_rot.append(state_t_rot)
            t_points_sequence.append(t_sequence)
            Delta_eff_signal_sequence.append(gate_param["Delta_eff_signal"][i])
            V_signal_lab_sequence.append(gate_param["V_signal_lab"][i])
            V_signal_lab_90_sequence.append(gate_param["V_signal_lab_90"][i])
            V_I_amplitude_sequence.append(gate_param["V_I_amplitude"][i])
            V_Q_amplitude_sequence.append(gate_param["V_Q_amplitude"][i])
            Omega_R_sequence.append(gate_param["Omega_R"])
            Omega_0_sequence.append(gate_param["Omega_0"])
            Omega_eff_sequence.append(gate_param["Omega_eff"])
            Omega_d_sequence.append(gate_param["Omega_d"])

        gate_info.append({
            "gate": gate,
            "Delta_eff": Delta_eff,
            "V_I_mag": V_I_mag,
            "V_Q_mag": V_Q_mag,
            "Omega_R": gate_param["Omega_R"],
            "Omega_0": gate_param["Omega_0"],
            "Omega_eff": gate_param["Omega_eff"],
            "Omega_d": gate_param["Omega_d"],
            "t_start": t_accumulated,
            "t_end": t_accumulated + t_current_gate[-1],
            "t_duration": t_current_gate[-1]
        })

        # Update the input state so that the final state of this gate becomes the initial state for the next gate
        input_state = Qobj(U_gate_lab[:, :, -1]) * input_state
        input_state_rot = Qobj(U_gate_rot[:, :, -1]) * input_state_rot

        # Update the accumulated time duration of previous gate so that the next gate starts after this gate
        t_accumulated = t_accumulated + t_current_gate[-1]

    return {
        "t_points": np.array(t_points_sequence),
        "Delta_eff_signal": np.array(Delta_eff_signal_sequence),
        "V_signal_lab": np.array(V_signal_lab_sequence),
        "V_signal_lab_90": np.array(V_signal_lab_90_sequence),
        "V_I_amplitude": np.array(V_I_amplitude_sequence),
        "V_Q_amplitude": np.array(V_Q_amplitude_sequence),
        "Omega_R": np.array(Omega_R_sequence),
        "Omega_0": np.array(Omega_0_sequence),
        "Omega_eff": np.array(Omega_eff_sequence),
        "Omega_d": np.array(Omega_d_sequence),
        "qubit_states_lab": qubit_states_lab,
        "qubit_states_rot": qubit_states_rot,
        "gate_info": gate_info
    }

sequence_param = gate_operation(initial_state, gate_sequence)
qubit_states_lab = sequence_param["qubit_states_lab"]
qubit_states_rot = sequence_param["qubit_states_rot"]

for info in sequence_param["gate_info"]:
    print("\nGate =", info["gate"])
    print("Delta_eff =", info["Delta_eff"]*1e6, "ueV")
    print("V_I_mag =", info["V_I_mag"]*1e6, "ueV")
    print("V_Q_mag =", info["V_Q_mag"]*1e6, "ueV")
    print("t_start =", info["t_start"]*1e9, "ns")
    print("t_end =", info["t_end"]*1e9, "ns")
    print("t_duration =", info["t_duration"]*1e9, "ns")
    print("Omega_R =", (info["Omega_R"]/(2*np.pi))*1e-9, "GHz")
    print("Omega_0 =", (info["Omega_0"]/(2*np.pi))*1e-9, "GHz")
    print("Omega_d =", (info["Omega_d"]/(2*np.pi))*1e-9, "GHz")
    print("Omega_0 - Omega_d =", (info["Omega_eff"]/(2*np.pi))*1e-9, "GHz")

# Obtaining the Bloch-sphere coordinates of the qubit states at each time step
x_lab = np.array([expect(sigmax(), state).real for state in qubit_states_lab])
y_lab = np.array([expect(sigmay(), state).real for state in qubit_states_lab])
z_lab = np.array([expect(sigmaz(), state).real for state in qubit_states_lab])
r_mag_lab = np.sqrt(x_lab**2 + y_lab**2 + z_lab**2)

# Obtain the rotating-frame Bloch-vector coordinates separately from the laboratory-frame coordinates
x_rot = np.array([expect(sigmax(), state).real for state in qubit_states_rot])
y_rot = np.array([expect(sigmay(), state).real for state in qubit_states_rot])
z_rot = np.array([expect(sigmaz(), state).real for state in qubit_states_rot])
r_mag_rot = np.sqrt(x_rot**2 + y_rot**2 + z_rot**2)

# Computing the probabilities of measuring the qubit in states |0> and |1> at each time step
p0_rot = np.array([ket2dm(state)[0, 0].real for state in qubit_states_rot])
p1_rot = np.array([ket2dm(state)[1, 1].real for state in qubit_states_rot])
p_total_rot = p0_rot + p1_rot


# ----------------------------- Creating Bloch sphere and all Time-plots-----------------------------
plt.close('all')

fig_bloch = plt.figure(figsize=(14, 7))
ax_bloch_lab = fig_bloch.add_subplot(221, projection='3d')
b_lab = Bloch(axes=ax_bloch_lab)
fig_bloch.suptitle("Time-Evolution of Qubit State", fontsize=14, y=0.95)
fig_bloch.text(0.3, 0.9, "In Laboratory Frame", ha="center", va="center", fontsize=12)
b_lab.font_size = 14

# Create a second Bloch sphere for the state represented in the rotating reference frame
ax_bloch_rot = fig_bloch.add_subplot(222, projection='3d')
b_rot = Bloch(axes=ax_bloch_rot)
fig_bloch.text(0.72, 0.9, "In Rotating Frame", ha="center", va="center", fontsize=12)
b_rot.font_size = 14

fig_time, (ax_tuning, ax_coordinates, ax_probabilities) = plt.subplots(
    3, 1,
    figsize=(10, 14),
    sharex=True
)
fig_time.subplots_adjust(hspace=0.45, bottom=0.15, right=0.72)


# ----------------------------- Plotting Time-evolution of Coupling V(t) and Effective Detuning Delta_eff signals -----------------------------
ax_tuning.set_title("Time-Evolution of Coupling and Effective Detuning signals", fontsize=12, pad=10)
ax_tuning.set_xlim(sequence_param["t_points"][0]*1e9, sequence_param["t_points"][-1]*1e9)
ax_tuning.set_ylim(-11, 11)
ax_tuning.set_ylabel("Energy (ueV)", fontsize=10)
ax_tuning.minorticks_on()
ax_tuning.grid(True, which="major", linestyle="-", linewidth=0.8)
ax_tuning.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

coupling_I_amplitude, = ax_tuning.plot([], [], label=r"$V_{I_{mag}}$")
coupling_Q_amplitude, = ax_tuning.plot([], [], label=r"$V_{Q_{mag}}$")
coupling_signal, = ax_tuning.plot([], [], label=r"$V_{sig}(t)$")
detuning_line, = ax_tuning.plot([], [], label=r"$\Delta_{eff}$")

time_tuning_cursor = ax_tuning.axvline(
    sequence_param["t_points"][0]*1e9,
    color='r',
    linestyle='--',
    linewidth=1.5
)

tuning_cursor, = ax_tuning.plot([], [], 'ro', markersize=4)

ax_tuning.legend(loc="upper right")

tuning_readout = ax_tuning.text(
    1.04, 0.50,
    "",
    transform=ax_tuning.transAxes,
    ha="left",
    va="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9)
)


# ----------------------------- Plotting Time-evolution of Bloch-vector coordinates -----------------------------
ax_coordinates.set_title("Time-Evolution of Bloch-vector coordinates (Rotating frame)", fontsize=12, pad=10)
ax_coordinates.set_xlim(sequence_param["t_points"][0]*1e9, sequence_param["t_points"][-1]*1e9)
ax_coordinates.set_ylim(-1.1, 1.1)
ax_coordinates.set_ylabel("Pauli Expectation Values", fontsize=10)
ax_coordinates.minorticks_on()
ax_coordinates.grid(True, which="major", linestyle="-", linewidth=0.8)
ax_coordinates.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

coordinate_x, = ax_coordinates.plot([], [], label=r"$\langle\sigma_x\rangle$")
coordinate_y, = ax_coordinates.plot([], [], label=r"$\langle\sigma_y\rangle$")
coordinate_z, = ax_coordinates.plot([], [], label=r"$\langle\sigma_z\rangle$")
r_magnitude, = ax_coordinates.plot([], [], color="black", label=r"$|\mathbf{r}|$")

time_coordinate_cursor = ax_coordinates.axvline(
    sequence_param["t_points"][0]*1e9,
    color='r',
    linestyle='--',
    linewidth=1.5
)

coordinate_cursor, = ax_coordinates.plot([], [], 'ro', markersize=4)

ax_coordinates.legend(loc="upper right")

coordinate_readout = ax_coordinates.text(
    1.04, 0.50,
    "",
    transform=ax_coordinates.transAxes,
    ha="left",
    va="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9)
)


# ----------------------------- Plotting Time-evolution of transition probabilities -----------------------------
ax_probabilities.set_title("Time-Evolution of Qubit-state probabilities (Rotating frame)", pad=10)
ax_probabilities.set_xlim(sequence_param["t_points"][0]*1e9, sequence_param["t_points"][-1]*1e9)
ax_probabilities.set_ylim(0, 1.1)
ax_probabilities.set_xlabel("Time (ns)", fontsize=10)
ax_probabilities.set_ylabel("Probability", fontsize=10)
ax_probabilities.minorticks_on()
ax_probabilities.grid(True, which="major", linestyle="-", linewidth=0.8)
ax_probabilities.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

prob_0, = ax_probabilities.plot([], [], label=r"$P_0(t)$")
prob_1, = ax_probabilities.plot([], [], label=r"$P_1(t)$")
prob_total, = ax_probabilities.plot([], [], color="black", label=r"$P_{\mathrm{total}}(t)$")

time_probability_cursor = ax_probabilities.axvline(
    sequence_param["t_points"][0]*1e9,
    color='r',
    linestyle='--',
    linewidth=1.5
)

probability_cursor, = ax_probabilities.plot([], [], 'ro', markersize=4)

ax_probabilities.legend(loc="upper right")

probability_readout = ax_probabilities.text(
    1.04, 0.50,
    "",
    transform=ax_probabilities.transAxes,
    ha="left",
    va="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9)
)


# ----------------------------- Animating the Bloch sphere and all Time plots -----------------------------
interval = 10       # Time between frames in milliseconds

def update_animation(frame):
    global paused, animation_finished

    # Updating Bloch sphere in the laboratory reference frame over time
    t_current = sequence_param["t_points"][frame] * 1e9
    b_lab.clear()
    b_lab.point_color = ['r']
    b_lab.vector_color = ['b']

    b_lab.add_states(qubit_states_lab[frame])
    b_lab.add_points([x_lab[frame], y_lab[frame], z_lab[frame]], 's')
    b_lab.add_points([x_lab[:frame+1], y_lab[:frame+1], z_lab[:frame+1]], 'l')
    b_lab.make_sphere()

    # Updating Bloch sphere in the rotating reference frame over time
    b_rot.clear()
    b_rot.point_color = ['r']
    b_rot.vector_color = ['b']

    b_rot.add_states(qubit_states_rot[frame])
    b_rot.add_points([x_rot[frame], y_rot[frame], z_rot[frame]], 's')
    b_rot.add_points([x_rot[:frame+1], y_rot[:frame+1], z_rot[:frame+1]], 'l')
    b_rot.make_sphere()

    # Updating I/Q Coupling and Detuning signals over time frames 
    coupling_I_amplitude.set_data(sequence_param["t_points"][:frame+1]*1e9, sequence_param["V_I_amplitude"][:frame+1]*1e6)
    coupling_Q_amplitude.set_data(sequence_param["t_points"][:frame+1]*1e9, sequence_param["V_Q_amplitude"][:frame+1]*1e6)
    coupling_signal.set_data(sequence_param["t_points"][:frame+1]*1e9, sequence_param["V_signal_lab"][:frame+1]*1e6)
    detuning_line.set_data(sequence_param["t_points"][:frame+1]*1e9, sequence_param["Delta_eff_signal"][:frame+1]*1e6)

    time_tuning_cursor.set_xdata([t_current, t_current])

    tuning_cursor.set_data(
        [t_current, t_current, t_current, t_current],
        [sequence_param["V_I_amplitude"][frame]*1e6, sequence_param["V_Q_amplitude"][frame]*1e6, sequence_param["V_signal_lab"][frame]*1e6, sequence_param["Delta_eff_signal"][frame]*1e6]
    )

    tuning_readout.set_text(
        "Cursor\n"
        f"t = {t_current:.3f} ns\n"
        f"V_I = {sequence_param['V_I_amplitude'][frame]*1e6:.3f} ueV\n"
        f"V_Q = {sequence_param['V_Q_amplitude'][frame]*1e6:.3f} ueV\n"
        f"V_sig = {sequence_param['V_signal_lab'][frame]*1e6:.3f} ueV\n"
        f"Delta_eff = {sequence_param['Delta_eff_signal'][frame]*1e6:.3f} ueV\n"
        f"Omega_0 = {sequence_param['Omega_0'][frame]/(2*np.pi)*1e-9:.3f} GHz\n"
        f"Omega_d = {sequence_param['Omega_d'][frame]/(2*np.pi)*1e-9:.3f} GHz\n"
        f"Omega_0 - Omega_d = {sequence_param['Omega_eff'][frame]/(2*np.pi)*1e-9:.3f} GHz"
    )

    # Updating Bloch-vector coordinates over time frames
    coordinate_x.set_data(sequence_param["t_points"][:frame+1]*1e9, x_rot[:frame+1])
    coordinate_y.set_data(sequence_param["t_points"][:frame+1]*1e9, y_rot[:frame+1])
    coordinate_z.set_data(sequence_param["t_points"][:frame+1]*1e9, z_rot[:frame+1])
    r_magnitude.set_data(sequence_param["t_points"][:frame+1]*1e9, r_mag_rot[:frame+1])

    time_coordinate_cursor.set_xdata([t_current, t_current])

    coordinate_cursor.set_data(
        [t_current, t_current, t_current, t_current],
        [x_rot[frame], y_rot[frame], z_rot[frame], r_mag_rot[frame]]
    )

    coordinate_readout.set_text(
        "Cursor\n"
        f"t = {t_current:.3f} ns\n"
        f"<sigma_x> = {x_rot[frame]:.3f}\n"
        f"<sigma_y> = {y_rot[frame]:.3f}\n"
        f"<sigma_z> = {z_rot[frame]:.3f}\n"
        f"|r| = {r_mag_rot[frame]:.3f}"
    )

    # Updating qubit state probabilities over time frames
    prob_0.set_data(sequence_param["t_points"][:frame+1]*1e9, p0_rot[:frame+1])
    prob_1.set_data(sequence_param["t_points"][:frame+1]*1e9, p1_rot[:frame+1])
    prob_total.set_data(sequence_param["t_points"][:frame+1]*1e9, p_total_rot[:frame+1])

    time_probability_cursor.set_xdata([t_current, t_current])

    probability_cursor.set_data(
        [t_current, t_current, t_current],
        [p0_rot[frame], p1_rot[frame], p_total_rot[frame]]
    )

    probability_readout.set_text(
        "Cursor\n"
        f"t = {t_current:.3f} ns\n"
        f"P0 = {p0_rot[frame]:.3f}\n"
        f"P1 = {p1_rot[frame]:.3f}\n"
        f"P_total = {p_total_rot[frame]:.3f}\n"
        f"Omega_R = {sequence_param['Omega_R'][frame]/(2*np.pi)*1e-9:.3f} GHz"
    )

    fig_time.canvas.draw_idle()

    if frame >= len(sequence_param["t_points"]) - 1:
        anim.event_source.stop()
        button.label.set_text("Play")
        paused = True
        animation_finished = True

anim = FuncAnimation(
    fig_bloch,
    update_animation,
    frames=len(sequence_param["t_points"]),
    interval=interval,
    blit=False,
    repeat=False
)
    

# ----------------------------- Pause/Play button -----------------------------
paused = True
animation_finished = False
anim.event_source.stop()

def toggle_animation(event):
    global paused, animation_finished

    if paused:
        if animation_finished:
            anim.frame_seq = anim.new_frame_seq()
            animation_finished = False

        anim.event_source.start()
        button.label.set_text("Pause")
        paused = False
    else:
        anim.event_source.stop()
        button.label.set_text("Play")
        paused = True

button_ax = fig_time.add_axes([0.45, 0.03, 0.1, 0.05])
button = Button(button_ax, "Play")

button.on_clicked(toggle_animation)

# Stop the animation immediately after the first draw event
def pause_animation_on_first_draw(event):
    global paused, animation_finished, first_draw_cid

    if event.canvas == fig_bloch.canvas:
        anim.event_source.stop()
        button.label.set_text("Play")
        paused = True
        animation_finished = False
        fig_bloch.canvas.mpl_disconnect(first_draw_cid)

first_draw_cid = fig_bloch.canvas.mpl_connect(
    "draw_event",
    pause_animation_on_first_draw
)

plt.show()