# ----------------------------- Import libraries -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from qutip import Bloch, Qobj, basis, expect, sigmax, sigmay, sigmaz, ket2dm


# ----------------------------- Parameters -----------------------------
hbar = 6.582119569e-16        # REDUCED Planck constant in eV*s
E_mean = 0                    # Mean unperturbed energy of the two basis states |0> and |1>

def tuning_parameters(gate_name, Delta, V_I_mag, V_Q_mag):
    V_mag = np.sqrt(V_I_mag**2 + V_Q_mag**2)
    Omega_0 = 2*Delta/hbar 
    Omega_r = 2*np.sqrt(Delta**2 + V_mag**2)/hbar 
    
    if gate_name == 'X':
        t_duration = (np.pi/2)*(2/Omega_r)
    
    elif gate_name == 'Y':  
        t_duration = (np.pi/2)*(2/Omega_r)

    elif gate_name == 'Z':
        t_duration = (np.pi/2)*(2/Omega_r)

    elif gate_name == 'S':      
        t_duration = (np.pi/4)*(2/Omega_r)

    elif gate_name == 'T':      
        t_duration = (np.pi/8)*(2/Omega_r)

    elif gate_name == 'H':      
        t_duration = (np.pi/2)*(2/Omega_r)

    elif gate_name == 'I':   
        t_duration = 0*(2/Omega_r)

    t_points = np.linspace(0, t_duration, 100) 
    Delta_signal = np.full_like(t_points, Delta)
    V_I_amplitude = np.full_like(t_points, V_I_mag)
    V_Q_amplitude = np.full_like(t_points, V_Q_mag)
    
    return {
        "Omega_r": Omega_r, 
        "Omega_0": Omega_0, 
        "t_points": t_points, 
        "Delta_signal": Delta_signal, 
        "V_I_amplitude": V_I_amplitude, 
        "V_Q_amplitude": V_Q_amplitude
        }


# ----------------------------- Unitary time-evolution operator for statically coupled two-level system ----------------------------- 
def unitary_operator(t, hbar, E_mean, V_I_amplitude, V_Q_amplitude, Omega_r, Omega_0):        
    U = np.exp(-1j*E_mean*t/hbar) * np.array([[np.cos(Omega_r*t/2) + 1j*(Omega_0/Omega_r)*np.sin(Omega_r*t/2), 
                                            -1j*(V_I_amplitude - 1j*V_Q_amplitude)*(2/(hbar*Omega_r))*np.sin(Omega_r*t/2)],
                                            [-1j*(V_I_amplitude + 1j*V_Q_amplitude)*(2/(hbar*Omega_r))*np.sin(Omega_r*t/2), 
                                            np.cos(Omega_r*t/2) - 1j*(Omega_0/Omega_r)*np.sin(Omega_r*t/2)]])
    
    return U


# ----------------------------- Time evolution of Qubit state -----------------------------
# Select the single-qubit gates by uncommenting/commenting
gate_sequence = [
    #{"gate": "H", "Delta": -10e-6,  "V_I_mag": -10e-6, "V_Q_mag": 10e-9},
    {"gate": "X", "Delta": 10e-9, "V_I_mag": 10e-6, "V_Q_mag": 10e-9},
    #{"gate": "Y", "Delta": 10e-9, "V_I_mag": 10e-9, "V_Q_mag": 10e-6},
    #{"gate": "Z", "Delta": 10e-6, "V_I_mag": 10e-9, "V_Q_mag": 10e-9},
    #{"gate": "S", "Delta": -10e-6, "V_I_mag": 10e-9, "V_Q_mag": 10e-9},
    #{"gate": "T", "Delta": -10e-6, "V_I_mag": 10e-9, "V_Q_mag": 10e-9},
    #{"gate": "H", "Delta": -10e-6,  "V_I_mag": -10e-6, "V_Q_mag": 10e-9},
]

# Select an initial state by uncommenting/commenting
initial_state = basis(2, 0)                                      # initial state |0>
#initial_state = basis(2, 1)                                      # initial state |1>
#initial_state = (basis(2, 0) + (1+0j)*basis(2, 1)).unit()        # initial state |+>
#initial_state = (basis(2, 0) + (-1+0j)*basis(2, 1)).unit()       # initial state |->
#initial_state = (basis(2, 0) + (0+1j)*basis(2, 1)).unit()        # initial state |+i>
#initial_state = (basis(2, 0) + (0-1j)*basis(2, 1)).unit()        # initial state |-i>

def gate_operation(initial_state, gate_sequence):
    t_accumulated = 0
    input_state = initial_state
    
    t_points_sequence = []
    Delta_signal_sequence = []
    V_I_amplitude_sequence = []
    V_Q_amplitude_sequence = []
    Omega_r_sequence = []
    Omega_0_sequence = []
    qubit_states = []

    gate_info = []

    for gate_settings in gate_sequence:
        gate = gate_settings["gate"]
        Delta = gate_settings["Delta"]
        V_I_mag = gate_settings["V_I_mag"]
        V_Q_mag = gate_settings["V_Q_mag"]

        gate_param = tuning_parameters(gate, Delta, V_I_mag, V_Q_mag)
        t_current_gate = gate_param["t_points"]

        U_gate = unitary_operator(
            t_current_gate, 
            hbar, 
            E_mean, 
            gate_param["V_I_amplitude"], 
            gate_param["V_Q_amplitude"], 
            gate_param["Omega_r"], 
            gate_param["Omega_0"]
            )

        for i in range(len(t_current_gate)):
            state_t = Qobj(U_gate[:, :, i]) * input_state

            t_sequence = t_accumulated + t_current_gate[i]

            qubit_states.append(state_t)
            t_points_sequence.append(t_sequence)
            Delta_signal_sequence.append(gate_param["Delta_signal"][i])
            V_I_amplitude_sequence.append(gate_param["V_I_amplitude"][i])
            V_Q_amplitude_sequence.append(gate_param["V_Q_amplitude"][i])
            Omega_r_sequence.append(gate_param["Omega_r"])
            Omega_0_sequence.append(gate_param["Omega_0"])

        gate_info.append({
            "gate": gate,
            "Delta": Delta,
            "V_I_mag": V_I_mag,
            "V_Q_mag": V_Q_mag,
            "Omega_r": gate_param["Omega_r"],
            "Omega_0": gate_param["Omega_0"],
            "t_start": t_accumulated,
            "t_end": t_accumulated + t_current_gate[-1],
            "t_duration": t_current_gate[-1]
        })

        input_state = Qobj(U_gate[:, :, -1]) * input_state

        t_accumulated = t_accumulated + t_current_gate[-1]

    return {
        "t_points": np.array(t_points_sequence),
        "Delta_signal": np.array(Delta_signal_sequence),
        "V_I_amplitude": np.array(V_I_amplitude_sequence),
        "V_Q_amplitude": np.array(V_Q_amplitude_sequence),
        "Omega_r": np.array(Omega_r_sequence),
        "Omega_0": np.array(Omega_0_sequence),
        "qubit_states": qubit_states,
        "gate_info": gate_info
    }

sequence_param = gate_operation(initial_state, gate_sequence)
qubit_states = sequence_param["qubit_states"]

for info in sequence_param["gate_info"]:
    print("\nGate =", info["gate"])
    print("Delta =", info["Delta"]*1e6, "ueV")
    print("V_I_mag =", info["V_I_mag"]*1e6, "ueV")
    print("V_Q_mag =", info["V_Q_mag"]*1e6, "ueV")
    print("t_start =", info["t_start"]*1e9, "ns")
    print("t_end =", info["t_end"]*1e9, "ns")
    print("t_duration =", info["t_duration"]*1e9, "ns")
    print("Omega_r =", (info["Omega_r"]/(2*np.pi))*1e-9, "GHz")
    print("Omega_0 =", (info["Omega_0"]/(2*np.pi))*1e-9, "GHz")

x = np.array([expect(sigmax(), state).real for state in qubit_states])
y = np.array([expect(sigmay(), state).real for state in qubit_states])
z = np.array([expect(sigmaz(), state).real for state in qubit_states])
r_mag = np.sqrt(x**2 + y**2 + z**2)

p0 = np.array([ket2dm(state)[0, 0].real for state in qubit_states])
p1 = np.array([ket2dm(state)[1, 1].real for state in qubit_states])
p_total = p0 + p1


# ----------------------------- Creating Bloch sphere and all Time-plots-----------------------------
plt.close('all')

fig_bloch = plt.figure(figsize=(7, 7))
ax_bloch = fig_bloch.add_subplot(111, projection='3d')
b = Bloch(axes=ax_bloch)
fig_bloch.suptitle("Time-Evolution of Qubit State", fontsize=14, y=0.98)
b.font_size = 14

fig_time, (ax_tuning, ax_coordinates, ax_probabilities) = plt.subplots(
    3, 1,
    figsize=(10, 14),
    sharex=True
)
fig_time.subplots_adjust(hspace=0.45, bottom=0.15, right=0.72)


# ----------------------------- Plotting Time-evolution of Coupling V(t) and Detuning Delta signals -----------------------------
ax_tuning.set_title("Time-Evolution of Coupling and Detuning signals", fontsize=12, pad=10)
ax_tuning.set_xlim(sequence_param["t_points"][0]*1e9, sequence_param["t_points"][-1]*1e9)
ax_tuning.set_ylim(-11, 11)
ax_tuning.set_ylabel("Energy (ueV)", fontsize=10)
ax_tuning.minorticks_on()
ax_tuning.grid(True, which="major", linestyle="-", linewidth=0.8)
ax_tuning.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

coupling_I_amplitude, = ax_tuning.plot([], [], label=r"$V_{I_{mag}}$")
coupling_Q_amplitude, = ax_tuning.plot([], [], label=r"$V_{Q_{mag}}$")
detuning_line, = ax_tuning.plot([], [], label=r"$\Delta$")

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
ax_coordinates.set_title("Time-Evolution of Bloch-vector coordinates", fontsize=12, pad=10)
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
ax_probabilities.set_title("Time-Evolution of Qubit-state probabilities", pad=10)
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
interval = 10      

def update_animation(frame):
    global paused, animation_finished

    t_current = sequence_param["t_points"][frame] * 1e9
    b.clear()
    b.point_color = ['r']
    b.vector_color = ['b']

    b.add_states(qubit_states[frame])
    b.add_points([x[frame], y[frame], z[frame]], 's')
    b.add_points([x[:frame+1], y[:frame+1], z[:frame+1]], 'l')
    b.make_sphere()

    # Updating I/Q Coupling and Detuning signals over time frames 
    coupling_I_amplitude.set_data(sequence_param["t_points"][:frame+1]*1e9, sequence_param["V_I_amplitude"][:frame+1]*1e6)
    coupling_Q_amplitude.set_data(sequence_param["t_points"][:frame+1]*1e9, sequence_param["V_Q_amplitude"][:frame+1]*1e6)
    detuning_line.set_data(sequence_param["t_points"][:frame+1]*1e9, sequence_param["Delta_signal"][:frame+1]*1e6)

    time_tuning_cursor.set_xdata([t_current, t_current])

    tuning_cursor.set_data(
        [t_current, t_current, t_current],
        [sequence_param["V_I_amplitude"][frame]*1e6, sequence_param["V_Q_amplitude"][frame]*1e6, sequence_param["Delta_signal"][frame]*1e6]
    )

    tuning_readout.set_text(
        "Cursor\n"
        f"t = {t_current:.3f} ns\n"
        f"V_I = {sequence_param['V_I_amplitude'][frame]*1e6:.3f} ueV\n"
        f"V_Q = {sequence_param['V_Q_amplitude'][frame]*1e6:.3f} ueV\n"
        f"Delta = {sequence_param['Delta_signal'][frame]*1e6:.3f} ueV\n"
        f"Omega_0 = {sequence_param['Omega_0'][frame]/(2*np.pi)*1e-9:.3f} GHz"
    )

    # Updating Bloch-vector coordinates over time frames
    coordinate_x.set_data(sequence_param["t_points"][:frame+1]*1e9, x[:frame+1])
    coordinate_y.set_data(sequence_param["t_points"][:frame+1]*1e9, y[:frame+1])
    coordinate_z.set_data(sequence_param["t_points"][:frame+1]*1e9, z[:frame+1])
    r_magnitude.set_data(sequence_param["t_points"][:frame+1]*1e9, r_mag[:frame+1])

    time_coordinate_cursor.set_xdata([t_current, t_current])

    coordinate_cursor.set_data(
        [t_current, t_current, t_current, t_current],
        [x[frame], y[frame], z[frame], r_mag[frame]]
    )

    coordinate_readout.set_text(
        "Cursor\n"
        f"t = {t_current:.3f} ns\n"
        f"<sigma_x> = {x[frame]:.3f}\n"
        f"<sigma_y> = {y[frame]:.3f}\n"
        f"<sigma_z> = {z[frame]:.3f}\n"
        f"|r| = {r_mag[frame]:.3f}"
    )

    # Updating qubit state probabilities over time frames
    prob_0.set_data(sequence_param["t_points"][:frame+1]*1e9, p0[:frame+1])
    prob_1.set_data(sequence_param["t_points"][:frame+1]*1e9, p1[:frame+1])
    prob_total.set_data(sequence_param["t_points"][:frame+1]*1e9, p_total[:frame+1])

    time_probability_cursor.set_xdata([t_current, t_current])

    probability_cursor.set_data(
        [t_current, t_current, t_current],
        [p0[frame], p1[frame], p_total[frame]]
    )

    probability_readout.set_text(
        "Cursor\n"
        f"t = {t_current:.3f} ns\n"
        f"P0 = {p0[frame]:.3f}\n"
        f"P1 = {p1[frame]:.3f}\n"
        f"P_total = {p_total[frame]:.3f}\n"
        f"Omega_r = {sequence_param['Omega_r'][frame]/(2*np.pi)*1e-9:.3f} GHz"
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
