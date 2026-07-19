# ----------------------------- Import libraries -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from qutip import Bloch, Qobj, basis, expect, sigmax, sigmay, sigmaz, ket2dm


# ----------------------------- Parameters -----------------------------
hbar = 6.582119569e-16        # REDUCED Planck constant in eV*s
E_mean = 0                    # Mean unperturbed energy of the two basis states |0> and |1>

def tuning_parameters(t_duration, Delta, V_I_mag, V_Q_mag):
    V_mag = np.sqrt(V_I_mag**2 + V_Q_mag**2)
    Omega_0 = 2*Delta/hbar 
    Omega_r = 2*np.sqrt(Delta**2 + V_mag**2)/hbar 

    t_points = np.linspace(0, t_duration, 50) 
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
gates_in_parallel = [
    {
        "t_duration": 2.5e-11,
        "Delta": -1e-6,
        "V_I_mag": 10e-6,
        "V_Q_mag": 1e-6,
        "initial_state": basis(2, 0),
        "initial_state_name": "|0>",
        "initial_state_label": r"$|0\rangle$"
    },
    {
        "t_duration": 2.5e-11,
        "Delta": -1e-6,
        "V_I_mag": 20e-6,
        "V_Q_mag": 1e-6,
        "initial_state": basis(2, 0),
        "initial_state_name": "|0>",
        "initial_state_label": r"$|0\rangle$"
    },
    {
        "t_duration": 2.5e-11,
        "Delta": -1e-6,
        "V_I_mag": 30e-6,
        "V_Q_mag": 1e-6,
        "initial_state": basis(2, 0),
        "initial_state_name": "|0>",
        "initial_state_label": r"$|0\rangle$"
    },
    {
        "t_duration": 2.5e-11,
        "Delta": -1e-6,
        "V_I_mag": 40e-6,
        "V_Q_mag": 1e-6,
        "initial_state": basis(2, 0),
        "initial_state_name": "|0>",
        "initial_state_label": r"$|0\rangle$"
    },
]

def gate_operation(gate_settings):
    t_duration = gate_settings["t_duration"]
    Delta = gate_settings["Delta"]
    V_I_mag = gate_settings["V_I_mag"]
    V_Q_mag = gate_settings["V_Q_mag"]
    initial_state = gate_settings["initial_state"]
    initial_state_name = gate_settings["initial_state_name"]
    initial_state_label = gate_settings["initial_state_label"]

    gate_param = tuning_parameters(t_duration, Delta, V_I_mag, V_Q_mag)
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

    qubit_states = []

    for i in range(len(t_current_gate)):
        state_t = Qobj(U_gate[:, :, i]) * initial_state

        qubit_states.append(state_t)

    gate_info = {
        "t_duration": t_duration,
        "Delta": Delta,
        "V_I_mag": V_I_mag,
        "V_Q_mag": V_Q_mag,
        "Omega_r": gate_param["Omega_r"],
        "Omega_0": gate_param["Omega_0"],
        "t_start": 0,
        "t_end": t_current_gate[-1],
        "initial_state_name": initial_state_name,
        "initial_state_label": initial_state_label
    }

    return {
        "t_points": np.array(t_current_gate),
        "Delta_signal": np.array(gate_param["Delta_signal"]),
        "V_I_amplitude": np.array(gate_param["V_I_amplitude"]),
        "V_Q_amplitude": np.array(gate_param["V_Q_amplitude"]),
        "qubit_states": qubit_states,
        "gate_info": gate_info
    }


def compute_bloch_data(qubit_states):
    x = np.array([expect(sigmax(), state).real for state in qubit_states])
    y = np.array([expect(sigmay(), state).real for state in qubit_states])
    z = np.array([expect(sigmaz(), state).real for state in qubit_states])
    r_mag = np.sqrt(x**2 + y**2 + z**2)

    p0 = np.array([ket2dm(state)[0, 0].real for state in qubit_states])
    p1 = np.array([ket2dm(state)[1, 1].real for state in qubit_states])
    p_total = p0 + p1

    return {
        "x": x,
        "y": y,
        "z": z,
        "r_mag": r_mag,
        "p0": p0,
        "p1": p1,
        "p_total": p_total
    }

parallel_gate_results = []

for gate_number, gate_settings in enumerate(gates_in_parallel, start=1):
    gate_param = gate_operation(gate_settings)
    gate_param["bloch_data"] = compute_bloch_data(gate_param["qubit_states"])
    parallel_gate_results.append(gate_param)


# ----------------------------- Creating Bloch spheres -----------------------------
plt.close('all')

interval = 10       # Time between frames in milliseconds

animation_controls = []        # Keeps all animations and buttons alive while the figures are open
plot_controls = []             # Stores the artists that must be updated for each gate

if len(parallel_gate_results) == 0:
    print("No gates selected. Add at least one gate dictionary to gate_sequence.")

else:
    fig_bloch = plt.figure(figsize=(16, 4), num="Bloch Spheres and Energy-Delta Diagram")
    fig_bloch.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.86,
        bottom=0.18,
        wspace=0.00,
        hspace=0.25
    )

    combined_grid = fig_bloch.add_gridspec(
        1, 5,
        width_ratios=[1.00, 0.85, 0.85, 0.85, 0.85],
        wspace=0.00
    )

    bloch_axes = []
    bloch_spheres = []

    energy_curve_colours = {
        10: "blue",
        20: "orange",
        30: "green",
        40: "purple"
    }

    for subplot_index in range(4):
        ax_bloch = fig_bloch.add_subplot(combined_grid[0, subplot_index + 1], projection='3d')
        bloch_axes.append(ax_bloch)

        if subplot_index < len(parallel_gate_results):
            b = Bloch(axes=ax_bloch)
            b.font_size = 12
            bloch_spheres.append(b)
        else:
            ax_bloch.set_axis_off()

    for subplot_index in range(4):
        if subplot_index < len(parallel_gate_results):
            gate_param = parallel_gate_results[subplot_index]

            qubit_states = gate_param["qubit_states"]
            bloch_data = gate_param["bloch_data"]
            x = bloch_data["x"]
            y = bloch_data["y"]
            z = bloch_data["z"]
            r_mag = bloch_data["r_mag"]
            p0 = bloch_data["p0"]
            p1 = bloch_data["p1"]
            p_total = bloch_data["p_total"]
            V_I_mag = gate_param["gate_info"]["V_I_mag"]
            V_Q_mag = gate_param["gate_info"]["V_Q_mag"]
            V_mag = np.sqrt(V_I_mag**2 + V_Q_mag**2)
            V_mag_ueV = int(round(V_mag*1e6))
            curve_colour = energy_curve_colours[V_mag_ueV]

            plot_controls.append({
                "gate_param": gate_param,
                "bloch_sphere": bloch_spheres[subplot_index],
                "bloch_axis": bloch_axes[subplot_index],
                "qubit_states": qubit_states,
                "x": x,
                "y": y,
                "z": z,
                "r_mag": r_mag,
                "p0": p0,
                "p1": p1,
                "p_total": p_total,
                "curve_colour": curve_colour
            })


    # ----------------------------- Creating Energy-Delta diagrams -----------------------------
    ax_energy = fig_bloch.add_subplot(combined_grid[0, 0])

    Delta_values = [gate_param["gate_info"]["Delta"] for gate_param in parallel_gate_results]
    V_mag_values = [np.sqrt(gate_param["gate_info"]["V_I_mag"]**2 + gate_param["gate_info"]["V_Q_mag"]**2) for gate_param in parallel_gate_results]

    Delta_limit = 1.1*max(
        max(np.abs(Delta_values)),
        max(np.abs(V_mag_values)),
        1e-12
    )

    Delta_range = np.linspace(-Delta_limit, Delta_limit, 500)

    ax_energy.plot(Delta_range*1e6, Delta_range*1e6, color="black", linestyle="--", label=r"$|V|=0$ ueV")
    ax_energy.plot(Delta_range*1e6, -Delta_range*1e6, color="black", linestyle="--")

    for simulation_index, gate_param in enumerate(parallel_gate_results, start=1):
        Delta = gate_param["gate_info"]["Delta"]
        V_I_mag = gate_param["gate_info"]["V_I_mag"]
        V_Q_mag = gate_param["gate_info"]["V_Q_mag"]
        V_mag = np.sqrt(V_I_mag**2 + V_Q_mag**2)
        V_mag_ueV = int(round(V_mag*1e6))
        curve_colour = energy_curve_colours[V_mag_ueV]

        E_plus = E_mean + np.sqrt(Delta_range**2 + V_mag**2)
        E_minus = E_mean - np.sqrt(Delta_range**2 + V_mag**2)
        E_plus_current = E_mean + np.sqrt(Delta**2 + V_mag**2)
        E_minus_current = E_mean - np.sqrt(Delta**2 + V_mag**2)

        ax_energy.plot(
            Delta_range*1e6,
            E_plus*1e6,
            color=curve_colour,
            label=fr"$|V|={V_mag_ueV}$ ueV"
        )
        ax_energy.plot(
            Delta_range*1e6,
            E_minus*1e6,
            color=curve_colour
        )

        ax_energy.axvline(Delta*1e6, color="red", linestyle=":", linewidth=1.0, zorder=1)
        ax_energy.plot([Delta*1e6], [E_plus_current*1e6], 'o', color=curve_colour, markersize=4, zorder=5)
        ax_energy.plot([Delta*1e6], [E_minus_current*1e6], 'o', color=curve_colour, markersize=4, zorder=5)

    ax_energy.set_xlabel(r"$\Delta$ (ueV)", fontsize=11)
    ax_energy.set_ylabel(r"$E-\overline{E}$ (ueV)", fontsize=11, labelpad=2)
    ax_energy.tick_params(axis='both', labelsize=9)
    ax_energy.minorticks_on()
    ax_energy.grid(True, which="major", linestyle="-", linewidth=0.6)
    ax_energy.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.7)
    ax_energy.legend(loc="best", fontsize=6)


    # ----------------------------- Creating combined Time plots -----------------------------
    fig_time = plt.figure(figsize=(16, 6), num="Parallel Time Plots")
    fig_time.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.91,
        bottom=0.09,
        wspace=0.30,
        hspace=0.35
    )

    time_grid = fig_time.add_gridspec(
        2, 4,
        wspace=0.30,
        hspace=0.35
    )

    for subplot_index in range(4):
        if subplot_index < len(plot_controls):
            controls = plot_controls[subplot_index]
            gate_param = controls["gate_param"]
            t_points_ns = gate_param["t_points"]*1e9
            curve_colour = controls["curve_colour"]
            V_mag = np.sqrt(
                gate_param["gate_info"]["V_I_mag"]**2
                + gate_param["gate_info"]["V_Q_mag"]**2
            )
            V_mag_ueV = int(round(V_mag*1e6))

            ax_coordinates = fig_time.add_subplot(time_grid[0, subplot_index])
            ax_probabilities = fig_time.add_subplot(time_grid[1, subplot_index], sharex=ax_coordinates)

            ax_coordinates.set_title(
                fr"$|V|={V_mag_ueV}$ ueV",
                fontsize=12,
                color=curve_colour,
                pad=10
            )
            ax_coordinates.set_xlim(t_points_ns[0], t_points_ns[-1])
            ax_coordinates.set_ylim(-1.1, 1.1)
            ax_coordinates.set_ylabel("Pauli Expectation Values", fontsize=9)
            ax_coordinates.minorticks_on()
            ax_coordinates.grid(True, which="major", linestyle="-", linewidth=0.8)
            ax_coordinates.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

            coordinate_x, = ax_coordinates.plot([], [], label=r"$\langle\sigma_x\rangle$")
            coordinate_y, = ax_coordinates.plot([], [], label=r"$\langle\sigma_y\rangle$")
            coordinate_z, = ax_coordinates.plot([], [], label=r"$\langle\sigma_z\rangle$")
            r_magnitude, = ax_coordinates.plot([], [], color="black", label=r"$|\mathbf{r}|$")
            time_coordinate_cursor = ax_coordinates.axvline(
                t_points_ns[0],
                color='r',
                linestyle='--',
                linewidth=1.2
            )
            coordinate_cursor, = ax_coordinates.plot([], [], 'ro', markersize=3)
            ax_coordinates.legend(loc="upper right", fontsize=7)

            ax_probabilities.set_xlim(t_points_ns[0], t_points_ns[-1])
            ax_probabilities.set_ylim(0, 1.1)
            ax_probabilities.set_xlabel("Time (ns)", fontsize=9)
            ax_probabilities.set_ylabel("Probability", fontsize=9)
            ax_probabilities.minorticks_on()
            ax_probabilities.grid(True, which="major", linestyle="-", linewidth=0.8)
            ax_probabilities.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

            prob_0, = ax_probabilities.plot([], [], label=r"$P_0(t)$")
            prob_1, = ax_probabilities.plot([], [], label=r"$P_1(t)$")
            prob_total, = ax_probabilities.plot([], [], color="black", label=r"$P_{\mathrm{total}}(t)$")
            time_probability_cursor = ax_probabilities.axvline(
                t_points_ns[0],
                color='r',
                linestyle='--',
                linewidth=1.2
            )
            probability_cursor, = ax_probabilities.plot([], [], 'ro', markersize=3)
            ax_probabilities.legend(loc="upper right", fontsize=7)

            controls.update({
                "coordinate_x": coordinate_x,
                "coordinate_y": coordinate_y,
                "coordinate_z": coordinate_z,
                "r_magnitude": r_magnitude,
                "time_coordinate_cursor": time_coordinate_cursor,
                "coordinate_cursor": coordinate_cursor,
                "prob_0": prob_0,
                "prob_1": prob_1,
                "prob_total": prob_total,
                "time_probability_cursor": time_probability_cursor,
                "probability_cursor": probability_cursor
            })
        else:
            for row_index in range(2):
                ax_empty = fig_time.add_subplot(time_grid[row_index, subplot_index])
                ax_empty.set_axis_off()


    # ----------------------------- Animating the combined Bloch sphere and combined Time plots -----------------------------
    n_frames = max(len(gate_param["t_points"]) for gate_param in parallel_gate_results)

    def update_animation(frame):
        global paused, animation_finished

        for controls in plot_controls:
            gate_param = controls["gate_param"]
            frame_i = min(frame, len(gate_param["t_points"]) - 1)

            qubit_states = controls["qubit_states"]
            x = controls["x"]
            y = controls["y"]
            z = controls["z"]
            r_mag = controls["r_mag"]
            p0 = controls["p0"]
            p1 = controls["p1"]
            p_total = controls["p_total"]

            b = controls["bloch_sphere"]
            ax_bloch = controls["bloch_axis"]
            curve_colour = controls["curve_colour"]

            b.clear()
            b.point_color = [curve_colour]
            b.vector_color = [curve_colour]

            b.add_states(qubit_states[frame_i])
            b.add_points([x[frame_i], y[frame_i], z[frame_i]], 's')
            b.add_points([x[:frame_i+1], y[:frame_i+1], z[:frame_i+1]], 'l')
            b.make_sphere()

            t_current = gate_param["t_points"][frame_i]*1e9
            t_points_ns = gate_param["t_points"][:frame_i+1]*1e9

            controls["coordinate_x"].set_data(t_points_ns, x[:frame_i+1])
            controls["coordinate_y"].set_data(t_points_ns, y[:frame_i+1])
            controls["coordinate_z"].set_data(t_points_ns, z[:frame_i+1])
            controls["r_magnitude"].set_data(t_points_ns, r_mag[:frame_i+1])
            controls["time_coordinate_cursor"].set_xdata([t_current, t_current])
            controls["coordinate_cursor"].set_data(
                [t_current, t_current, t_current, t_current],
                [x[frame_i], y[frame_i], z[frame_i], r_mag[frame_i]]
            )

            controls["prob_0"].set_data(t_points_ns, p0[:frame_i+1])
            controls["prob_1"].set_data(t_points_ns, p1[:frame_i+1])
            controls["prob_total"].set_data(t_points_ns, p_total[:frame_i+1])
            controls["time_probability_cursor"].set_xdata([t_current, t_current])
            controls["probability_cursor"].set_data(
                [t_current, t_current, t_current],
                [p0[frame_i], p1[frame_i], p_total[frame_i]]
            )

        fig_time.canvas.draw_idle()

        if frame >= n_frames - 1:
            anim.event_source.stop()
            button.label.set_text("Play")
            paused = True
            animation_finished = True

    anim = FuncAnimation(
        fig_bloch,
        update_animation,
        frames=n_frames,
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

    button_ax = fig_bloch.add_axes([0.46, 0.005, 0.08, 0.05])
    button = Button(button_ax, "Play")

    button.on_clicked(toggle_animation)

    
    def pause_animation_on_first_draw(event):
        global paused, animation_finished

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
