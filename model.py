from dataclasses import dataclass
from collections import namedtuple
import matplotlib.pyplot as plt
import math
import numpy as np

from pint import UnitRegistry


############################################
# Define basic helper functions
############################################
def calc_area(diameter):
    return math.pi * diameter**2 / 4


def calc_circumference(diameter):
    return math.pi * diameter


def calc_velocity_head(velocity):
    return velocity * abs(velocity) / (2 * G)


def calc_dv_dt(acc_head, length):
    return G / length * acc_head


def calc_pipe_loss_coef(L):
    return f * L / D


def clamp(lower_bound, value, upper_bound):
    return max(lower_bound, min(value, upper_bound))


u = UnitRegistry()
u.meter_per_sec = u.meter / u.second
u.meter_per_sec_2 = u.meter / u.second**2

############################################
# Define constants for the system
############################################
G = 9.81 * u.meter_per_sec_2  # Gravitational constant
RHO = 1000 * u.kilogram / u.meter**3  # Density of water
L1 = 3000 * u.meter  # Conduit 1 length
L2 = 400 * u.meter  # Conduit 2 length
L3 = 100 * u.meter  # Conduit 3 length
f = 0.015  # Friction factor
D = 3 * u.meter  # Diameter
efficiency = 0.94  # Power absorbed by turbine at steady-state
HA = 400 * u.meter  # Head at A
zB = 80 * u.meter  # Elevation at B
HE = 100 * u.meter  # Head at E
minor_loss_coef = 5  # Minor loss coefficient for surge tank

############################################
# Define assumptions
############################################
exit_loss = 0.5  # Assuming there is a diffuser
# A typical value for steel pipes. Higher is more conservative.
speed_of_wave_in_conduit = 1200 * u.meter_per_sec

############################################
# Calculated paramters
############################################
A = calc_area(D)  # Cross-sectional area
k1 = f * L1 / D  # Velocity heads lost in conduit 1
k2 = f * L2 / D  # Velocity heads lost in conduit 2
k3 = f * L3 / D  # Velocity heads lost in conduit 3
L = L1 + L2 + L3  # Total length of conduits
H_sys = HA - HE  # Total head in system

############################################
# Steady state (ss) conditions
############################################
ss_H_valve = H_sys * efficiency
ss_H_loss = H_sys - ss_H_valve
ss_velocity_heads_lost = exit_loss + k1 + k2 + k3
ss_velocity_head = ss_H_loss / ss_velocity_heads_lost
ss_v = (2 * G * ss_velocity_head) ** 0.5
ss_Q = A * ss_v
ss_HS = HA - ss_velocity_head * k1
k_open = ss_H_valve / ss_velocity_head

# Check that results match hand calculations
assert round(ss_v, 2) == 4.43 * u.meter / u.second
assert ss_HS == 385 * u.meter


############################################
# Define design parameters
############################################
@dataclass(frozen=True)
class DesignParams:
    # Defaults form our selected design

    # Diameter of the surge tank (above widening point)
    surge_tank_diameter: u.Quantity = 10 * u.meters

    # Time to open/close the valve fully.
    valve_operation_time: u.Quantity = 5 * u.second

    # At this height the 3m conduit widens into to the surge tank
    elevation_of_widening: u.Quantity = 410 * u.meter

    freeboard_height: u.Quantity = 3 * u.meters


SELECTED_DESIGN = DesignParams()


############################################
# Define scenarios
############################################
@dataclass(frozen=True)
class Scenario:
    name: str
    start_tau: int
    end_tau: int
    initial_conditions: tuple  # (v1, v2, Hs)
    plotting_color: str


FULL_REJECTION = Scenario(
    "Full load rejection",
    start_tau=1,
    end_tau=0,
    initial_conditions=(ss_v, ss_v, ss_HS),
    plotting_color="#ff6361",
)
HALF_REJECTION = Scenario(
    "Half load rejection",
    start_tau=1,
    end_tau=0.5,
    initial_conditions=(ss_v, ss_v, ss_HS),
    plotting_color="#ffa600",
)
LOAD_ACCEPTANCE = Scenario(
    "Full load acceptance",
    start_tau=0,
    end_tau=1,
    initial_conditions=(0 * u.meter_per_sec, 0 * u.meter_per_sec, HA),
    plotting_color="#458B74",
)


############################################
# Define our model!
############################################
def calc_tau(t, scenario, design_params, settings):
    """Computes tau via linear interpolation for the given time and scenario."""
    delta_tau = (t - settings.start_event_time) / design_params.valve_operation_time
    if scenario.end_tau < scenario.start_tau:
        delta_tau *= -1
    tau = scenario.start_tau + delta_tau
    lower_bound = min(scenario.start_tau, scenario.end_tau)
    upper_bound = max(scenario.start_tau, scenario.end_tau)
    tau = clamp(lower_bound, tau, upper_bound)
    if tau < settings.tau_cutoff:
        tau = 0
    return tau


@u.check(u.second, u.meter_per_sec, u.meter_per_sec, u.meter, None, None, None)
def model_system(t, v1, v2, Hs, design_params, settings, scenario):
    """
    Computes for a given snapshot of the system the rate of change of each variable (v1, v2, HS)
    and some additional useful outputs (Hb, Pc, Pd, tau)
    """
    tau = calc_tau(t, scenario, design_params, settings)
    vs = v1 - v2
    if Hs < design_params.elevation_of_widening:
        tank_diameter = D
    else:
        tank_diameter = design_params.surge_tank_diameter
    dHs = vs * (D / tank_diameter) ** 2
    Hb = Hs + (
        minor_loss_coef + calc_pipe_loss_coef(design_params.elevation_of_widening - zB)
    ) * calc_velocity_head(vs)
    Hab_acc = HA - Hb - k1 * calc_velocity_head(v1)
    dv1 = calc_dv_dt(Hab_acc, L1)
    if tau == 0:
        Hbe_acc = 0 * u.meter
    else:
        Hbe_acc = Hb - HE - (k2 + k3 + k_open / tau**2) * calc_velocity_head(v2)
    dv2 = calc_dv_dt(Hbe_acc, L2 + L3)
    Hc = Hb - L2 / (L2 + L3) * Hbe_acc - k2 * calc_velocity_head(v2)
    Hd = HE + L3 / (L2 + L3) * Hbe_acc + (k3 + exit_loss) * calc_velocity_head(v2)
    Pc = Hc - zB
    Pd = Hd - zB
    if t < settings.start_event_time:
        Power_turbine = 0 * u.watt
    else:
        Power_turbine = A * RHO * G * v2 * (Hc - Hd)
    return dv1, dv2, dHs, Hb, Pc, Pd, tau, Power_turbine


############################################
# Code to run the simulation
############################################
@dataclass
class SimulationSettings:
    # Defaults I found to work well

    # Simulation duration
    duration: u.Quantity = 6 * u.minutes

    # When we start the event (opening or closing the valve). Not zero to ensure the system starts at steady state
    start_event_time: u.Quantity = 5 * u.seconds

    # Time step settings
    max_dt: u.Quantity = 0.1 * u.seconds
    min_dt: u.Quantity = 0.001 * u.seconds
    max_delta: u.Quantity = 0.02

    # We don't let tau get too small since this worsens the numerical stability
    tau_cutoff: float = 1 / 200


DEFAULT_SETTINGS = SimulationSettings()

# Object to store the results of the simulation
Snapshot = namedtuple(
    "Snapshot", ["t", "v1", "v2", "Hs", "dv1", "dv2", "Pc", "Pd", "Hb", "Power_turbine"]
)


def simulate(
    scenario: Scenario, design_params=SELECTED_DESIGN, settings=DEFAULT_SETTINGS
):
    """The main simulation loop is here!"""
    t = 0 * u.second
    v1, v2, Hs = scenario.initial_conditions

    results = []

    total_time = settings.duration + settings.start_event_time
    log_every_x_percent = 0.02
    last_log = 0

    while t < total_time:
        dv1, dv2, dHs, Hb, Pc, Pd, tau, Power_turbine = model_system(
            t, v1, v2, Hs, design_params, settings, scenario
        )
        if tau == 0:
            v2 = 0 * u.meter_per_sec
        dt = calc_optimal_time_step((v1, v2, Hs), (dv1, dv2, dHs), settings, t)
        results.append(Snapshot(t, v1, v2, Hs, dv1, dv2, Pc, Pd, Hb, Power_turbine))

        # Update our variables
        t, v1, v2, Hs = (t + dt, v1 + dv1 * dt, v2 + dv2 * dt, Hs + dHs * dt)

        progress = (t / total_time)
        if progress + log_every_x_percent > last_log:
            progress = progress.to_reduced_units()
            progress = min(progress.magnitude, 1)
            print(f"{progress:.1%} done", end="\r")
            last_log = progress
    print("")


    return results


def calc_optimal_time_step(vars, dvars, settings, t):
    """Finds the largest step size between min_dt and max_dt that doesn't result in a variable increasing by more than max_delta %"""
    upper_bound = settings.max_dt
    if t < settings.start_event_time:
        upper_bound = min(upper_bound, settings.start_event_time - t)
    elif t - settings.start_event_time < settings.max_dt:
        return settings.min_dt

    max_rel_change = max([abs(dv) / v for dv, v in zip(dvars, vars) if v != 0])
    if max_rel_change == 0:
        return settings.max_dt
    optimal_dt = settings.max_delta / max_rel_change
    return clamp(settings.min_dt, optimal_dt, upper_bound)


############################################
# Visualization code
############################################
def optimize_operation_time():
    times_to_test = [2, 3, 5, 8, 10]
    plot_power = False  # Useful flag to make screenshots of plots for the report

    ax_c_open = plt.subplot(3, 2, 1)
    ax_d_open = plt.subplot(3, 2, 3, sharex=ax_c_open)

    ax_c_close = plt.subplot(3, 2, 2)
    ax_d_close = plt.subplot(3, 2, 4, sharex=ax_c_close)

    if plot_power:
        ax_power_close = plt.subplot(3, 2, 6, sharex=ax_c_close)
    else:
        ax_power_close = None

    settings = SimulationSettings(duration=12 * u.second)

    for scenario, ax_c, ax_d, ax_power in [
        (LOAD_ACCEPTANCE, ax_c_open, ax_d_open, None),
        (FULL_REJECTION, ax_c_close, ax_d_close, ax_power_close),
    ]:
        for operation_time in times_to_test:
            res_snapshots = simulate(
                scenario,
                DesignParams(valve_operation_time=operation_time * u.second),
                settings,
            )
            t = np.array([s.t.magnitude for s in res_snapshots])
            pressure_d = [s.Pd.magnitude for s in res_snapshots]
            pressure_c = [s.Pc.magnitude for s in res_snapshots]
            ax_c.plot(t, pressure_c, label=f"{operation_time} s")
            ax_d.plot(t, pressure_d, label=f"{operation_time} s")
            if ax_power:
                power = [s.Power_turbine.to("MW").magnitude for s in res_snapshots]
                ax_power.plot(t, power, label=f"{operation_time} s")
                ax_power.set_ylabel("Power absorbed\nby turbine (MW)")
                ax_power.legend(title="Closing time")
                ax_power.set_xlabel("Time (s)")
            else:
                ax_d.set_xlabel("Time (s)")
            ax_d.set_ylabel("Pressure at D (m)")
            ax_c.set_ylabel("Pressure at C (m)")
            ax_c.set_title(scenario.name)
            ax_c.legend(title="Operation time (s)")
    plt.show()


def optimize_tank_diameter():
    tank_diameters = [5, 8, 10, 15]
    required_heights = []
    volume_execavated = []
    max_hbs = []

    for tank_diameter in tank_diameters:
        tank_diameter_m = tank_diameter * u.meters
        design_params = DesignParams(surge_tank_diameter=tank_diameter_m)
        results = simulate(FULL_REJECTION, design_params=design_params)

        required_height = max(s.Hs for s in results) + design_params.freeboard_height
        max_hb = max(s.Hb for s in results)
        v = calc_area(tank_diameter_m) * (
            required_height - design_params.elevation_of_widening
        ) + calc_area(D) * (design_params.elevation_of_widening - zB)

        required_heights.append(required_height.magnitude)
        volume_execavated.append(v.magnitude)
        max_hbs.append(max_hb.magnitude)
    ax = plt.subplot(3, 1, 1)
    ax.set_ylabel(
        f"Required Tank Height (m)\nIncludes {design_params.freeboard_height.magnitude}m of freeboard"
    )
    ax.plot(tank_diameters, required_heights, ".-")
    plt.tick_params("x", labelbottom=False)

    ax = plt.subplot(3, 1, 2, sharex=ax)
    ax.plot(tank_diameters, max_hbs, ".-")
    ax.set_ylabel("Max Head at B (m)")
    plt.tick_params("x", labelbottom=False)

    ax = plt.subplot(3, 1, 3, sharex=ax)
    ax.plot(tank_diameters, volume_execavated, ".-")
    ax.set_ylabel("Rock excavated (m^3)")

    plt.xlabel("Surge Tank Diameter (m)")
    plt.show()


def optimize_widening_height():
    widening_heights = [350, 385, 400, 415]
    volume_execavated = []
    min_hbs = []

    ax = plt.subplot(3, 1, 1)
    ax.set_ylabel("Surge Tank Level (m)")
    ax.set_xlabel("Time (min)")

    for widening_height in widening_heights:
        widening_height_m = widening_height * u.meters
        design_params = DesignParams(elevation_of_widening=widening_height_m)
        results = simulate(LOAD_ACCEPTANCE, design_params=design_params)

        t = np.array([s.t.magnitude for s in results])
        Hs = np.array([s.Hs.magnitude for s in results])
        required_height = max(s.Hs for s in results) + design_params.freeboard_height
        min_hb = min(s.Hb for s in results)
        v = calc_area(design_params.surge_tank_diameter) * (
            required_height - design_params.elevation_of_widening
        ) + calc_area(D) * (design_params.elevation_of_widening - zB)

        volume_execavated.append(v.magnitude)
        min_hbs.append(min_hb.magnitude)
        ax.plot(t, Hs, label=f"{widening_height} m")
    plt.legend()

    ax = plt.subplot(3, 1, 2)
    ax.plot(widening_heights, min_hbs, ".-")
    ax.set_ylabel("Min Head at B (m)")
    plt.tick_params("x", labelbottom=False)

    ax = plt.subplot(3, 1, 3, sharex=ax)
    ax.plot(widening_heights, volume_execavated, ".-")
    ax.set_ylabel("Rock excavated (m^3)")

    plt.xlabel("Elevation of surge tank (m)")
    plt.show()


def visualize_main_scenarios():
    ax_surge = plt.subplot(3, 2, 1)
    ax_pc = plt.subplot(3, 2, 2)
    ax_pb = plt.subplot(3, 2, 3, sharex=ax_surge)
    ax_pd = plt.subplot(3, 2, 4, sharex=ax_pc)
    ax_velocity = plt.subplot(3, 2, 5, sharex=ax_surge)

    for scenario in [FULL_REJECTION, HALF_REJECTION, LOAD_ACCEPTANCE]:
        res_snapshots = simulate(scenario)
        t = np.array([s.t.to("minutes").magnitude for s in res_snapshots])
        t_sec = np.array([s.t.magnitude for s in res_snapshots])
        Hs = np.array([s.Hs.magnitude for s in res_snapshots])
        Pb = np.array([(s.Hb - zB).magnitude for s in res_snapshots])
        Pc = np.array([s.Pc.magnitude for s in res_snapshots])
        Pd = np.array([s.Pd.magnitude for s in res_snapshots])

        ax_surge.plot(t, Hs, label=scenario.name, color=scenario.plotting_color)
        ax_pb.plot(t, Pb, label=scenario.name, color=scenario.plotting_color)
        ax_pc.plot(t_sec, Pc, label=scenario.name, color=scenario.plotting_color)
        ax_pd.plot(t_sec, Pd, label=scenario.name, color=scenario.plotting_color)

        ax_velocity.plot(
            t,
            [s.v1.magnitude for s in res_snapshots],
            label=scenario.name,
            color=scenario.plotting_color,
            linestyle="--",
        )

        ax_velocity.plot(
            t,
            [s.v2.magnitude for s in res_snapshots],
            label=scenario.name,
            color=scenario.plotting_color,
        )
    ax_pb.set_xlabel("Time (min)")
    ax_pd.set_xlabel("Time (sec)")
    ax_surge.set_ylabel("Surge Tank Level (m)")
    ax_pb.set_ylabel("Pressure at B (m)")
    ax_pc.set_ylabel("Pressure at C (m)")
    ax_pd.set_ylabel("Pressure at D (m)")
    ax_pd.set_xlim(0, 30)
    for ax, unit in [
        (ax_surge, "minutes"),
        (ax_pb, "minutes"),
        (ax_pc, "seconds"),
        (ax_pd, "seconds"),
        (ax_velocity, "minutes"),
    ]:
        ax.axvline(
            DEFAULT_SETTINGS.start_event_time.to(unit).magnitude,
            linestyle=":",
            # label="Start",
            color="gray",
        )
        ax.axvline(
            (DEFAULT_SETTINGS.start_event_time + SELECTED_DESIGN.valve_operation_time)
            .to(unit)
            .magnitude,
            linestyle=":",
            # label="End",
            color="gray",
        )
        ax.legend()
    # ax_surge.legend()
    plt.show()


def main():
    visualize_main_scenarios()
    optimize_operation_time()
    optimize_tank_diameter()
    optimize_widening_height()


if __name__ == "__main__":
    main()
