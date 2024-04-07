from dataclasses import dataclass
from collections import namedtuple
import matplotlib.pyplot as plt
import math
import numpy as np

from pint import UnitRegistry


def calc_area(diameter):
    return math.pi * diameter**2 / 4


def calc_circumference(diameter):
    return math.pi * diameter


u = UnitRegistry()
u.meter_per_sec = u.meter / u.second
u.meter_per_sec_2 = u.meter / u.second**2

# Givens
G = 9.81 * u.meter_per_sec_2  # Gravitational constant
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

# Assumptions
exit_loss = 0.5  # Assuming there is a diffuser
speed_of_wave_in_conduit = 400 * u.meter_per_sec  # Speed of wave in conduit

# Calculated paramters
A = calc_area(D)  # Cross-sectional area
k1 = f * L1 / D  # Velocity heads lost in conduit 1
k2 = f * L2 / D  # Velocity heads lost in conduit 2
k3 = f * L3 / D  # Velocity heads lost in conduit 3
L = L1 + L2 + L3  # Total length of conduits
H_sys = HA - HE  # Total head in system

# Steady state (ss) conditions
ss_H_valve = H_sys * efficiency
ss_H_loss = H_sys - ss_H_valve
ss_velocity_heads_lost = exit_loss + k1 + k2 + k3
ss_velocity_head = ss_H_loss / ss_velocity_heads_lost
ss_v = (2 * G * ss_velocity_head) ** 0.5
ss_Q = A * ss_v
ss_HS = HA - ss_velocity_head * k1
k_open = ss_H_valve / ss_velocity_head

# Check results match hand calculations
assert round(ss_v, 2) == 4.43 * u.meter / u.second
assert ss_HS == 385 * u.meter


# Define scenarios
@dataclass(frozen=True)
class Scenario:
    name: str
    start_tau: int
    end_tau: int
    initial_conditions: tuple = (ss_v, ss_v, ss_HS)


# Define different scenarios
FULL_REJECTION = Scenario("Full load rejection", start_tau=1, end_tau=0)
HALF_REJECTION = Scenario("Half load rejection", start_tau=1, end_tau=0.5)
LOAD_ACCEPTANCE = Scenario(
    "Full load acceptance",
    start_tau=0,
    end_tau=1,
    initial_conditions=(0 * u.meter_per_sec, 0 * u.meter_per_sec, HA),
)


# Define rigid water column model
def calc_velocity_head(velocity):
    return velocity * abs(velocity) / (2 * G)


def calc_vs(v1, v2):
    return v1 - v2


def calc_dHs(vs, Hs, design_params):
    tank_diameter = D if Hs < design_params.height_of_widening else design_params.D_ST
    return vs * (D / tank_diameter) ** 2


def calc_Hb(Hs, vs):
    return Hs + minor_loss_coef * calc_velocity_head(vs)


def calc_Hab_acc(Hb, v1):
    max_acceleration = speed_of_wave_in_conduit / G * max(abs(v1), 5 * u.meter_per_sec)
    computed_acceleration = HA - Hb - k1 * calc_velocity_head(v1)
    if abs(computed_acceleration) > max_acceleration:
        return max_acceleration * (computed_acceleration / abs(computed_acceleration))
    return computed_acceleration


def calc_Hbe_acc(Hb, v2, tau):
    if tau == 0:
        assert v2 != 0, "We expected us to be stopping"
        return -speed_of_wave_in_conduit / G * v2
    max_acceleration = speed_of_wave_in_conduit / G * max(abs(v2), 5 * u.meter_per_sec)
    computed_acceleration = (
        Hb - HE - (k2 + k3 + k_open / tau**2) * calc_velocity_head(v2)
    )
    if abs(computed_acceleration) > max_acceleration:
        return max_acceleration * (computed_acceleration / abs(computed_acceleration))
    return computed_acceleration


def calc_dv(acc_head, length):
    return G / length * acc_head


def calc_pressure_d(v2, dv2):
    return L3 / G * dv2 + (k3 + exit_loss) * calc_velocity_head(v2) + HE - zB


def calc_pressure_c(v2, dv2, Hb):
    return Hb - L2 / G * dv2 + k2 * calc_velocity_head(v2) - zB


@u.check(u.meter_per_sec, u.meter_per_sec, u.meter, None, None)
def compute_derivates(v1, v2, Hs, tau, design_params):
    """Computes for a given snapshot of the system the rate of change of each variable (v1, v2, HS)"""
    vs = calc_vs(v1, v2)
    dHs = calc_dHs(vs, Hs, design_params)
    Hb = calc_Hb(Hs, vs)
    Hab_acc = calc_Hab_acc(Hb, v1)
    dv1 = calc_dv(Hab_acc, L1)
    if tau == 0 and v2 == 0:
        dv2 = 0 * u.meter_per_sec_2
    else:
        Hbe_acc = calc_Hbe_acc(Hb, v2, tau)
        dv2 = calc_dv(Hbe_acc, L2 + L3)
    Pc = calc_pressure_c(v2, dv2, Hb)
    Pd = calc_pressure_d(v2, dv2)
    return dv1, dv2, dHs, Hb, vs, Pc, Pd


def calc_tau(scenario, t, design_params, start_event_time):
    """Uses linear interpolation to obtain tau for the given time and scenario information."""
    if t <= start_event_time:
        return scenario.start_tau
    elif t >= start_event_time + design_params.event_duration:
        return scenario.end_tau
    else:
        # Linearly interpolate between start and end tau
        return (
            scenario.start_tau
            + (scenario.end_tau - scenario.start_tau)
            * (t - start_event_time)
            / design_params.event_duration
        )


@dataclass
class SimulationSettings:
    duration: u.Quantity = 10 * u.minutes
    max_dt: u.Quantity = 0.1 * u.seconds
    min_dt: u.Quantity = 0.01 * u.seconds
    max_delta: u.Quantity = 0.01


@dataclass
class DesignParams:
    D_ST: u.Quantity = 5 * u.meters
    event_duration: u.Quantity = 3 * u.second
    height_of_widening: u.Quantity = 385 * u.meter


# The time dependent variables (velocity in conduit AB, velocity in conduit BE, level of surge tank)
Snapshot = namedtuple("Snapshot", ["t", "v1", "v2", "Hs", "dv1", "dv2", "Pc", "Pd"])


def simulate_system(
    scenario: Scenario,
    design_params=DesignParams(),
    settings=SimulationSettings(),
):
    start_event_time = settings.duration * 0.1

    t = 0 * u.second
    v1, v2, Hs = scenario.initial_conditions

    results = []

    while t < settings.duration + start_event_time:
        tau = calc_tau(scenario, t, design_params, start_event_time)
        # if tau == 0:
        #     v2 = 0 * u.meter_per_sec
        dv1, dv2, dHs, Hb, vs, Pc, Pd = compute_derivates(
            v1, v2, Hs, tau, design_params
        )
        dt = compute_optimal_step_size((v1, v2, Hs), (dv1, dv2, dHs), settings)
        results.append(Snapshot(t, v1, v2, Hs, dv1, dv2, Pc, Pd))

        # Update our variables
        t, v1, v2, Hs = (t + dt, v1 + dv1 * dt, v2 + dv2 * dt, Hs + dHs * dt)

    return results


def compute_optimal_step_size(vars, dvars, settings):
    """Finds the largest step size between min_dt and max_dt that doesn't result in a variable increasing by more than max_delta %"""
    max_rel_change = max([abs(dv) / v for dv, v in zip(dvars, vars) if v != 0])
    if max_rel_change == 0:
        return settings.max_dt
    optimal_dt = settings.max_delta / max_rel_change
    return max(settings.min_dt, min(settings.max_dt, optimal_dt))


def plot_results(res_snapshots, label=None):
    t = np.array([s.t.to("minutes").magnitude for s in res_snapshots])
    Hs = np.array([s.Hs.magnitude for s in res_snapshots])

    plt.plot(t, Hs, label=label)
    plt.xlabel("Time (min)")
    plt.ylabel("Surge Tank Level (m)")


def optimize_open_time(opening_times=[2, 3]):
    # ax_s = plt.subplot(3, 1, 1)
    ax_c = plt.subplot(2, 1, 1)
    ax_d = plt.subplot(2, 1, 2, sharex=ax_c)

    settings = SimulationSettings(duration=12 * u.second, max_dt=0.005 * u.second)

    for opening_time in opening_times:
        res_snapshots = simulate_system(
            LOAD_ACCEPTANCE,
            DesignParams(event_duration=opening_time * u.second),
            settings,
        )
        t = np.array([s.t.magnitude for s in res_snapshots])
        hs = [s.Hs.magnitude for s in res_snapshots]
        pressure_d = [s.Pd.magnitude for s in res_snapshots]
        pressure_c = [s.Pc.magnitude for s in res_snapshots]
        # ax_s.plot(t, hs, label=f"{closing_time} s", marker=".")
        ax_d.plot(t, pressure_d, label=f"{opening_time} s", marker=".")
        ax_c.plot(t, pressure_c, label=f"{opening_time} s", marker=".")

    plt.xlabel("Time (s)")
    ax_d.set_ylabel("Piezometric head at D (m)")
    ax_c.set_ylabel("Piezometric head at C (m)")
    plt.legend(title="Opening time (s)")
    plt.show()


def optimize_closing_time(closing_times=[2, 3, 5, 10]):
    # ax_s = plt.subplot(3, 1, 1)
    ax_c = plt.subplot(2, 1, 1)
    ax_d = plt.subplot(2, 1, 2, sharex=ax_c)

    for closing_time in closing_times:
        res_snapshots = simulate_system(
            FULL_REJECTION,
            DesignParams(event_duration=closing_time * u.second),
            settings=SimulationSettings(12 * u.second),
        )
        t = np.array([s.t.magnitude for s in res_snapshots])
        hs = [s.Hs.magnitude for s in res_snapshots]
        pressure_d = [s.Pd.magnitude for s in res_snapshots]
        pressure_c = [s.Pc.magnitude for s in res_snapshots]
        # ax_s.plot(t, hs, label=f"{closing_time} s", marker=".")
        ax_d.plot(t, pressure_d, label=f"{closing_time} s", marker=".")
        ax_c.plot(t, pressure_c, label=f"{closing_time} s", marker=".")

    plt.xlabel("Time (s)")
    ax_d.set_ylabel("Piezometric head at D (m)")
    ax_c.set_ylabel("Piezometric head at C (m)")
    plt.legend(title="Closing time (s)")
    plt.show()


def optimize_tank_diameter(
    tank_diameters=[3, 5, 8, 10, 15, 20], safety_margin=3 * u.meters
):
    max_heights = []
    surge_tank_surface_areas = []
    volume_execavated = []
    for tank_diameter in tank_diameters:
        res_snapshots = simulate_system(
            FULL_REJECTION,
            DesignParams(tank_diameter * u.meters),
        )
        max_height = max(s.Hs for s in res_snapshots) + safety_margin
        extra_height = max_height - ss_HS
        surge_tank_surface_areas.append(
            (extra_height * calc_circumference(tank_diameter)).magnitude
        )
        max_heights.append(max_height.magnitude)
        volume_execavated.append((extra_height * calc_area(tank_diameter)).magnitude)
    ax = plt.subplot(2, 1, 1)
    ax.set_ylabel("Peak Surge Tank Level (m)")
    ax.plot(tank_diameters, max_heights, ".-")
    plt.tick_params("x", labelbottom=False)

    ax = plt.subplot(2, 1, 2, sharex=ax)
    ax.plot(tank_diameters, surge_tank_surface_areas, ".-")
    ax.set_ylabel("Surge Tank Wall Surface Area (m^2)")
    # plt.tick_params("x", labelbottom=False)

    # ax = plt.subplot(3, 1, 3, sharex=ax)
    # ax.plot(tank_diameters, volume_execavated, "o-")
    # ax.set_ylabel("Volume Excavated (m^3)")

    plt.xlabel("Surge Tank Diameter (m)")
    plt.show()


def visualize_main_scenarios():
    for scenario in [FULL_REJECTION, HALF_REJECTION, LOAD_ACCEPTANCE]:
        res_snapshots = simulate_system(scenario)
        plot_results(res_snapshots, label=scenario.name)
    plt.legend()
    plt.show()


def main():
    optimize_open_time()
    optimize_closing_time()
    optimize_tank_diameter()
    visualize_main_scenarios()


if __name__ == "__main__":
    main()
