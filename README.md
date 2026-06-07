# Lunar PNT: Cislunar Navigation Simulation

A simulation framework for spacecraft trajectory design and navigation filtering during Earth-to-Lunar transfers, developed for Stanford AA278.

The project covers three mission scenarios (impulsive, optimal impulsive, and low-thrust spiral), a GPS/Galileo pseudorange measurement generator, and a UDU-factorized Extended Kalman Filter (EKF) for onboard state estimation.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Scenarios](#scenarios)
- [Navigation Filter](#navigation-filter)
- [GNSS Measurement Model](#gnss-measurement-model)
- [Orbital Dynamics](#orbital-dynamics)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Overview

This project simulates cislunar navigation from a 500 km LEO parking orbit to lunar arrival, modeling:

- **Trajectory planning** via Lambert's problem, grid search, and continuous low-thrust steering laws
- **High-fidelity propagation** with J2, lunar/solar third-body, and solar radiation pressure (SRP) perturbations
- **GNSS visibility simulation** for GPS L1/L5 and Galileo E1/E5 from beyond GEO, including a realistic link budget model based upon mainlobe/sidelobe and free path loss
- **Navigation filtering** using a UDU-factorized EKF with a 9-state model (position, velocity, clock bias/drift, SRP coefficient)

---

## Project Structure

```
project/
├── misc/
│   ├── constants.py          # Physical and mission constants
│   └── utils.py              # Orbital element conversions, visualization helpers
├── orbital/
│   ├── ephemeris.py          # SPICE kernel interface (Moon/Sun positions)
│   ├── lambert.py            # Lambert's problem solver (universal variables)
│   ├── trajectory_planner.py # Impulsive and continuous-burn plan definitions
│   └── mission_planner.py    # Mission phase sequencer
├── sensors/
│   ├── gnss_constellation.py # GPS/Galileo Walker constellation geometry
│   ├── gnss_measurements.py  # Dual-frequency pseudorange/carrier-phase generator
│   ├── gnss_sensor.py        # Signal band definitions and antenna model
│   └── __init__.py
├── nav/
│   ├── udu_filter.py         # UD-factorized EKF (Thornton MWGS + Bierman update)
│   ├── ekf_dynamics.py       # 9-state dynamics model and Jacobian
│   └── pseudorange_filter.py # Pseudorange measurement interface and main filter loop
├── sim_infra/
│   ├── dynamics.py           # DOP853 ODE propagator
│   ├── event.py              # Event detection (collision, fuel exhaustion)
│   └── satellite.py          # Spacecraft state and properties
├── scenarios/
│   ├── direct_transfer_scenario.py   # Single impulsive TLI burn
│   ├── optimal_transfer_scenario.py  # 28-day grid search
│   └── spiral_transfer_scenario.py   # Continuous low-thrust spiral
└── spice_kernels/
    ├── de440s.bsp            # JPL planetary ephemeris
    └── naif0012.tls          # NAIF leap-second kernel
```

---

## Scenarios

### Direct Transfer

A single impulsive Trans-Lunar Injection (TLI) burn from a 500 km LEO parking orbit. Uses Lambert's problem to solve for the transfer orbit given a fixed departure epoch and a 5.2-day time-of-flight.

- TLI ΔV: ~5.1 km/s
- Duration: ~5.2 days

### Optimal Transfer (Porkchop Search)

A brute-force grid search over a 28-day departure window and 2.5–7 day time-of-flight range. Evaluates ~2,700 Lambert solutions to find the minimum total ΔV trajectory, and produces a porkchop contour plot.

- Optimal ΔV: ~3.1 km/s
- Optimal departure: roughly 26 days into the search window

### Spiral Transfer (Low-Thrust)

Continuous electric propulsion with a prograde + 15° normal steering law to raise the orbit and adjust inclination over ~35 days of thrusting.

- Thruster: 0.1 N, Isp = 4,000 s
- Total mission duration: ~39 days
- Propellant consumed: ~8 kg (from a 50 kg wet mass)

---

## Navigation Filter

The navigation filter is a **UDU-factorized Extended Kalman Filter**, maintaining the covariance as P = U · diag(d) · Uᵀ for numerical stability.

### State Vector (9 elements)

| Index | State | Units |
|-------|-------|-------|
| 0–2 | Position (r) | km |
| 3–5 | Velocity (v) | km/s |
| 6 | Clock bias | km |
| 7 | Clock drift | km/s |
| 8 | SRP coefficient (γ_srp) | m²/kg |

### Algorithm

- **Time update**: Thornton Modified Weighted Gram-Schmidt (MWGS) factorization
- **Measurement update**: Bierman scalar update, one pseudorange at a time
- **Dynamics**: Earth gravity + J2 + lunar/solar third-body + SRP, with full 9×9 Jacobian
- **Impulsive burns**: velocity uncertainty inflation using sigma_dv at maneuver time

## GNSS Measurement Model

### Constellation

| System | Config | Altitude |
|--------|--------|----------|
| GPS | Walker 24/6/2 | 26,559 km |
| Galileo | Walker 24/3/1 | 29,600 km |

### Link Budget

- Nadir-pointing spacecraft antenna: 13 dBic mainlobe (0–21.3°), sidelobes down to −20 dBic
- Free-space path loss computed per satellite at each timestep
- Earth occlusion detection
- Visibility threshold: C/N₀ ≥ 15 dB-Hz

---

## Orbital Dynamics

The propagator integrates a 7-state vector [x, y, z, vx, vy, vz, mass] using SciPy's DOP853 integrator in the J2000 Earth-Centered Inertial (ECI) frame.

**Force model**:

| Perturbations |
|---|
| Earth gravity |
| J2 oblateness |
| Lunar third-body |
| Solar third-body |
| Solar radiation pressure |

Ephemeris positions (Moon and Sun) are looked up from JPL DE440s via SpiceyPy (SPICE kernel).

---
