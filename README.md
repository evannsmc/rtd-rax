# RTD-RAX

![Status](https://img.shields.io/badge/Status-Active-blue)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![evannsmc.com](https://img.shields.io/badge/evannsmc.com-Project%20Page-blue)](https://www.evannsmc.com)

**RTD-RAX** is a Python implementation of Reachability-based Trajectory Design (RTD) augmented with formal reachability verification via [immrax](https://github.com/gtfactslab/immrax).

Standard RTD computes parameterized Forward Reachable Sets (FRS) offline and uses them online to solve for collision-free trajectory parameters in real time. However, because high-fidelity system models are too expensive to compute reachable sets for, RTD relies on simplified models and must inflate the FRS to account for the resulting tracking error. This introduces conservatism — in many scenarios, RTD fails to find a safe trajectory even when one exists. Ignoring the tracking error (the `noerror` variant) gives the planner access to more aggressive trajectories but sacrifices safety guarantees. Moreover, offline FRS computations cannot account for runtime disturbances (wind, ice, slippage), so traditional RTD's safety guarantees are only as good as the assumptions made offline.

RTD-RAX closes both gaps with a three-stage online loop:

1. **Plan** with the less-conservative `noerror` FRS to find candidate trajectories.
2. **Verify** each candidate using immrax interval-arithmetic reachability under the actual uncertainty and disturbance bounds.
3. **Repair** any candidate flagged as unsafe via speed-backoff and CEGIS-style obstacle buffer tightening, then re-verify before execution.

This produces trajectories that are both less conservative *and* formally verified safe under measured runtime conditions.

## Key Results

- Safe trajectory planning under a priori unknown disturbances
- Reduction in conservatism compared to standard RTD, enabling navigation through narrow corridors and around angled obstacles
- Elimination of needless fail-safe maneuvers present in standard RTD
- Real-time verification and repair under measured disturbance bounds
- Demonstrated on a 4D unicycle model with three case studies

## Quick Start

All case studies run inside a Docker container with pre-installed dependencies (JAX, immrax, matplotlib, etc.).

```bash
cd docker/
make build
make run_gui          # starts container with X11 forwarding for plots
make rtd-gap          # standard FRS — too conservative, no path found
make rtd-gap FRS=noerror   # noerror FRS — feasible path found
```

For GUI plotting on Linux/X11, allow Docker access first:

```bash
xhost +local:docker
```

After finishing, revoke access:

```bash
xhost -local:docker
```

## Case Studies

All `make` targets are run from the `docker/` directory.

### Study 1: Gap Scenario

Two rectangular obstacles leave a narrow 0.619 m gap. Standard RTD is too conservative to pass; RTD-RAX uses the noerror FRS plus immrax verification to certify a safe path through.

```bash
make manuscript-case1-gap-suite
```

![Gap Scenario](assets/gap_compare.gif)

### Study 2: Angled Obstacles with Repair

Angled obstacles require complex trajectories. The noerror FRS proposes candidates that may be unsafe under uncertainty — immrax catches the collision risk, and the hybrid repair loop finds safer alternatives.

```bash
make rtd-case2-suite
make manuscript-case2-suite
```

<!-- ![Angled Obstacle Comparison](assets/angled_animate.gif) -->
![Angled Obstacle Comparison](assets/angled_repair_view_two_repairs.gif)

### Study 3: Disturbance Compare

A randomized multi-gap course with disturbance patches. Standard RTD collides on cycle 3; RTD-RAX detects the risk via immrax, repairs, and reaches the goal safely.

| Planner | Outcome | Cycles | Repairs | Mean / p95 Compute |
|---|---|---|---|---|
| Standard RTD | Collision (cycle 3) | 3 | -- | 10.5 ms / 21.9 ms |
| RTD-RAX | Goal reached | 19 | 3 | 10.5 ms / 37.4 ms |

```bash
make rtd-disturbance-compare
make manuscript-disturbance-gallery
```

![Disturbance Comparison](assets/disturbance_compare.gif)

## Repository Structure

```
rtd_rax/
├── turtlebot_rtd_numpy/          # Core RTD-RAX implementation (NumPy/SciPy)
│   ├── one_shot_rtd.py           # Basic single-shot planner
│   ├── one_shot_rtd_gap.py       # Gap scenario with immrax verification
│   ├── rtd_gap_journey.py        # Multi-step receding-horizon replanning
│   ├── rtd_gap_journey_compare.py        # Standard vs noerror comparison with repair
│   ├── rtd_angled_obstacle_compare.py    # Angled obstacle scenario with repair
│   ├── rtd_random_disturbance_compare.py # Disturbance course comparison
│   ├── immrax_verify.py          # immrax reachability verification
│   ├── disturbance_case_study_utils.py   # Shared utilities for case studies
│   ├── frs_loader.py             # FRS .mat file loader
│   ├── constraints.py            # FRS polynomial constraint builder
│   ├── cost.py                   # Trajectory cost function
│   ├── polynomial_utils.py       # Polynomial evaluation utilities
│   ├── geometry_utils.py         # Obstacle geometry helpers
│   ├── trajectory.py             # Trajectory generation
│   └── turtlebot_agent.py        # Turtlebot agent simulation
├── python_preprocessed_frs/      # Precomputed FRS data files (.mat)
├── docker/                       # Docker setup and make targets
│   ├── Dockerfile
│   ├── makefile                  # All experiment and figure targets
│   ├── requirements.txt
│   └── README.md                 # Detailed Docker reference
└── README.md
```

## Make Targets

### Core Experiments

| Target | Description |
|---|---|
| `rtd-gap` | Gap scenario (`FRS=standard` or `FRS=noerror`) |
| `rtd-gap-verify` | Gap scenario with immrax verification |
| `rtd-journey-gap-compare` | Journey comparison with verification and repair |
| `rtd-case2-suite` | Case 2 representative + two-repair outputs |
| `rtd-disturbance-compare` | Disturbance course comparison |

### Manuscript Figures

| Target | Description |
|---|---|
| `manuscript-case1-gap-suite` | Case 1 GIF/PDF family |
| `manuscript-case2-suite` | Case 2 GIF/PDF family |
| `manuscript-disturbance-gallery` | Disturbance comparison outputs |
| `manuscript-figures` | Regenerate all manuscript assets |

### Common Parameters

| Variable | Default | Description |
|---|---|---|
| `FRS` | `standard` | FRS variant (`standard` or `noerror`) |
| `UNCERTAINTY` | `0.01` | immrax positional uncertainty (m) |
| `DISTURBANCE` | `0.0` | Bounded additive disturbance |
| `JOURNEY_VERIFY` | `0` | Enable immrax verification (`1` to enable) |
| `JOURNEY_REPAIR` | `0` | Enable hybrid repair (`1` to enable) |

```bash
# Example: gap verification with custom uncertainty
make rtd-gap-verify FRS=noerror UNCERTAINTY=0.05 DISTURBANCE=0.01
```

See [`docker/README.md`](docker/README.md) for the full target and parameter reference.

## Container Lifecycle

| Command | Description |
|---|---|
| `make build` | Build the Docker image |
| `make run` | Start container (headless) |
| `make run_gui` | Start container with X11 forwarding |
| `make start` | Restart a stopped container |
| `make stop` | Stop the running container |
| `make attach` | Attach a shell to the running container |

## Dependencies

Installed automatically by Docker:

- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/), [Shapely](https://shapely.readthedocs.io/)
- [JAX](https://jax.readthedocs.io/) (CPU)
- [immrax](https://github.com/gtfactslab/immrax) — interval-arithmetic reachability library

## License

MIT

## Website

This project is part of the [evannsmc open-source portfolio](https://www.evannsmc.com/projects).
