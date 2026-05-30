# RTD-RAX

**RTD-RAX** is a runtime-assurance extension of Reachability-based Trajectory Design (RTD) that replaces conservative offline reachable sets with fast online safety certification via mixed-monotone reachability ([immrax](https://github.com/gtfactslab/immrax)).

Paper: [arXiv:2603.21635](https://arxiv.org/abs/2603.21635)

Source: [github.com/evannsmc/rtd-rax](https://github.com/evannsmc/rtd-rax)

Documentation: [evannsmc.github.io/ws_RTD](https://evannsmc.github.io/ws_RTD)

## What's Included

This image ships a complete environment for running all RTD-RAX case studies:

- **Base:** `osrf/ros:jazzy-desktop-full`
- **Python (venv):** NumPy, SciPy, Matplotlib, Shapely, JAX (CPU), immrax
- **System:** libcdd-dev, libgmp-dev

## Quick Start

```bash
git clone https://github.com/evannsmc/rtd-rax.git && cd rtd-rax/docker
make pull             # pulls this image and tags it locally
make run_gui          # starts container with X11 forwarding for plots
```

For GUI plotting on Linux/X11, allow Docker access first:

```bash
xhost +local:docker
```

## Run Examples

```bash
make rtd-gap                           # standard FRS — too conservative, no path found
make rtd-gap FRS=noerror               # noerror FRS — feasible path found
make rtd-gap-verify FRS=noerror        # + immrax verification (safe)
make rtd-case2-suite                   # Case 2 representative + two-repair outputs
make rtd-disturbance-compare           # disturbance course comparison
```

See the [full target and parameter reference](https://github.com/evannsmc/rtd-rax/blob/main/docker/README.md) for all available experiments and manuscript figure generation targets.

## License

MIT
