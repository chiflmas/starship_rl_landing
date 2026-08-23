# Starship V1/V1.1: terminal landing with reinforcement learning

An educational 2D environment in which a **Soft Actor-Critic (SAC)** agent
learns the terminal portion of a Starship-inspired landing: belly flop, flip,
braking manoeuvre, touchdown, engine shutdown, and post-contact stability.

> [!IMPORTANT]
> This is not a high-fidelity flight simulator and does not reproduce SpaceX
> flight software. Vehicle-inspired orders of magnitude provide physical
> context for a reinforcement-learning experiment that can be expanded in
> controlled increments.

V1 deliberately uses one centred virtual actuator. Later versions can add
independent engines, propellant, ignition constraints, failures, weather,
sensors, and other sources of uncertainty without hiding the learning problem
behind all of that complexity from the beginning.

## Acknowledgement

This repository is based on
**[Rocket-recycling with Reinforcement Learning](https://github.com/jiupinjia/rocket-recycling)**
by **[Zhengxia Zou, Ph.D.](https://zhengxiazou.github.io/)**. Many thanks to
Zhengxia Zou for publishing the original environment and providing an
excellent educational starting point.

This project adapts that foundation to a staged Starship terminal-landing
experiment. The original project and this derivative are not affiliated with
SpaceX.

## V1 at a glance

| Property | V1 value |
| --- | ---: |
| Simulation | 2D rigid body |
| Vehicle mass | 140,000 kg |
| Vehicle height | 50 m |
| Virtual actuator | One centred, up to 6 MN |
| Physics timestep | 0.05 s (20 Hz) |
| Policy observations | 12 in V1; 10 in V1.1 |
| Continuous actions | 2 |
| Episode limit | 750 steps (37.5 s) |
| Learning algorithm | Soft Actor-Critic |

The 6 MN actuator represents simplified aggregate control authority. It is not
the specification of an individual Raptor engine.

## Environment

### Physics

`rocket.py` models:

- gravity and rigid-body translational and rotational inertia;
- orientation-dependent aerodynamic drag;
- aerodynamic angular damping;
- a restoring aerodynamic moment during the high-altitude belly flop;
- gimballed thrust and its torque around the centre of mass;
- ground contact, impact limits, settling, and tip-over detection.

The restoring aerodynamic moment is fully active above 450 m AGL and fades
between 450 m and 350 m AGL. The reward and observation attitude reference
transitions from an 80-degree belly flop at 450 m to vertical at 250 m. This
gives the policy control authority during the flip while retaining a simplified
aerodynamic belly-flop phase.

V1 does not include six-degree-of-freedom motion, fuel consumption, changing
mass, flaps, RCS, wind, weather, sensor noise, or individual engine dynamics.

### Observation vector

Only V1 uses the compact 12-value observation returned by `_flatten_v1()`:

| Index | Observation | Internal scaling |
| ---: | --- | --- |
| 0 | Horizontal position `x` | `x / 200` |
| 1 | Altitude above ground level | `AGL / 600` |
| 2 | Horizontal velocity `vx` | `vx / 50` |
| 3 | Vertical velocity `vy` | `vy / 100` |
| 4 | `sin(theta)` | Already bounded |
| 5 | `cos(theta)` | Already bounded |
| 6 | Angular velocity | Clipped to `[-2, 2]` |
| 7 | Current throttle | `[0, 1]` |
| 8 | Current gimbal | Divided by 30 degrees |
| 9 | Phase-dependent attitude error | Divided by `pi` |
| 10 | Signed braking margin | Divided by 100 |
| 11 | Remaining episode fraction | `[0, 1]` |

`VecNormalize` then normalises these observations during training. A checkpoint
must therefore be evaluated or resumed with its associated
`vec_normalize.pkl` file.

Attitude error and braking margin are engineered observations. V1 is not
claiming that guidance emerged from entirely raw sensor measurements.

### V1.1 reduced-engineering variant

V1.1 removes the two engineered guidance inputs from the policy:

- phase-dependent attitude error;
- signed braking margin.

Its 10-value observation contains `x`, AGL, `vx`, `vy`, `sin(theta)`,
`cos(theta)`, angular rate, current throttle, current gimbal, and remaining
episode time. Physics, actions, curriculum phases, success conditions, and SAC
hyperparameters remain identical to V1. V1.1 uses its own reward function,
described below, so it does not depend on the two removed guidance variables.

V1 and V1.1 checkpoints and `VecNormalize` files are not interchangeable
because their observation dimensions differ.

### Actions

The actor emits two continuous commands:

| Action | Policy range | Physical interpretation |
| --- | ---: | --- |
| Throttle | `[0, 1]` | Cubed before it reaches the actuator, providing finer low-thrust control |
| Gimbal | `[-1, 1]` | Scaled to `+/-30` degrees |

Gimbal movement is rate-limited to one degree per physics step. With a 20 Hz
simulation, the maximum commanded gimbal rate is therefore 20 degrees/s.

## Curriculum learning

The complete manoeuvre was too difficult to explore reliably from random SAC
actions. The curriculum changes only the initial-state distribution across
three phases and is shared by V1 and V1.1.

Initial states are mirrored around `x = 0`. Horizontal velocity points towards
the pad and the attitude/angular-rate signs follow the side from which the
vehicle approaches.

| Phase | AGL | Absolute x | Inward vx | vy | Absolute tilt | Angular rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase_1` | 80-150 m | 20-25 m | 3 m/s | -25 to -12 m/s | 3-8 deg, plus +/-2 deg noise | 0 deg/s |
| `phase_2` | 180-300 m | 30-50 m | 3-5 m/s | -40 to -25 m/s | 15-30 deg | 0-4 deg/s |
| `phase_3` | 500-600 m | 150-200 m | 9-15 m/s | -95 to -85 m/s | 75-85 deg, plus +/-2 deg noise | 0 deg/s |

`phase_1` is the default. The curriculum phase is selected explicitly with
`--phase`; promotion is currently a manual training decision rather than an
automatic callback.

## Reward design

`improved_rewards_v1.py` uses progress-based shaping. Positive dense rewards
are changes in error, so the policy cannot farm reward merely by remaining in
a centred hover.

Positive progress terms measure reductions in:

- horizontal pad error;
- phase-dependent attitude error;
- lateral speed;
- braking-safety deficit.

Non-positive terms apply:

- a constant time cost;
- an ascent penalty;
- angular-rate and attitude penalties;
- a braking-viability penalty when safe stopping distance is being exhausted;
- stronger attitude, spin, and outside-pad penalties near the ground.

V1 deliberately does **not** reward losing altitude, prescribe a vertical
speed profile, impose a fixed burn altitude, or penalise thrust consumption.
There is no fuel model yet. The agent remains free to discover a late or
conservative braking strategy.

Terminal rewards dominate the shaping terms:

| Outcome | Reward |
| --- | ---: |
| Successful landing | `+300` |
| Crash | `-250` |
| Timeout | `-200` |

`improved_rewards_v1_1.py` is independent from the V1 reward system. It removes
the phase-dependent attitude target and the model-based braking envelope. Its
positive terms reward progress towards the pad, lower lateral speed, becoming
upright during the terminal approach, and reducing excessive downward speed
near touchdown. Its penalties use only raw V1.1 quantities: ascent, angular
rate, uprightness, vertical and lateral speed, pad position, and time.

V1.1 does not prescribe intermediate flip angles or estimate stopping
distance. Upright guidance begins below 250 m AGL, while speed and pad-control
penalties become active below 100 m AGL. Terminal rewards remain `+300`,
`-250`, and `-200`, matching V1.

## Touchdown and success conditions

Crossing the ground boundary is not sufficient for success.

### Impact limits

At first contact, the vehicle crashes if any hard limit is violated:

- total impact speed is at least 7 m/s;
- vertical impact speed is at least 6 m/s;
- `|x|` is at least 15 m;
- absolute tilt is at least 10 degrees;
- absolute angular rate is at least 5 degrees/s.

### Stable landing

After valid contact, all of the following must hold:

- the vehicle remains on the ground;
- total speed is below 5 m/s;
- `|x|` is below 15 m;
- absolute tilt is below 5 degrees;
- absolute angular rate is below 3 degrees/s;
- total thrust is below 1% of maximum thrust.

The conditions must hold for 20 consecutive steps, or one second. The episode
then continues for one additional second so the green success state is visible
in recorded video. A touchdown that cannot begin stable settling within two
seconds is classified as a crash.

## SAC configuration

| Parameter | Value |
| --- | ---: |
| Actor network | `[256, 256]` |
| Two critic networks | `[256, 256, 256]` each |
| Learning rate | `3e-4` |
| Replay buffer | 400,000 transitions |
| Learning starts | 20,000 transitions |
| Batch size | 512 |
| Discount factor `gamma` | 0.998 |
| Target smoothing `tau` | 0.005 |
| Gradient steps | 3 per vector step |
| Entropy coefficient | `auto_0.1` |
| Parallel environments | 6 |
| Default seed | 42 |

The simulation is mostly CPU-bound. `SubprocVecEnv` runs the six training
environments in separate processes, while evaluation and video rendering use a
single `DummyVecEnv`. A CUDA-enabled PyTorch build can accelerate network
updates, but it does not accelerate the Python physics processes.

Reward normalisation is disabled. Observation normalisation remains enabled.

## Installation

Python 3.12 is the recommended environment for the current dependency stack.
In PowerShell, from the repository directory:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Requirements.txt
```

On Linux or macOS:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Requirements.txt
```

The default PyPI PyTorch package may be CPU-only depending on the platform and
installer. If CUDA is required, install the wheel recommended for the local
Python version, GPU, and NVIDIA driver using the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Training

Start V1 Phase 1 from scratch:

```powershell
python main.py --version v1 --phase phase_1 --seed 42
```

Start V1.1 with the same curriculum conditions:

```powershell
python main.py --version v1_1 --phase phase_1 --seed 42
```

Fresh training currently runs for four million environment timesteps with six
parallel environments. Models, normalisation statistics, TensorBoard events,
and periodic videos are written under `models/`, `logs/`, and `videos/`.

Start TensorBoard:

```powershell
tensorboard --logdir .\logs --port 6006
```

Then open `http://localhost:6006`.

Useful custom metrics include:

- `rollout/landing_success_rate_50`;
- episode reward and length;
- actor and critic losses;
- entropy coefficient.

## Moving to the next curriculum phase

Resume the Phase 1 policy in Phase 2:

```powershell
python main.py `
  --version v1 `
  --resume .\models\v1_single_engine\final_model.zip `
  --vecnorm .\models\v1_single_engine\vec_normalize.pkl `
  --phase phase_2 `
  --timesteps 300000 `
  --n-envs 6 `
  --learning-rate 1e-4 `
  --seed 42
```

Then use the resulting checkpoint and normaliser to resume with
`--phase phase_3`.

`SAC.save()` does not include the replay buffer. Resume mode preserves the
model weights, optimiser state, timestep counter, and `VecNormalize`
statistics, but creates a fresh replay buffer and collects 50,000 new
transitions before updating the network. Preserve or rename the outputs from
each phase before launching the next one because resumed runs share the
`models/v1_single_engine_resumed/` directory.

## Deterministic evaluation

Evaluate a model against a selected phase without training:

```powershell
python main.py `
  --version v1 `
  --evaluate .\models\v1_single_engine\final_model.zip `
  --vecnorm .\models\v1_single_engine\vec_normalize.pkl `
  --phase phase_1 `
  --episodes 10 `
  --seed 42
```

The policy runs deterministically. The first episode uses the base seed and
each subsequent episode increments it by one. Repeating the same command
therefore reproduces the same evaluation scenarios and trajectories. Change
the base seed or increase `--episodes` to test a different sample.

Evaluation exports one MP4 per episode under `videos/evaluation/` and prints
success, crash, timeout, reward, final altitude, and velocity summaries.

## Curriculum evaluation chart

Generate the Plotly comparison of V1 Phase 1-3 evaluation reward and episode
length from the existing TensorBoard event files:

```powershell
python plot_v1_curriculum_evaluations.py
```

The command writes a LinkedIn-ready PNG, an interactive self-contained HTML
chart, and the extracted evaluation values as CSV under `gallery/`. The plot
uses the original global timestep values and does not smooth or interpolate
the evaluation checkpoints.

## Neural activation visualisation

Add `--activation-visualization` to evaluation:

```powershell
python main.py `
  --version v1 `
  --evaluate .\models\v1_single_engine\final_model.zip `
  --vecnorm .\models\v1_single_engine\vec_normalize.pkl `
  --phase phase_1 `
  --episodes 10 `
  --seed 42 `
  --activation-visualization
```

For every episode this additionally creates:

- `episode_NN_network.gif`: sampled actor activations;
- `episode_NN_combined.mp4`: synchronised simulation and network views.

The visualiser shows a representative subset of hidden neurons; it does not
change the policy or the inference result. Input labels are selected
automatically for the 12-input V1 actor or the 10-input V1.1 actor.

Combine all activation videos and optionally accelerate them:

```powershell
python combine_evaluation_episodes.py `
  --speed 2 `
  --gif-fps 10 `
  --gif-width 1000 `
  --overwrite
```

Use `--preserve-frames` only when every source frame must be retained. It
raises the output frame rate and can produce very large GIF files.

To combine the plain simulation videos instead, change the pattern:

```powershell
python combine_evaluation_episodes.py `
  --pattern "episode_??.mp4" `
  --output-prefix .\videos\evaluation\simulation_episodes `
  --speed 2 `
  --overwrite
```

## Repository structure

```text
main.py                         Gymnasium wrapper, SAC setup, callbacks, and CLI
rocket.py                       Dynamics, contact, success logic, and rendering
improved_rewards_v1.py          V1 progress-based reward system
improved_rewards_v1_1.py        Isolated V1.1 reward class
curriculum_phases.py            Initial-state distributions for Phases 1-3
activation_visualizer.py        Actor activation GIF and combined-video export
combine_evaluation_episodes.py  Concatenate and accelerate evaluation media
plot_v1_curriculum_evaluations.py  Plot Phase 1-3 TensorBoard evaluations
utils.py                        Recovered project utilities
Requirements.txt                Python dependencies
```

Training outputs, checkpoints, TensorBoard logs, virtual environments, and
videos are ignored by Git.

## Roadmap

1. **V1:** centred virtual actuator, terminal belly flop, curriculum learning,
   and stable touchdown.
2. **V2:** three engines, thrust allocation, per-engine gimbal, and ignition
   constraints.
3. **V3:** fuel, variable mass, failures, and propellant-margin management.
4. **V4:** flaps, wind, weather, uncertainty, and domain randomisation.
5. **V5:** a higher-altitude glide and transition into the terminal manoeuvre.
