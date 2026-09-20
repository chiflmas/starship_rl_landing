# Starship V1/V1.1/V2: terminal landing with reinforcement learning

## Checkpoint saving (current workflow)

New training and `--resume` create a unique UTC session directory under the
model output folder. Existing checkpoints are preserved. Within each session:
TensorBoard events and `evaluations.npz` are also isolated under
`logs/<version-and-phase>/<session-id>/`, using the same session ID as the
checkpoints. Resume explicitly redirects TensorBoard to the new session rather
than retaining the directory stored in the source model. Fresh V2 training uses
`ent_coef="auto_0.01"` and `target_entropy=-9`; resume retains learned entropy.

| Snapshot | When | Contents |
| --- | --- | --- |
| `best_<steps>_<id>/` | Highest evaluation success rate; mean reward breaks ties | `model.zip`, `vec_normalize.pkl`, `metadata.json` |
| `periodic_<steps>_<id>/` | Every 100,000 additional transitions, after rollout collection | Same files plus `replay_buffer.pkl` |
| `final_<steps>_<id>/` | Normal completion | Same files plus replay buffer |
| `interrupted_<steps>_<id>/` | Ctrl+C | Same files plus replay buffer |

`best.json`, `periodic.json`, `final.json` and `interrupted.json` point to the
latest complete bundle of that kind. Best snapshots are retained, not replaced
by worse policies. Selection uses 30 deterministic episodes every 25,000
transitions, with the same scenario sequence seeded by training seed + 100,000.
Each resumed session selects its own best; scores across phases are not comparable.

Always use the model and normalizer from the **same bundle**. For example,
set `$snapshot` to the complete directory printed by `[Checkpoints] best:`:

```powershell
python main.py --version v2 --phase v2_phase_1a --evaluate "$snapshot/model.zip" --vecnorm "$snapshot/vec_normalize.pkl"
```

To resume a periodic/final/interrupted bundle, set `$snapshot` to its directory:

```powershell
python main.py --version v2 --phase v2_phase_1a --resume "$snapshot/model.zip" --vecnorm "$snapshot/vec_normalize.pkl" --replay-buffer "$snapshot/replay_buffer.pkl" --n-envs 8
```

Use the `n_envs` recorded in `metadata.json` when restoring replay memory.
Best bundles omit replay memory to keep repeated improvements inexpensive;
resume those without `--replay-buffer`. An interrupted write leaves a
`.pending_*` directory and does not update the manifest. Periodic snapshots
are retained without automatic deletion and consume disk space, especially
their replay buffers. Old flat-file examples below apply only to older runs;
the current workflow uses the bundle paths above. Saving is recoverable training
state, not a guarantee of bit-for-bit continuation of simulator/RNG state.

An educational 2D environment in which a **Soft Actor-Critic (SAC)** agent
learns the terminal portion of a Starship-inspired landing: belly flop, flip,
braking manoeuvre, touchdown, engine shutdown, and post-contact stability.

> [!IMPORTANT]
> This is not a high-fidelity flight simulator and does not reproduce SpaceX
> flight software. Vehicle-inspired orders of magnitude provide physical
> context for a reinforcement-learning experiment that can be expanded in
> controlled increments.

V1 deliberately uses one centred virtual actuator. The isolated V2 environment
introduces three independently throttled and gimballed engines while preserving
the symmetric V1 physics. Later iterations can add propellant, ignition
constraints, failures, weather, sensors, and other sources of uncertainty.

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

### V2 three-engine variant

V2 is isolated from both the single-engine versions and the recovered legacy
environment. It shares the V1 curriculum, inertia, aerodynamic belly-flop
moment, angle wrapping, ground contact, success criteria, and mirror-symmetric
left/right dynamics.

Its three engines are placed at `x = -1.15, 0, +1.15 m`. Each healthy engine has
a maximum sea-level thrust of 2.45 MN. V2 exposes nine independent actions:
three throttle commands, three gimbal commands, and three ON/OFF signals. The
engine gate uses a Schmitt-style latch: signals at or above `+0.20` ignite or
keep an engine ON, signals at or below `0.0` command OFF, and values between
`0.0` and `+0.20` preserve the previous state. Once an ignited engine is shut
down it is permanently locked out for the remainder of that episode. While
active, throttle `u` maps linearly to
`30% + 70% * u`. Every initial ignition first produces exactly 30%
thrust for one 50 ms physics step, independently of the stored throttle
command. From the following step, physical throttle applies the linear command
directly; V2 has no additional throttle ramp. The three gimbal actions cover
`+/-30 degrees`, with the same 20 degrees/s slew limit as V1. Random engine
failures remain disabled.

V2 uses a reconstructed 17-value compact observation contract with
`-1/0/+1` engine state encoding. Training uses `norm_obs=True` and
`norm_reward=False`; evaluation uses the associated frozen VecNormalize.

| Group | Count | Values |
| --- | ---: | --- |
| Vehicle and episode state | 7 | `x`, `y`, `vx`, `vy`, angle, angular rate, elapsed time |
| Engine gimbals | 3 | Current left, centre, and right gimbal |
| Altitude context | 1 | AGL |
| Physical throttles | 3 | Current left, centre, and right physical throttle |
| Engine state | 3 | `-1` unavailable, `0` ready/OFF, `+1` ON |

Unavailable includes curriculum-disabled, failed and permanently shut-down
engines. Engine health and lockout remain separate internally. Phase 1A starts
with state codes `[-1, 0, -1]` and only its centre engine enabled, initially
OFF. The agent controls ignition and shutdown: remaining OFF before ignition
preserves availability, but shutting down after ignition permanently locks
the engine for that episode. Phases 1B and 1C use the same lifecycle:
1B starts with codes `[0, -1, 0]` (two lateral engines available), and 1C
with `[0, 0, 0]` (all three available). No engine is forced ON in these phases.

This is a controlled reconstruction of the observation representation and
normalization, not an exact historical rollback: current rewards, SAC settings
and curriculum remain in place. Train from scratch; split 23-input checkpoints
and replay buffers are incompatible. New runs use
`v2_three_engine_compact17_<phase>` output folders to preserve previous runs.

The V2 reward is a separate copy of the V1.1 progress-based reward. It does
not prescribe which engines to use, a throttle schedule, a burn altitude, or
a gimbal allocation. Engine shutdown is irreversible within an episode. It
therefore receives a small convex option cost of
`-5 * delta(locked_engines**2)`: the first shutdown costs `-5`, the second
`-15`, and the third `-25`. This uses no speed, altitude, stopping-distance
model, or prescribed engine sequence. The actuator itself prevents artificial
pulse-width modulation, so there is no repeated ON/OFF switching cost. V2
additionally rewards reduction in AGL by `0.20` per
metre, clipped to two metres per step. The scale transitions smoothly from
`0.20` to `0.60` between 15 and 10 m AGL, then remains at `0.60` through
touchdown to give slow terminal descent a useful learning signal. The same
term becomes negative during ascent, so the stronger terminal shaping cannot
be farmed by repeatedly climbing and descending. Hover still earns no
vertical-progress reward, while ascent is also penalised up to `-3.0` per
step. The additional unpowered-risk term remains zero above 300 m. From 300
to 100 m a smooth penalty grows with downward speed up to `-1.0`, and the positive ground-progress
term is reduced by the same risk factor. This supplies a broad safety signal
without calculating an optimal ignition point. Rotation beyond 90 degrees is
also penalised progressively up to `-3.0` at 120 degrees, leaving the normal
75-87 degree belly-flop spawn unaffected. An interruption of an already-started burn that leaves every curriculum-enabled
engine OFF while the vehicle is airborne and descending receives a one-time
deadstick penalty of
`-(200 + 10 * min(downward_speed, 20))`. Initial OFF engines, engines disabled
by the curriculum, and automatic engine cutoff after touchdown are excluded.
An available engine that has never ignited does not suppress this penalty once
another engine has been shut down permanently; it must be ignited as part of
the same transition to preserve continuous thrust.
A successful V2 landing receives `+1000`, making rare valid touchdowns unambiguous relative
to the `-200` crash and `-350` timeout. The `-0.02` per-step cost and timeout
ordering make waiting indefinitely worse than attempting a landing. V1,
V1.1, V2, and legacy checkpoints are mutually incompatible and require their
own `VecNormalize` files.

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

V2 adds an actuator curriculum before its higher-energy phases:

| Phase | Available engines | AGL | Absolute x | vy | Absolute tilt |
| --- | --- | ---: | ---: | ---: | ---: |
| `v2_phase_1a` | centre | 80-120 m | 0-8 m | -20 to -12 m/s | 0.5-3 deg |
| `v2_phase_1b` | left and right | 100-160 m | 5-15 m | -28 to -18 m/s | 1-5 deg |
| `v2_phase_1c` | all three | 160-220 m | 10-20 m | -35 to -25 m/s | 2-6 deg |

Curriculum-disabled engines remain healthy and OFF; a separate phase mask
ignores their actions without presenting them as failed or locked out. Its
effect is now exposed through the three `engine available` observations, which
supports policy transfer between phases without confusing disabled engines
with available engines that have not yet ignited.
`v2_phase_1` remains an alias of `v2_phase_1c` for old commands. The CLI
defaults to `phase_1` for V1/V1.1 and `v2_phase_1a` for V2.
Promotion is a manual training decision rather than an automatic callback.

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

Start the isolated three-engine V2 actuator curriculum from Phase 1A:

```powershell
python main.py --version v2 --phase v2_phase_1a --seed 42
```

Fresh training currently runs for four million environment timesteps with eight
parallel environments. Models, normalisation statistics, TensorBoard events,
and periodic videos are written under `models/`, `logs/`, and `videos/`.
V2 adds the selected phase to those output directories, so Phase 1A, 1B, and
1C checkpoints and TensorBoard runs do not overwrite one another.

Interrupting either a fresh or resumed SAC run with `Ctrl+C` writes a
synchronised recovery set containing `interrupted_checkpoint.zip`,
`vec_normalize_checkpoint.pkl`, and `replay_buffer_checkpoint.pkl`. The replay
buffer is stored separately because it is not included in the normal SAC model
archive.

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
  --replay-buffer .\models\v1_single_engine\replay_buffer_checkpoint.pkl `
  --phase phase_2 `
  --timesteps 300000 `
  --n-envs 8 `
  --learning-rate 1e-4 `
  --seed 42
```

Then use the resulting checkpoint and normaliser to resume with
`--phase phase_3`.

`SAC.save()` does not include the replay buffer. Passing `--replay-buffer`
restores it and starts gradient updates immediately. The buffer must have been
created with the same observation/action contract and the same number of
parallel environments supplied through `--n-envs`. It should also correspond
to the supplied model and `VecNormalize` snapshot. If the option is omitted,
resume mode creates a fresh buffer and collects 50,000 new transitions before
updating the network. Do not reuse a buffer after changing actuator semantics,
physics, or rewards; in particular, buffers produced before V2's corrected
normalised gimbal mapping are incompatible. Preserve or rename the outputs
from each phase before launching the next one because resumed runs share their
version/phase output directory.

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

Use the optional close camera and pad-relative top-view projection without
changing physics, observations, rewards, or the default render:

```powershell
python main.py `
  --version v2 `
  --evaluate .\models\v2_three_engine_compact17_v2_phase_1a\final_model.zip `
  --vecnorm .\models\v2_three_engine_compact17_v2_phase_1a\vec_normalize.pkl `
  --phase v2_phase_1a `
  --render-layout close_pad `
  --episodes 10 `
  --seed 42
```

`close_pad` increases the tracking-camera zoom from approximately `1.25x` to
`2.5x`. Because the simulation has no lateral `z` coordinate, its top view is
explicitly labelled as a 2-D projection: it shows the pad, the effective
`+/-15 m` touchdown region, projected vehicle position, horizontal velocity,
and current touchdown-envelope status. Omitting `--render-layout` preserves
the previous video layout exactly.

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
improved_rewards_v2.py          Isolated three-engine V2 reward class
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
2. **V2:** three engines, thrust allocation, and per-engine gimbal; later V2
   iterations can add ignition constraints and engine failures.
3. **V3:** fuel, variable mass, failures, and propellant-margin management.
4. **V4:** flaps, wind, weather, uncertainty, and domain randomisation.
5. **V5:** a higher-altitude glide and transition into the terminal manoeuvre.
