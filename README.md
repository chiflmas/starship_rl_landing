# Starship V1: terminal landing with reinforcement learning

An educational 2D project that explores how a **Soft Actor-Critic (SAC)**
agent can control a simplified *belly flop*, *flip*, and *landing burn*
manoeuvre to a landing pad.

> This is not a flight simulator or a model of Starship operational software.
> It uses vehicle-inspired orders of magnitude to provide physical context for
> a reproducible reinforcement-learning problem that can be expanded over time.

The project is designed to increase complexity **incrementally**. Each version
should add one layer, retain a trainable task, and allow meaningful comparison:
first terminal control with one actuator; then multiple engines and thrust
allocation; later fuel, ignition constraints, weather disturbances, wind,
sensors, and other sources of uncertainty.

## Acknowledgement and original project

This repository is based on **[Rocket-recycling with Reinforcement
Learning](https://github.com/jiupinjia/rocket-recycling)** by
**[Zhengxia Zou, Ph.D.](https://zhengxiazou.github.io/)**. Many thanks to
Zhengxia Zou for publishing the original environment and making this excellent
educational starting point available.

The recovered foundation retains the original idea: a rigid rocket in a 2D
plane, thrust vectoring, and reinforcement learning. This branch reorganises
the experiment as a deliberately simple first version of Starship terminal
landing.

## What V1 learns

V1 solves one task: start in a belly-flop state, orient the vehicle, discover
an effective braking strategy, and settle safely on the pad.

The agent receives two continuous actions:

| Action | Range | Meaning |
| --- | --- | --- |
| `throttle` | `0..1` | Total thrust of the virtual actuator. |
| `gimbal` | `-1..1` | Thrust direction, scaled to ±30°. |

Individual ignition, shutdown, and thrust allocation are outside the scope of
this version. This is intentional: it reduces the action space so the
experiment focuses on the manoeuvre itself. Three-engine control will be a
later extension.

## Main changes from the original repository

### V1 centred single-engine environment

- `v1_single_engine` is added and selected by default in `main.py`.
- Three engines are replaced by a centred virtual actuator with up to **6 MN**
  of total thrust. It is centred, so it does not generate torque through an
  engine offset; rotation is controlled through gimbal.
- Mass is **140,000 kg** and height is **50 m**. V1 inertia uses a rigid
  cylinder approximation.
- Control is continuous: shutting down the engine neither locks it out nor
  imposes a minimum throttle. On ground contact, engines are cut and the
  environment validates stability on the landing legs.

### Physics, phases, and symmetry

- Episodes start randomly to the left or right of the pad, **150–200 m** away,
  at **500–600 m AGL**, with initial `vy` between **−95 and −85 m/s** and a
  **75–85°** belly-flop attitude.
- The attitude reference transitions from 80° to upright between **450 m and
  250 m AGL**. The restoring aerodynamic moment fades progressively from
  **450 m to 350 m**, giving gimbal control authority to complete the flip.
- V1 aerodynamics use the geometric projection between the vehicle axis and
  its velocity vector. Mirrored left/right states therefore create the same
  drag and reflected dynamics, removing the asymmetry caused by a fixed angular
  reference.
- The effective touchdown area is **±15 m** from the pad centre. Contact also
  requires safe impact speed, attitude, and angular rate. The vehicle must
  remain stable for 20 steps with engines off; a further second is simulated so
  the green success state is visible in the video.

### Progress-based rewards

`improved_rewards_v1.py` combines progress shaping with terminal signals. It
does **not** prescribe a vertical-speed profile or burn altitude. Instead, it:

- rewards reducing altitude, pad error, attitude error, and lateral velocity;
- uses rewards based on state change rather than occupying a centred or upright
  state, so a hover cannot farm reward indefinitely;
- estimates a braking safety envelope from altitude, downward speed, and
  available deceleration; it penalises states that can no longer reach a safe
  touchdown, without prescribing a burn altitude or velocity profile;
- penalises ascent, angular rate, control effort, and elapsed time;
- lets the agent discover its own late or conservative braking solution;
- assigns `+300` for landing, `-250` for a crash, and `-200` for a timeout.

## Success conditions

Touching the ground alone is not a successful episode. V1 separates impact
validation from post-contact stability validation:

| Stage | Requirement |
| --- | --- |
| Valid contact | Total impact speed `< 7 m/s`, vertical component `< 6 m/s`, `|x| < 15 m`, and attitude `< 10°`. |
| Pad stability | Vehicle on the ground, speed `< 5 m/s`, `|x| < 15 m`, attitude `< 5°`, angular rate `< 3°/s`, and engine off. |
| Confirmation | Stability conditions must hold for **20 consecutive steps** (1 second at `dt = 0.05 s`). |
| Display | After reaching 20/20, the environment runs for 20 more steps to show the green success state in the video. |

If the vehicle leaves the ±15 m zone, exceeds impact limits, tips over, or
fails to stabilise within the two seconds after first contact, the episode ends
as a crash. The automatic engine cut-off after contact prevents a policy from
turning a touchdown into another take-off during this validation.

## Algorithm and current configuration

The project uses Stable-Baselines3 SAC, observations normalised with
`VecNormalize`, and several parallel environments.

| Parameter | Value |
| --- | ---: |
| Policy / critics | `[256, 256]` / `[256, 256, 256]` |
| Initial learning rate | `3e-4` |
| Replay buffer | `400,000` transitions |
| Warm-up | `20,000` steps |
| Batch size | `512` |
| `gamma` / `tau` | `0.998` / `0.005` |
| Entropy coefficient | `auto_0.1` |
| Parallel environments | `6` |
| Gradient steps | `3` |
| Default seed | `42` |

The simulation is lightweight and mostly CPU-bound. Vectorised environments
usually provide more performance than a larger GPU. A GPU accelerates neural
network updates, especially with large batches, but it does not replace physics
parallelism.

## Installation

In PowerShell, from the project directory:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Requirements.txt
```

If you already have a working environment, activate it instead of creating a
new one. To use CUDA, install a PyTorch build compatible with your Python,
Windows version, and NVIDIA driver before installing or adjusting project
dependencies.

## Run

Train V1 from scratch:

```powershell
python main.py
```

By default, training runs for 4 million steps and saves results to:

```text
models/v1_single_engine/
logs/v1_single_engine/
videos/
```

Start TensorBoard:

```powershell
tensorboard --logdir .\logs --port 6006
```

Then open `http://localhost:6006`. Metrics include reward, losses, entropy,
and touchdown/success metrics logged during training.

### Evaluate a checkpoint without training

An SAC checkpoint must always use the `VecNormalize` file created alongside it;
otherwise the policy receives observations on a different scale.

```powershell
python main.py --evaluate .\models\v1_single_engine\final_model.zip `
  --vecnorm .\models\v1_single_engine\vec_normalize.pkl --episodes 10 --seed 42
```

### Resume training

```powershell
python main.py --resume .\models\v1_single_engine\interrupted_checkpoint.zip `
  --vecnorm .\models\v1_single_engine\vec_normalize_checkpoint.pkl `
  --timesteps 300000 --n-envs 6 --learning-rate 1e-4 --seed 42
```

The replay buffer is not restored from an SAC `.zip` file. When resuming, the
program first collects a fresh experience window before updating the network
again. Material changes to physics, rewards, actions, or observations require
training from scratch; previous checkpoints should not be reused.

## Structure

```text
main.py                 Gymnasium wrapper, SAC, callbacks, and CLI
rocket.py               2D dynamics, contact, rendering, and engine state
improved_rewards_v1.py  V1 phase-based rewards
utils.py                Utilities from the recovered project
Requirements.txt        Python dependencies
```

## Limitations and next versions

V1 does not model six degrees of freedom, propellants, mass loss, weather,
wind, flaps, RCS attitude control, sensor noise, Raptor ignition behaviour, or
a full trajectory from tens of kilometres of altitude. Its parameters should
therefore be interpreted as physical context, not as a reproduction of a
SpaceX flight.

A sensible roadmap for future posts and experiments is:

1. **V1:** centred actuator, terminal belly flop, and stable touchdown.
2. **V2:** three engines, throttle allocation, per-engine gimbal, and ignition
   constraints.
3. **V3:** fuel, variable mass, failures, and propellant-margin management.
4. **V4:** flaps, wind, weather conditions, uncertainty, and domain
   randomisation.
5. **V5:** glide trajectory and transition from a higher altitude.

## Licence and citation

Zhengxia Zou's original project is released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This
educational derivative retains attribution, the non-commercial condition, and
the same licence for derivative works. Refer to the original repository for
the complete licence text and attribution.

```bibtex
@misc{zou2021rocket,
  author = {Zhengxia Zou},
  title = {Rocket-recycling with Reinforcement Learning},
  year = {2021},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/jiupinjia/rocket-recycling}}
}
```
