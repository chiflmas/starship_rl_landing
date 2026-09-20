"""Actor activation visualizations for deterministic SAC evaluations.

This module is intentionally independent from the environment and training
code.  It records the two ReLU layers used by the V1 actor, selects a stable
subset of representative neurons for an episode, and renders an explanatory
network animation without changing model inference.
"""

from dataclasses import dataclass
from pathlib import Path
import unicodedata

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


V1_INPUT_LABELS = (
    "x",
    "AGL",
    "vx",
    "vy",
    "sin(theta)",
    "cos(theta)",
    "angular rate",
    "previous throttle",
    "previous gimbal",
    "attitude error",
    "braking margin",
    "time remaining",
)

V1_1_INPUT_LABELS = (
    "x",
    "AGL",
    "vx",
    "vy",
    "sin(theta)",
    "cos(theta)",
    "angular rate",
    "previous throttle",
    "previous gimbal",
    "time remaining",
)

V2_INPUT_LABELS = (
    "x",
    "y",
    "vx",
    "vy",
    "theta",
    "angular rate",
    "elapsed time",
    "left gimbal",
    "centre gimbal",
    "right gimbal",
    "AGL",
    "left throttle",
    "centre throttle",
    "right throttle",
    "left engine state",
    "centre engine state",
    "right engine state",
)

V1_ACTION_LABELS = ("Throttle", "Gimbal")
V2_ACTION_LABELS = (
    "Throttle L", "Throttle C", "Throttle R",
    "Gimbal L", "Gimbal C", "Gimbal R",
    "ON/OFF L", "ON/OFF C", "ON/OFF R",
)


def _input_labels(observation_dims):
    labels_by_dimension = {
        len(V1_INPUT_LABELS): V1_INPUT_LABELS,
        len(V1_1_INPUT_LABELS): V1_1_INPUT_LABELS,
        len(V2_INPUT_LABELS): V2_INPUT_LABELS,
    }
    try:
        return labels_by_dimension[int(observation_dims)]
    except KeyError as exc:
        raise ValueError(
            "No hay etiquetas de visualizacion para un vector de "
            f"{observation_dims} observaciones."
        ) from exc


def _action_labels(action_dims):
    labels_by_dimension = {
        len(V1_ACTION_LABELS): V1_ACTION_LABELS,
        len(V2_ACTION_LABELS): V2_ACTION_LABELS,
    }
    try:
        return labels_by_dimension[int(action_dims)]
    except KeyError as exc:
        raise ValueError(
            "No hay etiquetas de visualizacion para un vector de "
            f"{action_dims} acciones."
        ) from exc


@dataclass
class ActivationSample:
    observation: np.ndarray
    action: np.ndarray
    hidden_1: np.ndarray
    hidden_2: np.ndarray
    step: int


class ActorActivationRecorder:
    """Capture post-ReLU actor activations during ``model.predict()``."""

    def __init__(self, actor):
        self.actor = actor
        self._latest = {}
        relu_layers = [
            module
            for module in actor.latent_pi.modules()
            if isinstance(module, torch.nn.ReLU)
        ]
        if len(relu_layers) != 2:
            raise ValueError(
                "La visualizacion espera exactamente dos capas ReLU en el actor; "
                f"se encontraron {len(relu_layers)}."
            )

        self._handles = [
            layer.register_forward_hook(self._make_hook(index))
            for index, layer in enumerate(relu_layers)
        ]

    def _make_hook(self, index):
        def hook(_module, _inputs, output):
            self._latest[index] = output.detach().cpu().numpy().reshape(-1).copy()

        return hook

    def begin_step(self):
        self._latest.clear()

    def capture(self, observation, action, step):
        if 0 not in self._latest or 1 not in self._latest:
            raise RuntimeError(
                "No se capturaron las dos capas del actor durante model.predict()."
            )
        return ActivationSample(
            observation=np.asarray(observation, dtype=np.float32).reshape(-1).copy(),
            action=np.asarray(action, dtype=np.float32).reshape(-1).copy(),
            hidden_1=self._latest[0],
            hidden_2=self._latest[1],
            step=int(step),
        )

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _actor_weights(actor):
    linears = [
        module
        for module in actor.latent_pi.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if len(linears) != 2 or not isinstance(actor.mu, torch.nn.Linear):
        raise ValueError("Arquitectura del actor no compatible con la visualizacion V1.")
    return tuple(
        layer.weight.detach().cpu().numpy().copy()
        for layer in (*linears, actor.mu)
    )


def _representative_neurons(values, count):
    scores = np.mean(np.abs(values), axis=0)
    count = min(int(count), scores.size)
    return np.argsort(scores)[-count:][::-1]


def _percentile_scale(values, percentile=99.0):
    scale = float(np.percentile(np.abs(values), percentile))
    return max(scale, 1e-6)


def _positions(x, count, top=92, bottom=650):
    return [(int(x), int(y)) for y in np.linspace(top, bottom, count)]


def _blend_with_white(color, strength):
    strength = float(np.clip(strength, 0.0, 1.0))
    return tuple(
        int(round(255.0 * (1.0 - strength) + channel * strength))
        for channel in color
    )


def _draw_node(canvas, position, value, scale, base_color, radius=8):
    strength = min(abs(float(value)) / max(scale, 1e-6), 1.0)
    color = _blend_with_white(base_color, 0.18 + 0.82 * strength)
    cv2.circle(canvas, position, radius, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, position, radius, (54, 65, 82), 2, cv2.LINE_AA)


def _draw_edges(canvas, source_positions, target_positions, weights,
                source_values, contribution_scale):
    positive = (79, 104, 232)   # RGB blue
    negative = (245, 132, 31)   # RGB orange
    for target_index, target_position in enumerate(target_positions):
        for source_index, source_position in enumerate(source_positions):
            weight = float(weights[target_index, source_index])
            contribution = abs(weight * float(source_values[source_index]))
            strength = min(contribution / max(contribution_scale, 1e-6), 1.0)
            if strength < 0.025:
                continue
            color = positive if weight >= 0.0 else negative
            line_color = _blend_with_white(color, 0.10 + 0.65 * strength)
            thickness = 1 if strength < 0.60 else 2
            cv2.line(
                canvas,
                source_position,
                target_position,
                line_color,
                thickness,
                cv2.LINE_AA,
            )


def _put_text(canvas, text, position, scale=0.48, color=(26, 32, 44),
              thickness=1):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    cv2.putText(
        canvas,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _prepare_episode(samples, actor, shown_neurons):
    observations = np.stack([sample.observation for sample in samples])
    actions = np.stack([sample.action for sample in samples])
    hidden_1 = np.stack([sample.hidden_1 for sample in samples])
    hidden_2 = np.stack([sample.hidden_2 for sample in samples])
    w1, w2, wout = _actor_weights(actor)

    h1_indices = _representative_neurons(hidden_1, shown_neurons)
    h2_indices = _representative_neurons(hidden_2, shown_neurons)
    w1_shown = w1[h1_indices, :]
    w2_shown = w2[np.ix_(h2_indices, h1_indices)]
    wout_shown = wout[:, h2_indices]

    contribution_1 = observations[:, None, :] * w1_shown[None, :, :]
    contribution_2 = (
        hidden_1[:, h1_indices][:, None, :] * w2_shown[None, :, :]
    )
    contribution_out = (
        hidden_2[:, h2_indices][:, None, :] * wout_shown[None, :, :]
    )

    return {
        "input_labels": _input_labels(observations.shape[1]),
        "action_labels": _action_labels(actions.shape[1]),
        "observations": observations,
        "actions": actions,
        "hidden_1": hidden_1,
        "hidden_2": hidden_2,
        "h1_indices": h1_indices,
        "h2_indices": h2_indices,
        "w1": w1_shown,
        "w2": w2_shown,
        "wout": wout_shown,
        "input_scale": _percentile_scale(observations),
        "h1_scale": _percentile_scale(hidden_1[:, h1_indices]),
        "h2_scale": _percentile_scale(hidden_2[:, h2_indices]),
        "c1_scale": _percentile_scale(contribution_1),
        "c2_scale": _percentile_scale(contribution_2),
        "cout_scale": _percentile_scale(contribution_out),
    }


def _render_network_frame(sample, prepared, frame_index, phase, status,
                          width=1400, height=720):
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.rectangle(canvas, (12, 12), (width - 13, height - 13), (205, 213, 224), 1)

    input_labels = prepared["input_labels"]
    input_positions = _positions(245, len(input_labels), 80, 635)
    h1_positions = _positions(520, len(prepared["h1_indices"]), 85, 635)
    h2_positions = _positions(795, len(prepared["h2_indices"]), 85, 635)
    output_positions = (
        [(1045, 270), (1045, 450)]
        if len(prepared["action_labels"]) == 2
        else _positions(1045, len(prepared["action_labels"]), 80, 635)
    )

    observation = sample.observation
    hidden_1 = sample.hidden_1[prepared["h1_indices"]]
    hidden_2 = sample.hidden_2[prepared["h2_indices"]]
    action = sample.action

    _draw_edges(
        canvas, input_positions, h1_positions, prepared["w1"],
        observation, prepared["c1_scale"],
    )
    _draw_edges(
        canvas, h1_positions, h2_positions, prepared["w2"],
        hidden_1, prepared["c2_scale"],
    )
    _draw_edges(
        canvas, h2_positions, output_positions, prepared["wout"],
        hidden_2, prepared["cout_scale"],
    )

    for index, (label, position, value) in enumerate(
        zip(input_labels, input_positions, observation)
    ):
        color = (75, 145, 222) if value >= 0.0 else (245, 132, 31)
        _draw_node(canvas, position, value, prepared["input_scale"], color, 14)
        _put_text(canvas, label, (28, position[1] - 3), 0.39)
        _put_text(canvas, f"{value:+.2f}", (272, position[1] + 5), 0.34, (82, 92, 108))

    for position, value, neuron_index in zip(
        h1_positions, hidden_1, prepared["h1_indices"]
    ):
        _draw_node(canvas, position, value, prepared["h1_scale"], (117, 144, 180))
        _put_text(canvas, str(int(neuron_index)), (position[0] + 21, position[1] + 5), 0.32)

    for position, value, neuron_index in zip(
        h2_positions, hidden_2, prepared["h2_indices"]
    ):
        _draw_node(canvas, position, value, prepared["h2_scale"], (117, 144, 180))
        _put_text(canvas, str(int(neuron_index)), (position[0] + 21, position[1] + 5), 0.32)

    if len(action) == 2:
        throttle_command = float(np.clip(action[0], 0.0, 1.0))
        physical_throttle = throttle_command ** 3
        gimbal_command = float(np.clip(action[1], -1.0, 1.0))
        gimbal_degrees = 30.0 * gimbal_command
        _draw_node(canvas, output_positions[0], throttle_command, 1.0, (225, 91, 95), 22)
        _draw_node(canvas, output_positions[1], gimbal_command, 1.0, (225, 91, 95), 22)
        _put_text(canvas, "Throttle", (1080, 259), 0.46)
        _put_text(canvas, f"cmd {throttle_command:.2f} | physical {100*physical_throttle:.1f}%",
                  (1080, 281), 0.37, (82, 92, 108))
        _put_text(canvas, "Gimbal", (1080, 439), 0.46)
        _put_text(canvas, f"cmd {gimbal_command:+.2f} | {gimbal_degrees:+.1f} deg",
                  (1080, 461), 0.37, (82, 92, 108))
    else:
        for label, position, value in zip(
            prepared["action_labels"], output_positions, action
        ):
            _draw_node(canvas, position, value, 1.0, (225, 91, 95), 16)
            _put_text(canvas, label, (1070, position[1] - 3), 0.36)
            _put_text(
                canvas,
                f"{value:+.2f}",
                (1070, position[1] + 16),
                0.31,
                (82, 92, 108),
            )

    _put_text(canvas, "Policy inputs (after VecNormalize)", (28, 43), 0.56, thickness=2)
    _put_text(canvas, "Hidden layer 1", (455, 43), 0.56, thickness=2)
    _put_text(canvas, "Hidden layer 2", (730, 43), 0.56, thickness=2)
    _put_text(canvas, "Actions", (1010, 43), 0.56, thickness=2)
    _put_text(
        canvas,
        f"Phase {phase} | step {sample.step:03d} | t={sample.step * 0.05:5.2f}s | {status}",
        (28, 690),
        0.43,
        (58, 68, 84),
    )
    _put_text(canvas, "Blue: positive weight", (430, 690), 0.38, (79, 104, 232))
    _put_text(canvas, "Orange: negative weight", (650, 690), 0.38, (220, 105, 20))
    _put_text(canvas, "Brightness: activation", (900, 690), 0.38, (82, 92, 108))
    _put_text(canvas, "Top neurons by mean activation (fixed for this episode)",
              (420, 668), 0.34, (110, 119, 133))
    return canvas


def _resize_simulation_frame(frame, target_height):
    frame = np.asarray(frame, dtype=np.uint8)
    width = max(2, int(round(frame.shape[1] * target_height / frame.shape[0])))
    if width % 2:
        width += 1
    return cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)


def export_activation_visualization(
    simulation_frames,
    samples,
    actor,
    output_folder,
    episode_number,
    phase,
    status,
    shown_neurons=36,
    video_fps=20,
    
    gif_fps=10,
):
    """Export a network GIF and a synchronized simulation/network MP4."""
    frame_count = min(len(simulation_frames), len(samples))
    if frame_count == 0:
        return {}
    simulation_frames = simulation_frames[:frame_count]
    samples = samples[:frame_count]
    prepared = _prepare_episode(samples, actor, shown_neurons)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{int(episode_number):02d}"
    gif_path = output_folder / f"{stem}_network.gif"
    combined_path = output_folder / f"{stem}_combined.mp4"

    gif_stride = max(1, int(round(video_fps / gif_fps)))
    # Stream frames to the writers instead of retaining the large composite
    # animation in RAM. The evaluation already holds its simulation frames.
    with imageio.get_writer(
        gif_path,
        mode="I",
        duration=1000.0 / float(gif_fps),
        loop=0,
    ) as gif_writer, imageio.get_writer(
        combined_path,
        fps=video_fps,
        codec="libx264",
        macro_block_size=2,
    ) as combined_writer:
        for index, (simulation_frame, sample) in enumerate(
            zip(simulation_frames, samples)
        ):
            network_frame = _render_network_frame(
                sample, prepared, index, phase=phase, status=status
            )
            if index % gif_stride == 0:
                gif_frame = cv2.resize(
                    network_frame, (980, 504), interpolation=cv2.INTER_AREA
                )
                gif_writer.append_data(gif_frame)
            simulation_resized = _resize_simulation_frame(
                simulation_frame, network_frame.shape[0]
            )
            combined_writer.append_data(
                np.concatenate((simulation_resized, network_frame), axis=1)
            )
    return {
        "network_gif": str(gif_path),
        "combined_video": str(combined_path),
    }
