"""Create LinkedIn-ready V2 curriculum evaluation and training-time charts.

The V2 experiment was trained as Phase 1a -> Phase 1c -> Phase 2 -> Phase 3.
Some phases contain several resumed TensorBoard sessions, so scalar events are
merged recursively and duplicate global timesteps keep the newest event.

Outputs:
    gallery/v2_curriculum_evaluations.{html,png,csv}
    gallery/v1_1_vs_v2_training_comparison.{html,png,csv}
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


EVAL_REWARD = "eval/mean_reward"
EVAL_LENGTH = "eval/mean_ep_length"
EVAL_SUCCESS = "eval/success_rate"


@dataclass(frozen=True)
class PhaseSpec:
    key: str
    label: str
    directory: str
    color: str
    selected_step: int | None = None


V2_PHASES = (
    PhaseSpec(
        "phase_1a",
        "Phase 1a · Single-engine touchdown",
        "v2_three_engine_compact17_v2_phase_1a",
        "#4CC9F0",
        1_175_000,
    ),
    PhaseSpec(
        "phase_1c",
        "Phase 1c · Engine sequencing",
        "v2_three_engine_compact17_v2_phase_1c_resumed",
        "#B388FF",
        3_715_000,
    ),
    PhaseSpec(
        "phase_2",
        "Phase 2 · Intermediate flip",
        "v2_three_engine_compact17_v2_phase_2_resumed",
        "#F4A261",
        5_865_496,
    ),
    PhaseSpec(
        "phase_3",
        "Phase 3 · Full manoeuvre",
        "v2_three_engine_compact17_v2_phase_3_resumed",
        "#66D17A",
        6_465_496,
    ),
)

V1_1_PHASES = (
    PhaseSpec("phase_1", "Phase 1", "starship_sac_optimized_phase1", "#4CC9F0"),
    PhaseSpec("phase_2", "Phase 2", "starship_sac_resumed_phase2", "#F4A261"),
    PhaseSpec(
        "phase_2_5",
        "Phase 2.5",
        "starship_sac_resumed_phase2_5",
        "#FF8C42",
    ),
    PhaseSpec("phase_3", "Phase 3", "starship_sac_resumed_phase3", "#66D17A"),
)


def _event_files(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing TensorBoard directory: {path}")
    files = sorted(path.rglob("events.out.tfevents*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files below: {path}")
    return files


def _load_event_file(path: Path) -> tuple[dict[str, list], tuple[float, float] | None]:
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = accumulator.Tags().get("scalars", [])
    scalars = {tag: accumulator.Scalars(tag) for tag in available}

    # Active wall-clock duration is measured inside each event file, so pauses
    # between curriculum phases or between resumed commands are not counted.
    wall_times = [event.wall_time for events in scalars.values() for event in events]
    wall_range = (min(wall_times), max(wall_times)) if wall_times else None
    return scalars, wall_range


def _merge_tag(event_sets: list[dict[str, list]], tag: str, required: bool = True):
    latest_by_step = {}
    for scalars in event_sets:
        for event in scalars.get(tag, []):
            previous = latest_by_step.get(int(event.step))
            if previous is None or event.wall_time >= previous.wall_time:
                latest_by_step[int(event.step)] = event
    if required and not latest_by_step:
        raise KeyError(f"TensorBoard tag {tag!r} not found")
    return [
        (step, float(event.value), float(event.wall_time))
        for step, event in sorted(latest_by_step.items())
    ]


def load_phase(root: Path, spec: PhaseSpec):
    files = _event_files(root / spec.directory)
    event_sets = []
    wall_ranges = []
    for event_file in files:
        scalars, wall_range = _load_event_file(event_file)
        event_sets.append(scalars)
        if wall_range and wall_range[1] > wall_range[0]:
            wall_ranges.append(wall_range)

    reward = _merge_tag(event_sets, EVAL_REWARD)
    length = _merge_tag(event_sets, EVAL_LENGTH)
    success = _merge_tag(event_sets, EVAL_SUCCESS, required=False)
    reward_steps = [step for step, _, _ in reward]
    if reward_steps != [step for step, _, _ in length]:
        raise ValueError(f"Reward and episode-length steps do not align in {spec.directory}")
    if success and reward_steps != [step for step, _, _ in success]:
        raise ValueError(f"Reward and success-rate steps do not align in {spec.directory}")

    return {
        "key": spec.key,
        "label": spec.label,
        "color": spec.color,
        "reward": reward,
        "episode_length": length,
        "success_rate": success,
        "start_step": reward_steps[0],
        "end_step": reward_steps[-1],
        "active_seconds": sum(end - start for start, end in wall_ranges),
        "wall_ranges": wall_ranges,
        "event_files": len(files),
        "has_explicit_selection": spec.selected_step is not None,
        "selected_step": spec.selected_step or reward_steps[-1],
        "selected_wall_time": next(
            wall_time
            for step, _, wall_time in reward
            if step == (spec.selected_step or reward_steps[-1])
        ),
    }


def write_eval_csv(path: Path, phases) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "phase",
                "timestep",
                "mean_reward",
                "success_rate",
                "mean_episode_length",
            ),
        )
        writer.writeheader()
        for phase in phases:
            lengths = {step: value for step, value, _ in phase["episode_length"]}
            successes = {step: value for step, value, _ in phase["success_rate"]}
            for step, reward, _ in phase["reward"]:
                writer.writerow(
                    {
                        "phase": phase["key"],
                        "timestep": step,
                        "mean_reward": reward,
                        "success_rate": successes.get(step, ""),
                        "mean_episode_length": lengths[step],
                    }
                )


def build_eval_figure(phases, width: int, height: int):
    has_success = all(phase["success_rate"] for phase in phases)
    rows = 3 if has_success else 2
    titles = ["Mean evaluation reward"]
    if has_success:
        titles.append("Landing success rate")
    titles.append("Mean evaluation episode length")
    length_row = rows

    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09 if rows == 3 else 0.13,
        subplot_titles=tuple(titles),
    )

    for phase in phases:
        marker = {
            "color": phase["color"],
            "size": 9,
            "line": {"color": "#071426", "width": 1.5},
        }
        for row, field, name, suffix in (
            (1, "reward", "Mean reward", ""),
            (length_row, "episode_length", "Mean length", " steps"),
        ):
            values = phase[field]
            figure.add_trace(
                go.Scatter(
                    x=[item[0] for item in values],
                    y=[item[1] for item in values],
                    mode="lines+markers",
                    name=phase["label"],
                    legendgroup=phase["key"],
                    showlegend=row == 1,
                    line={"color": phase["color"], "width": 3.5},
                    marker=marker,
                    hovertemplate=(
                        f"<b>{phase['label']}</b><br>"
                        f"Timestep: %{{x:,}}<br>{name}: %{{y:.1f}}{suffix}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
        if has_success:
            values = phase["success_rate"]
            figure.add_trace(
                go.Scatter(
                    x=[item[0] for item in values],
                    y=[100.0 * item[1] for item in values],
                    mode="lines+markers",
                    name=phase["label"],
                    legendgroup=phase["key"],
                    showlegend=False,
                    line={"color": phase["color"], "width": 3.5},
                    marker=marker,
                    hovertemplate=(
                        f"<b>{phase['label']}</b><br>"
                        "Timestep: %{x:,}<br>Success: %{y:.0f}%<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

        figure.add_vrect(
            x0=phase["start_step"],
            x1=phase["end_step"],
            fillcolor=phase["color"],
            opacity=0.055,
            line_width=0,
            layer="below",
            row="all",
            col=1,
        )

    for phase in phases[1:]:
        figure.add_vline(
            x=phase["start_step"],
            line={"color": phase["color"], "width": 2, "dash": "dot"},
            row="all",
            col=1,
        )

    # A coral band is not another curriculum phase. It shows samples collected
    # after the checkpoint that was eventually selected, typically because an
    # unattended overnight command continued training past its best policy.
    for phase in phases:
        overrun = phase["end_step"] - phase["selected_step"]
        if overrun >= 250_000:
            figure.add_vrect(
                x0=phase["selected_step"],
                x1=phase["end_step"],
                fillcolor="#E76F51",
                opacity=0.10,
                line={"color": "#E76F51", "width": 1, "dash": "dash"},
                layer="below",
                row="all",
                col=1,
            )

    if has_success:
        selected_x = []
        selected_y = []
        selected_text = []
        for phase in phases:
            successes = {step: value for step, value, _ in phase["success_rate"]}
            selected_x.append(phase["selected_step"])
            selected_y.append(100.0 * successes[phase["selected_step"]])
            selected_text.append(phase["label"])
        figure.add_trace(
            go.Scatter(
                x=selected_x,
                y=selected_y,
                mode="markers",
                name="Selected best checkpoint",
                marker={
                    "symbol": "star",
                    "size": 18,
                    "color": "#FFD166",
                    "line": {"color": "#071426", "width": 2},
                },
                text=selected_text,
                hovertemplate=(
                    "<b>%{text}</b><br>Selected at %{x:,} steps"
                    "<br>Success: %{y:.0f}%<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    figure.add_hline(
        y=0,
        line={"color": "#8AA1B6", "width": 1, "dash": "dash"},
        row=1,
        col=1,
    )
    if has_success:
        figure.update_yaxes(range=[-3, 103], ticksuffix="%", row=2, col=1)

    figure.update_layout(
        width=width,
        height=height,
        title={
            "text": (
                "<b>Learning to land with three independently controlled engines</b>"
                "<br><span style='font-size:20px;color:#AFC5D8'>"
                "V2 curriculum · deterministic SAC evaluations · Phase 1a → 1c → 2 → 3"
                "</span>"
            ),
            "x": 0.055,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
            "font": {"size": 33, "color": "#F4F8FC"},
        },
        paper_bgcolor="#071426",
        plot_bgcolor="#0B1B2E",
        font={"family": "Arial", "color": "#DCE8F2", "size": 16},
        margin={"l": 120, "r": 70, "t": 220, "b": 115},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.075,
            "yanchor": "bottom",
            "font": {"size": 15},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hovermode="x unified",
    )
    figure.update_annotations(font={"size": 20, "color": "#F4F8FC"})
    for row in range(1, rows + 1):
        figure.update_xaxes(
            gridcolor="#263E53",
            zeroline=False,
            showline=True,
            linecolor="#5B738A",
            row=row,
            col=1,
        )
        figure.update_yaxes(
            gridcolor="#263E53",
            zeroline=False,
            showline=True,
            linecolor="#5B738A",
            row=row,
            col=1,
        )
    figure.update_xaxes(
        title_text="Total environment timesteps",
        tickformat=".2s",
        row=rows,
        col=1,
    )
    figure.update_yaxes(title_text="Mean reward", row=1, col=1)
    if has_success:
        figure.update_yaxes(title_text="Success", row=2, col=1)
    figure.update_yaxes(title_text="Steps", row=length_row, col=1)
    figure.add_annotation(
        x=0.0,
        y=-0.075,
        xref="paper",
        yref="paper",
        text=(
            "Source: TensorBoard deterministic evaluations. Resumed sessions are merged by "
            "global timestep; no smoothing applied. Stars mark selected checkpoints; coral "
            "bands show unattended training that continued after them."
        ),
        showarrow=False,
        xanchor="left",
        font={"size": 13, "color": "#8EA8BC"},
    )
    return figure


def experiment_summary(name: str, phases) -> dict:
    first_step = min(phase["start_step"] for phase in phases)
    final_step = max(phase["end_step"] for phase in phases)
    selected_step = phases[-1]["selected_step"]
    selected_wall_time = phases[-1]["selected_wall_time"]
    active_seconds = sum(phase["active_seconds"] for phase in phases)
    if phases[-1]["has_explicit_selection"]:
        active_to_selected = 0.0
        for phase in phases:
            for start, end in phase["wall_ranges"]:
                active_to_selected += max(0.0, min(end, selected_wall_time) - start)
        active_to_selected = min(active_to_selected, active_seconds)
    else:
        # Older V1.1 logs do not carry success-first checkpoint metadata. Use
        # the complete recorded experiment rather than inventing an overrun.
        active_to_selected = active_seconds
    return {
        "version": name,
        "first_eval_step": first_step,
        "final_eval_step": final_step,
        "selected_step": selected_step,
        "post_selected_steps": max(0, final_step - selected_step),
        "evaluated_span": final_step - first_step,
        "active_hours_to_selected": active_to_selected / 3600.0,
        "post_selected_active_hours": (active_seconds - active_to_selected) / 3600.0,
        "active_hours": active_seconds / 3600.0,
        "evaluation_checkpoints": sum(len(phase["reward"]) for phase in phases),
    }


def write_comparison_csv(path: Path, summaries) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=tuple(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def build_comparison_figure(summaries, width: int, height: int):
    colors = ["#4CC9F0", "#F4A261"]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Training horizon", "Active logged training time"),
        horizontal_spacing=0.16,
    )
    names = [item["version"] for item in summaries]
    figure.add_trace(
        go.Bar(
            x=names,
            y=[item["selected_step"] / 1_000_000 for item in summaries],
            marker_color=colors,
            text=[f"{item['selected_step'] / 1_000_000:.2f}M" for item in summaries],
            textposition="outside",
            hovertemplate="%{x}<br>Selected checkpoint: %{y:.2f}M<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=names,
            y=[item["active_hours_to_selected"] for item in summaries],
            marker_color=colors,
            text=[f"{item['active_hours_to_selected']:.1f} h" for item in summaries],
            textposition="outside",
            hovertemplate="%{x}<br>Active time to selection: %{y:.1f} h<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        width=width,
        height=height,
        title={
            "text": (
                "<b>Training cost of independent three-engine control</b>"
                "<br><span style='font-size:19px;color:#AFC5D8'>"
                "V1.1 single engine vs V2 three engines · selected checkpoints"
                "</span>"
            ),
            "x": 0.06,
            "xanchor": "left",
            "y": 0.96,
            "font": {"size": 32, "color": "#F4F8FC"},
        },
        paper_bgcolor="#071426",
        plot_bgcolor="#0B1B2E",
        font={"family": "Arial", "color": "#DCE8F2", "size": 17},
        margin={"l": 105, "r": 70, "t": 175, "b": 105},
        bargap=0.42,
    )
    figure.update_annotations(font={"size": 21, "color": "#F4F8FC"})
    figure.update_yaxes(
        title_text="Million environment timesteps",
        gridcolor="#263E53",
        rangemode="tozero",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Hours",
        gridcolor="#263E53",
        rangemode="tozero",
        row=1,
        col=2,
    )
    figure.add_annotation(
        x=0.0,
        y=-0.17,
        xref="paper",
        yref="paper",
        text=(
            "Active time is derived from TensorBoard event timestamps and excludes pauses "
            "between commands. Both bars use the checkpoint selected for the published policy."
        ),
        showarrow=False,
        xanchor="left",
        font={"size": 13, "color": "#8EA8BC"},
    )
    return figure


def _write_figure(figure, output_stem: Path, scale: float, skip_png: bool) -> None:
    figure.write_html(output_stem.with_suffix(".html"), include_plotlyjs=True)
    if not skip_png:
        figure.write_image(output_stem.with_suffix(".png"), scale=scale)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot V2 evaluations and compare V2 training time with V1.1."
    )
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("gallery"))
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1600)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--skip-png",
        action="store_true",
        help="Write HTML and CSV only when Plotly image export is unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v2_phases = [load_phase(args.logs_dir, spec) for spec in V2_PHASES]
    v1_1_root = args.logs_dir / "v1_1_single_engine"
    v1_1_phases = [load_phase(v1_1_root, spec) for spec in V1_1_PHASES]

    eval_stem = args.output_dir / "v2_curriculum_evaluations"
    eval_figure = build_eval_figure(v2_phases, args.width, args.height)
    _write_figure(eval_figure, eval_stem, args.scale, args.skip_png)
    write_eval_csv(eval_stem.with_suffix(".csv"), v2_phases)

    summaries = [
        experiment_summary("V1.1 · one engine", v1_1_phases),
        experiment_summary("V2 · three engines", v2_phases),
    ]
    comparison_stem = args.output_dir / "v1_1_vs_v2_training_comparison"
    comparison_figure = build_comparison_figure(
        summaries,
        args.width,
        max(900, int(args.height * 0.68)),
    )
    _write_figure(comparison_figure, comparison_stem, args.scale, args.skip_png)
    write_comparison_csv(comparison_stem.with_suffix(".csv"), summaries)

    print("\nEvaluation data")
    for phase in v2_phases:
        success = phase["success_rate"]
        final_success = f"{100.0 * success[-1][1]:.0f}%" if success else "not logged"
        print(
            f"  {phase['label']}: {len(phase['reward'])} checkpoints, "
            f"steps {phase['start_step']:,}–{phase['end_step']:,}, "
            f"final success {final_success}, active {phase['active_seconds'] / 3600:.2f} h"
        )
    print("\nComparison")
    for summary in summaries:
        print(
            f"  {summary['version']}: final eval {summary['final_eval_step']:,} steps, "
            f"selected at {summary['selected_step']:,}, "
            f"active logged time {summary['active_hours']:.2f} h "
            f"({summary['post_selected_active_hours']:.2f} h after selection)"
        )
    print(f"\nV2 chart:        {eval_stem.with_suffix('.html').resolve()}")
    print(f"Comparison chart: {comparison_stem.with_suffix('.html').resolve()}")


if __name__ == "__main__":
    main()
