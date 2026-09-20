"""Create LinkedIn-ready Plotly charts from the V1.1 curriculum logs.

This is intentionally separate from ``plot_v1_curriculum_evaluations.py`` so
the original V1 figure remains reproducible. V1.1 has four training stages,
including the Phase 2.5 angular-control bridge, and uses different TensorBoard
run-directory names.
"""

import argparse
import csv
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PHASES = (
    {
        "key": "phase1",
        "label": "Phase 1 · Touchdown",
        "directory": "starship_sac_optimized_phase1",
        "color": "#4CC9F0",
    },
    {
        "key": "phase2",
        "label": "Phase 2 · Approach",
        "directory": "starship_sac_resumed_phase2",
        "color": "#F4A261",
    },
    {
        "key": "phase2_5",
        "label": "Phase 2.5 · Angular-control bridge",
        "directory": "starship_sac_resumed_phase2_5",
        "color": "#FF8C42",
    },
    {
        "key": "phase3",
        "label": "Phase 3 · Full manoeuvre",
        "directory": "starship_sac_resumed_phase3",
        "color": "#66D17A",
    },
)

EVAL_REWARD = "eval/mean_reward"
EVAL_LENGTH = "eval/mean_ep_length"


def scalar_values(accumulator, tag):
    available = accumulator.Tags().get("scalars", [])
    if tag not in available:
        raise KeyError(f"TensorBoard tag {tag!r} not found")
    return [(int(event.step), float(event.value)) for event in accumulator.Scalars(tag)]


def deduplicate_by_step(values):
    """Keep the latest event when a resumed run contains duplicate steps."""
    by_step = {}
    for step, value in values:
        by_step[step] = value
    return sorted(by_step.items())


def load_phase(logs_dir, phase):
    event_dir = logs_dir / phase["directory"]
    if not event_dir.is_dir():
        raise FileNotFoundError(f"Missing TensorBoard directory: {event_dir}")

    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    reward = deduplicate_by_step(scalar_values(accumulator, EVAL_REWARD))
    episode_length = deduplicate_by_step(scalar_values(accumulator, EVAL_LENGTH))

    if not reward:
        raise ValueError(f"No evaluation checkpoints found in {event_dir}")
    if [step for step, _ in reward] != [step for step, _ in episode_length]:
        raise ValueError(f"Evaluation steps do not align in {event_dir}")

    # Evaluation ranges are used for the background bands. Rollout logs from
    # resumed checkpoints can overlap, so using them would falsely imply that
    # two curriculum phases were active at the same time.
    return {
        **phase,
        "reward": reward,
        "episode_length": episode_length,
        "start_step": reward[0][0],
        "end_step": reward[-1][0],
    }


def write_csv(path, phases):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("phase", "timestep", "mean_reward", "mean_episode_length"),
        )
        writer.writeheader()
        for phase in phases:
            lengths = dict(phase["episode_length"])
            for step, reward in phase["reward"]:
                writer.writerow({
                    "phase": phase["key"],
                    "timestep": step,
                    "mean_reward": reward,
                    "mean_episode_length": lengths[step],
                })


def add_phase_trace(figure, phase):
    reward_steps, rewards = zip(*phase["reward"])
    length_steps, lengths = zip(*phase["episode_length"])

    common_marker = {
        "color": phase["color"],
        "size": 11,
        "line": {"color": "#071426", "width": 2},
    }
    figure.add_trace(
        go.Scatter(
            x=reward_steps,
            y=rewards,
            mode="lines+markers",
            name=phase["label"],
            legendgroup=phase["key"],
            line={"color": phase["color"], "width": 4},
            marker=common_marker,
            hovertemplate=(
                f"<b>{phase['label']}</b><br>"
                "Timestep: %{x:,}<br>Mean reward: %{y:.1f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=length_steps,
            y=lengths,
            mode="lines+markers",
            name=phase["label"],
            legendgroup=phase["key"],
            showlegend=False,
            line={"color": phase["color"], "width": 4},
            marker=common_marker,
            hovertemplate=(
                f"<b>{phase['label']}</b><br>"
                "Timestep: %{x:,}<br>Mean length: %{y:.1f} steps<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )


def build_figure(phases, width, height):
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.13,
        subplot_titles=("Mean evaluation reward", "Mean evaluation episode length"),
    )

    for phase in phases:
        add_phase_trace(figure, phase)
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

    figure.add_hline(
        y=0,
        line={"color": "#8AA1B6", "width": 1, "dash": "dash"},
        row=1,
        col=1,
    )
    figure.update_layout(
        width=width,
        height=height,
        title={
            "text": (
                "<b>Learning without engineered guidance</b>"
                "<br><span style='font-size:20px;color:#AFC5D8'>"
                "V1.1 curriculum · 10-input SAC policy · deterministic evaluations"
                "</span>"
            ),
            "x": 0.055,
            "xanchor": "left",
            "y": 0.975,
            "yanchor": "top",
            "font": {"size": 34, "color": "#F4F8FC"},
        },
        paper_bgcolor="#071426",
        plot_bgcolor="#0B1B2E",
        font={"family": "Arial", "color": "#DCE8F2", "size": 17},
        margin={"l": 120, "r": 70, "t": 205, "b": 115},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.075,
            "yanchor": "bottom",
            "font": {"size": 16},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hovermode="x unified",
    )
    figure.update_annotations(font={"size": 22, "color": "#F4F8FC"})
    figure.update_xaxes(
        gridcolor="#263E53",
        zeroline=False,
        showline=True,
        linecolor="#5B738A",
        row=1,
        col=1,
    )
    figure.update_xaxes(
        title_text="Total environment timesteps",
        tickformat=".2s",
        gridcolor="#263E53",
        zeroline=False,
        showline=True,
        linecolor="#5B738A",
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title_text="Mean reward",
        gridcolor="#263E53",
        zeroline=False,
        showline=True,
        linecolor="#5B738A",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Steps",
        gridcolor="#263E53",
        zeroline=False,
        showline=True,
        linecolor="#5B738A",
        row=2,
        col=1,
    )
    figure.add_annotation(
        x=0.0,
        y=-0.105,
        xref="paper",
        yref="paper",
        text=(
            "Source: TensorBoard eval/mean_reward and eval/mean_ep_length. "
            "Evaluation ranges are shaded; no smoothing applied."
        ),
        showarrow=False,
        xanchor="left",
        font={"size": 14, "color": "#8EA8BC"},
    )
    return figure


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the four V1.1 curriculum evaluation runs."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/v1_1_single_engine"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("gallery"))
    parser.add_argument("--output-name", default="v1_1_curriculum_evaluations")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1350)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--skip-png",
        action="store_true",
        help="Write only HTML and CSV when Plotly image export is unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    phases = [load_phase(args.logs_dir, phase) for phase in PHASES]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_stem = args.output_dir / args.output_name
    html_path = output_stem.with_suffix(".html")
    png_path = output_stem.with_suffix(".png")
    csv_path = output_stem.with_suffix(".csv")

    figure = build_figure(phases, args.width, args.height)
    figure.write_html(html_path, include_plotlyjs=True)
    write_csv(csv_path, phases)
    if not args.skip_png:
        figure.write_image(png_path, scale=args.scale)

    total_evaluations = sum(len(phase["reward"]) for phase in phases)
    print(f"Loaded {total_evaluations} evaluation checkpoints across {len(phases)} phases")
    if not args.skip_png:
        print(f"PNG:  {png_path.resolve()}")
    print(f"HTML: {html_path.resolve()}")
    print(f"CSV:  {csv_path.resolve()}")


if __name__ == "__main__":
    main()
