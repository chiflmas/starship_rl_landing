"""Concatenate evaluation activation videos and export MP4 plus GIF.

Example:
    python combine_evaluation_episodes.py --speed 2
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg


EPISODE_NUMBER = re.compile(r"episode_(\d+)", re.IGNORECASE)


def episode_sort_key(path):
    match = EPISODE_NUMBER.search(path.stem)
    return (int(match.group(1)) if match else float("inf"), path.name.lower())


def find_episode_videos(input_dir, pattern):
    videos = sorted(input_dir.glob(pattern), key=episode_sort_key)
    if not videos:
        raise FileNotFoundError(
            f"No se encontraron videos con el patron {pattern!r} en {input_dir}"
        )
    return videos


def write_concat_manifest(path, videos):
    lines = []
    for video in videos:
        # FFmpeg concat accepts forward-slash absolute paths on Windows.
        escaped_path = video.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped_path}'\n")
    path.write_text("".join(lines), encoding="utf-8")


def run_ffmpeg(command):
    print("\nEjecutando FFmpeg:")
    print(" ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def combine_videos(
    videos,
    mp4_path,
    gif_path,
    speed,
    preserve_frames,
    mp4_fps,
    gif_fps,
    gif_width,
    overwrite,
):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    overwrite_flag = "-y" if overwrite else "-n"

    with tempfile.TemporaryDirectory(prefix="starship_concat_") as temp_dir:
        manifest = Path(temp_dir) / "episodes.txt"
        write_concat_manifest(manifest, videos)

        # setpts=PTS/2 means 2x. When preserving frames, FFmpeg raises the
        # output frame rate instead of dropping frames to keep a fixed FPS.
        video_filter = f"setpts=PTS/{speed:.8f}"
        if preserve_frames:
            capture = cv2.VideoCapture(str(videos[0]))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
            if source_fps <= 0.0:
                raise RuntimeError(f"No se pudo leer el FPS de {videos[0]}")
            target_fps = source_fps * speed
            video_filter += f",fps={target_fps:.8f}"
            print(
                f"Conservacion de frames: {source_fps:g} FPS -> "
                f"{target_fps:g} FPS"
            )
        else:
            video_filter += f",fps={mp4_fps}"
        mp4_command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "warning",
            "-stats",
            overwrite_flag,
            "-f", "concat",
            "-safe", "0",
            "-i", str(manifest),
            "-vf", video_filter,
            "-an",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(mp4_path),
        ]
        run_ffmpeg(mp4_command)

    # The MP4 is already accelerated. Build a palette inside one filter graph
    # so the GIF keeps useful colors without an intermediate palette file.
    gif_fps_filter = "" if preserve_frames else f"fps={gif_fps},"
    gif_filter = (
        gif_fps_filter
        + f"scale={gif_width}:-2:flags=lanczos,"
        "split[gif_a][gif_b];"
        "[gif_a]palettegen=max_colors=192[palette];"
        "[gif_b][palette]paletteuse=dither=sierra2_4a"
    )
    gif_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "warning",
        "-stats",
        overwrite_flag,
        "-i", str(mp4_path),
        "-filter_complex", gif_filter,
        "-loop", "0",
        str(gif_path),
    ]
    run_ffmpeg(gif_command)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Une los episode_*_combined.mp4 de evaluation y genera MP4 y GIF."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("videos/evaluation"),
        help="Directorio de entrada (por defecto: videos/evaluation)",
    )
    parser.add_argument(
        "--pattern",
        default="episode_*_combined.mp4",
        help="Patron de archivos de entrada",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("videos/evaluation/evaluation_episodes"),
        help="Ruta base de salida, sin extension",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Factor de velocidad para MP4 y GIF: 2=doble, 0.5=mitad",
    )
    parser.add_argument(
        "--preserve-frames",
        action="store_true",
        help=(
            "Conservar todos los frames elevando los FPS segun --speed; "
            "aumenta especialmente el tamano del GIF"
        ),
    )
    parser.add_argument(
        "--mp4-fps",
        type=int,
        default=20,
        help="FPS del MP4 final (por defecto: 20)",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="FPS del GIF final (por defecto: 10)",
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=1200,
        help="Anchura del GIF; la altura conserva proporcion (por defecto: 1200)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir las salidas si ya existen",
    )
    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed debe ser mayor que cero")
    if args.mp4_fps <= 0:
        parser.error("--mp4-fps debe ser mayor que cero")
    if args.gif_fps <= 0:
        parser.error("--gif-fps debe ser mayor que cero")
    if args.gif_width <= 0 or args.gif_width % 2:
        parser.error("--gif-width debe ser un numero par mayor que cero")
    return args


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    videos = find_episode_videos(input_dir, args.pattern)
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    mp4_path = output_prefix.with_suffix(".mp4")
    gif_path = output_prefix.with_suffix(".gif")

    print(f"Videos encontrados: {len(videos)}")
    for index, video in enumerate(videos, start=1):
        print(f"  {index:02d}. {video.name}")
    print(f"Velocidad final: {args.speed:g}x")

    combine_videos(
        videos=videos,
        mp4_path=mp4_path,
        gif_path=gif_path,
        speed=args.speed,
        preserve_frames=args.preserve_frames,
        mp4_fps=args.mp4_fps,
        gif_fps=args.gif_fps,
        gif_width=args.gif_width,
        overwrite=args.overwrite,
    )

    print("\nArchivos generados:")
    print(f"  MP4: {mp4_path}")
    print(f"  GIF: {gif_path}")


if __name__ == "__main__":
    main()
