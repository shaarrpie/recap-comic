"""
cli_cinematic_patch.py  –  Add `guided cinematic` command to your CLI
====================================================================
Run once to patch your cli.py, OR import and call add_cinematic_command(app)
before app() in cli.py.

Quick integration:
    # At the bottom of cli.py, before `if __name__ == "__main__": app()`
    from cli_cinematic_patch import add_cinematic_command
    add_cinematic_command(guided_app)

Then use:
    python cli.py guided cinematic panels.json --style dynamic
    python cli.py guided cinematic panels.json --style dynamic --bgm bgm.mp3
"""
from __future__ import annotations

from pathlib import Path

import typer


def add_cinematic_command(guided_app: typer.Typer) -> None:
    """Register the `cinematic` sub-command onto the given guided_app Typer."""

    @guided_app.command("cinematic")
    def guided_cinematic(
        panels: Path = typer.Argument(  # noqa: B008 (typer idiom)
            ..., exists=True, dir_okay=False,
            help="panels.json written by 'guided run' / 'guided cut'",
        ),
        out: Path | None = typer.Option(  # noqa: B008
            None, "--out",
            help="output mp4 (default: <panels dir>/cinematic_recap.mp4)",
        ),
        audio_dir: Path | None = typer.Option(  # noqa: B008
            None, "--audio-dir",
            help="directory with per-panel audio mp3/wav (default: panels dir/audio)",
        ),
        style: str = typer.Option(
            "dynamic", "--style",
            help="dynamic (full manhwa-recap) | subtle (light effects)",
        ),
        bgm: Path | None = typer.Option(  # noqa: B008
            None, "--bgm",
            help="optional background music mp3 mixed under narration",
        ),
        bgm_volume: float = typer.Option(
            0.18, "--bgm-volume",
            help="background music volume relative to narration (0-1)",
        ),
        letterbox: bool = typer.Option(
            False, "--letterbox/--no-letterbox",
            help="add cinematic letterbox bars",
        ),
        no_glitch: bool = typer.Option(
            False, "--no-glitch",
            help="disable glitch transitions between panels",
        ),
        no_shake: bool = typer.Option(
            False, "--no-shake",
            help="disable screen shake on action panels",
        ),
        no_speedlines: bool = typer.Option(
            False, "--no-speedlines",
            help="disable speed-lines overlay on action panels",
        ),
        no_vignette: bool = typer.Option(
            False, "--no-vignette",
            help="disable vignette overlay",
        ),
        fps: int = typer.Option(30, "--fps"),
        ffmpeg_exe: str = typer.Option("ffmpeg", "--ffmpeg"),
        log_level: str = typer.Option("INFO", "--log-level"),
    ) -> None:
        """Phase 3 (cinematic): panels.json -> dynamic manhwa-style recap.mp4

        Applies punch-zoom, Ken-Burns with zoom, screen shake, glitch
        transitions, vignette and color grade instead of the plain pan.
        """
        import logging

        from adapters._logging import setup_logging
        setup_logging(level=log_level)
        logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))

        try:
            from cinematic_effects import CinematicConfig, make_cinematic_video
        except ImportError:
            typer.echo(
                "ERROR: cinematic_effects.py not found. "
                "Copy cinematic_effects.py into the project root.", err=True
            )
            raise typer.Exit(1) from None

        if style not in ("dynamic", "subtle"):
            typer.echo(f"ERROR: --style must be 'dynamic' or 'subtle', got {style!r}",
                       err=True)
            raise typer.Exit(1)

        out_path = out or panels.parent / "cinematic_recap.mp4"

        cfg = CinematicConfig(
            style=style,  # type: ignore[arg-type]
            fps=fps,
            bgm_path=str(bgm) if bgm else None,
            bgm_volume=bgm_volume,
            letterbox_enabled=letterbox,
            glitch_enabled=not no_glitch,
            shake_enabled=not no_shake,
            speedlines_enabled=not no_speedlines,
            vignette_enabled=not no_vignette,
            ffmpeg_exe=ffmpeg_exe,
        )

        try:
            summary = make_cinematic_video(
                panels_json=panels,
                out_mp4=out_path,
                audio_dir=audio_dir,
                cfg=cfg,
            )
        except Exception as exc:
            typer.echo(f"ERROR: {exc}", err=True)
            raise typer.Exit(1) from exc

        mins, secs = divmod(summary["duration_s"], 60)
        typer.echo(f"panels : {summary['panels']}  clips : {summary['clips']}  "
                   f"length : {int(mins)}m{secs:04.1f}s")
        typer.echo(f"video  : {summary['out']}")
