"""Command-line entry point.

    dialogue-locator <url-or-file> "<dialogue>" [options]

Prints the result in the format requested by the problem statement::

    Timestamp : 00:05:25.312
    Frame     : 7799
    Text      : "My mind rebels at stagnation."
    Image     : data/output/<job>/frame.jpg

Exit codes: 0 found, 2 not found (near-misses printed), 3 found in the audio but
not onscreen (no face in the frame, or a face whose mouth never moves during the
line; details are still printed), 1 error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dialogue_locator.config import Settings, get_settings
from dialogue_locator.exceptions import DialogueLocatorError
from dialogue_locator.logging_config import configure_logging
from dialogue_locator.models import LocalizationResult, ProgressEvent, ResultStatus
from dialogue_locator.pipeline import DialoguePipeline, PipelineRequest, save_result
from dialogue_locator.pipeline.pipeline import RESULT_FILENAME

EXIT_FOUND, EXIT_ERROR, EXIT_NOT_FOUND, EXIT_NOT_ONSCREEN = 0, 1, 2, 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dialogue-locator", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="video URL (YouTube, ok.ru, ...) or local file")
    p.add_argument("dialogue", help="the spoken dialogue to find")
    p.add_argument("--fast-model", help="Whisper model for the streaming pass (default from config)")
    p.add_argument("--verify-model", help="Whisper model for verification (default from config)")
    p.add_argument("--no-verify", action="store_true", help="skip the large-model verification pass")
    p.add_argument("--threshold", type=float, help="fuzzy match threshold 0-100")
    p.add_argument("--max-height", type=int, help="max height for the final clip download (the search uses audio only)")
    p.add_argument("--output-dir", type=Path, help="where to write frame + result.json")
    p.add_argument("--work-dir", type=Path, help="where to cache downloads/audio")
    p.add_argument("--no-cache", action="store_true", help="ignore previously downloaded media")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Return a copy of ``settings`` with CLI flags applied (config stays the single source of truth)."""
    update: dict = {}
    whisper: dict = {}
    if args.fast_model:
        whisper["fast_model"] = args.fast_model
    if args.verify_model:
        whisper["verify_model"] = args.verify_model
    if whisper:
        update["whisper"] = settings.whisper.model_copy(update=whisper)
    if args.no_verify:
        update["verification"] = settings.verification.model_copy(update={"enabled": False})
    if args.threshold is not None:
        update["matching"] = settings.matching.model_copy(update={"match_threshold": args.threshold})
    if args.max_height:
        update["download"] = settings.download.model_copy(update={"max_height": args.max_height})
    storage: dict = {}
    if args.output_dir:
        storage["output_dir"] = args.output_dir
    if args.work_dir:
        storage["work_dir"] = args.work_dir
    if storage:
        update["storage"] = settings.storage.model_copy(update=storage)
    if args.verbose:
        update["logging"] = settings.logging.model_copy(update={"level": "DEBUG"})
    return settings.model_copy(update=update) if update else settings


def format_result(result: LocalizationResult) -> str:
    if result.match is not None:  # localised: FOUND or NOT_ONSCREEN
        lines = []
        if result.status is ResultStatus.NOT_ONSCREEN:
            reason = (
                "face visible but mouth not moving"
                if result.face_present
                else "no face in the matched frame"
            )
            lines.append(f"Verdict   : Not an onscreen dialogue - {reason}")
        lines += [
            f"Timestamp : {result.timestamp}",
            f"Frame     : {result.frame_number}",
            f'Text      : "{result.matched_text}"',
            f"Image     : {result.frame.image_path if result.frame else '-'}",
            f"Score     : {result.match.score:.1f}",
        ]
        if result.face_present is True:
            faces = result.face_detection.faces
            lines.append(f"Face      : {len(faces)} detected (best {faces[0].confidence:.2f})")
        elif result.face_present is False:
            lines.append("Face      : none detected")
        if result.mouth_movement is not None:
            mm = result.mouth_movement
            if mm.moving is None:
                # No verdict on the lips because nothing held up as a face - say
                # that, rather than "indeterminate" next to a definite result.
                state = f"no face to judge ({mm.frames_with_face}/{mm.frames_analyzed} frames)"
                score = ""
            else:
                state = "moving" if mm.moving else "not moving"
                score = "" if mm.movement_score is None else f" (score {mm.movement_score:.3f})"
            lines.append(f"Mouth     : {state}{score}")
        for v in result.verifications:
            lines.append(f"Verify    : {v.verifier} -> {v.status.value}" + (f" ({v.score:.1f})" if v.score is not None else ""))
    else:
        lines = [f'Not found : "{result.dialogue}"', "Closest windows:"]
        for c in result.near_misses:
            lines.append(f"  {c.score:5.1f}  {c.timestamp}  \"{c.matched_text}\"")
        if not result.near_misses:
            lines.append("  (no speech transcribed)")
    for w in result.warnings:
        lines.append(f"Warning   : {w}")
    if result.transcribed_seconds is not None:
        lines.append(f"Scanned   : {result.transcribed_seconds:.1f}s of audio")
    return "\n".join(line for line in lines if line)


def _progress_printer(event: ProgressEvent) -> None:
    pct = "" if event.fraction is None else f" {event.fraction:4.0%}"
    print(f"  [{event.stage.value:>14}]{pct}  {event.message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None, pipeline_factory=DialoguePipeline) -> int:
    args = build_parser().parse_args(argv)
    settings = apply_overrides(get_settings(), args)
    configure_logging(settings.logging)
    settings.ensure_directories()

    request = PipelineRequest(source=args.source, dialogue=args.dialogue, reuse_cached_media=not args.no_cache)
    pipeline = pipeline_factory(settings)
    try:
        result = pipeline.run(request, progress=None if args.quiet else _progress_printer)
    except DialogueLocatorError as exc:
        print(f"ERROR [{exc.stage}]: {exc.message}", file=sys.stderr)
        if args.json:
            print(json.dumps(exc.to_dict(), indent=2))
        return EXIT_ERROR

    out_path = settings.storage.output_dir / request.job_id / RESULT_FILENAME
    save_result(result, out_path)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_result(result))
        print(f"Result    : {out_path}")
    if result.found:
        return EXIT_FOUND
    return EXIT_NOT_ONSCREEN if result.status is ResultStatus.NOT_ONSCREEN else EXIT_NOT_FOUND


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
