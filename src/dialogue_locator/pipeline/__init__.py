"""Pipeline orchestration: wires the stages together, independent of any UI."""

from dialogue_locator.pipeline.pipeline import DialoguePipeline, PipelineRequest, save_result

__all__ = ["DialoguePipeline", "PipelineRequest", "save_result"]
