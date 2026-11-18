"""Typed exceptions for the clipping pipeline.

Every error raised by the application derives from ``ClipperError`` so the CLI
can print a single actionable message instead of a stack trace.
"""

from __future__ import annotations


class ClipperError(Exception):
    """Base class for all application errors."""

    #: Short, human readable hint on how to fix the problem.
    hint: str = ""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        if hint:
            self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.hint:
            return f"{self.message}\n  -> {self.hint}"
        return self.message


class ConfigError(ClipperError):
    """Invalid or missing configuration."""


class DependencyError(ClipperError):
    """A required external dependency is missing or unusable."""


class DownloadError(ClipperError):
    """The source video could not be downloaded."""


class TranscriptionError(ClipperError):
    """Speech-to-text failed."""


class LLMError(ClipperError):
    """Ollama/Qwen returned an unusable response."""


class SelectionError(ClipperError):
    """No usable clip could be selected from the transcript."""


class RenderError(ClipperError):
    """FFmpeg failed while producing an output file."""
