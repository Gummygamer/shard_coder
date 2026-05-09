"""Safe patch application."""

from .diff_utils import (
    DiffParseError,
    FileDiff,
    Hunk,
    PatchApplyError,
    apply_file_diff,
    apply_unified_diff,
    parse_unified_diff,
)
from .patcher import (
    NO_PATCH_PREFIX,
    PatchApplyResult,
    PatchValidationResult,
    apply_validated_patch,
    validate_model_output,
)

__all__ = [
    "DiffParseError",
    "FileDiff",
    "Hunk",
    "NO_PATCH_PREFIX",
    "PatchApplyError",
    "PatchApplyResult",
    "PatchValidationResult",
    "apply_file_diff",
    "apply_unified_diff",
    "apply_validated_patch",
    "parse_unified_diff",
    "validate_model_output",
]
