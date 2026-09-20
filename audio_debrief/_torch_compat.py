"""Compatibility shim: torch>=2.6 defaults torch.load(weights_only=True), which
rejects pyannote 3.x checkpoints (they pickle TorchVersion and other globals).
The pyannote models come from HuggingFace repos whose terms you must accept, so
loading them with weights_only=False is the intended trust boundary.
Call `patch_torch_load()` before any pyannote Pipeline/Inference load.
"""
import functools

_patched = False


def patch_torch_load() -> None:
    global _patched
    if _patched:
        return
    import torch

    original = torch.load

    @functools.wraps(original)
    def _load(*args, **kwargs):
        # Forced, not defaulted: lightning passes weights_only=True explicitly.
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = _load
    _patched = True
