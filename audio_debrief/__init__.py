"""audio_debrief — local diarized transcript + delivery metrics for debriefing recorded conversations.

Nothing here touches the network except the first-time model downloads from
HuggingFace (faster-whisper large-v3, pyannote diarization + embedding models).
Audio never leaves the machine.
"""
import importlib.util
import os
import sys

__version__ = "0.1.0"

_dll_handles = []


def _add_torch_dlls() -> None:
    """Let ctranslate2 (faster-whisper) find the cuBLAS/cuDNN DLLs bundled with torch.

    Done via importlib metadata instead of `import torch` so importing this
    package stays cheap for the pure-Python analysis modules and the tests.
    """
    if sys.platform != "win32":
        return
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return
    lib = os.path.join(os.path.dirname(spec.origin), "lib")
    if os.path.isdir(lib):
        _dll_handles.append(os.add_dll_directory(lib))
        os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")


_add_torch_dlls()
