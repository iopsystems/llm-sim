"""GPU-mock: boot the real vLLM V1 stack (EngineCore) GPU-free, in-process.

Registered as a vLLM out-of-tree platform plugin (entry point group
``vllm.platform_plugins`` in pyproject.toml). vLLM resolves platforms lazily;
on a GPU-less box with the CUDA wheel no builtin platform activates, so this
plugin is the sole winner — no monkeypatching anywhere.

See docs/journal/2026-07-09-gpu-mock-enginecore-boot.md for the spike record
and injection-surface inventory.
"""


def register() -> str | None:
    """vLLM platform-plugin entry point: activate MockPlatform unconditionally."""
    return "llm_sim.mock.platform.MockPlatform"
