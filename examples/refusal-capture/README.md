# refusal-capture

Relocated from the llama.cpp `ds4-flash-experiments` branch (2026-07-16 cleanup)
to keep the serving/perf branch's delta to just the flags + best setup.

`refusal-capture.cpp` is a llama.cpp example that runs prompts and captures
activation tensors (env: `CAPTURE_PROMPTS`, `CAPTURE_OUT`, `CAPTURE_OUT_THINK`,
`CAPTURE_TENSOR`) for building the refusal control-vector direction.

To build: drop this dir into a llama.cpp checkout's `examples/` (e.g.
`llamacpp-iq3/examples/refusal-capture/`) and add `add_subdirectory(refusal-capture)`
to `examples/CMakeLists.txt`. The runtime CVEC ablation mechanism (projection
ablation in `llama-adapter.cpp`, application wiring in `models/deepseek4.cpp`)
remains in the main branch — only this capture tool moved.
