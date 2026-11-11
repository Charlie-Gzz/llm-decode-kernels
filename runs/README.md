# Run specs

Parameter files for `python -m kernels.gpu multi --long --final --params "$(cat runs/<file>)"`.

- `final_a`, `final_b`: M6 final runs on L4. `final_b` reruns the Triton and CUDA decode sweeps as
  `final2` after the path-picker fix (see DECISIONS).
- `h100_a`, `h100_b`: the single H100 session (`--gpu H100`), each under the 30 min timeout,
  1.5 h cap in total. The last step of `h100_b` finishes any decode cases `h100_a` did not reach.

Sweeps are resumable through checkpoints on the Modal Volume.
