# Run specs

Parameter files for `python -m kernels.gpu multi --long --final --params "$(cat runs/<file>)"`.
`final_a` + `final_b` are the M6 final runs, done once on L4 and once on H100 (`--gpu H100`).
Sweeps are resumable: the last step of `final_b` finishes any decode cases `final_a` did not reach.
