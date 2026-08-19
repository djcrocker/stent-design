# FEA results

Raw output from Ansys runs.

Everything else the repo ignores is derived, but these are measurements produced by Ansys software on this machine, and re-running the deck overwrites them.

| file | run | notes |
|---|---|---|
| `s0_2_spike_sma.txt` | S0.2-spike, 2026-08-19 | Ansys Student 2026 R1, Mechanical APDL. Single SOLID185 cube, 5 % tension load–unload. PRVAR table: time, UZ, SZ. Parse with `sim3d.apdl.read_prvar()`. |
