# FEA results

Raw output from Ansys runs.

Everything else the repo ignores is derived, but these are measurements produced by Ansys software on this machine, and re-running the deck overwrites them.

| file | run | notes |
|---|---|---|
| `s0_2_spike_sma.txt` | S0.2-spike, 2026-08-19 | Ansys Student 2026 R1, Mechanical APDL. Single SOLID185 cube, 5 % tension load–unload. PRVAR table: time, UZ, SZ. Parse with `sim3d.apdl.read_prvar()`. |
| `s5_1_spikeA_sector_shpp.txt` | S5.1, 2026-08-19 | `SHPP,SUMM` mesh-quality report, 1/12 sector. |
| `s5_1_spikeA_fullring_shpp.txt` | S5.1, 2026-08-19 | `SHPP,SUMM` mesh-quality report, full ring (48,768 elem / 73,020 nodes; fits Ansys Student). |
| `s5_2_bending.txt` | S5.2, 2026-08-19 | Cantilever vs Euler-Bernoulli. Confirms SOLID185 enhanced strain doesn't shear-lock. |
| `s5_2_model.txt` | S5.2, 2026-08-19 | First elastic solve on the real sector mesh. |
| `s5_3_loadsteps.txt` | S5.3 (Part 1), 2026-08-20 | Four converged load steps. Raw: NOUTER, ZMAX, DAXIAL, FRADIAL, EPS3, EPS4, EPSAMP. Derived metrics in `sim3d/loadsteps.py`. |
| `s5_3_loadsteps_hist.txt` | S5.3 (Part 1), 2026-08-20 | POST26 time history, all four load steps. **Fixed-width**, parse with `loadsteps.read_prvar_history()`. |
| `s5_3_layers{1..4}.txt` | S5.3 (Part 2), 2026-08-20 | Through-thickness layer sweep. Self-contact pushes the full ring past Student's element ceiling at 4 layers, so this measures what fewer layers cost. Reports NELEM, UZTIP, ANALYTIC, RATIO. Measures tip **deflection** only, a global quantity. Says nothing about local surface-strain sampling. |
| `s5_3_strain{1..4}.txt` | S5.3 (Part 2), 2026-08-21 | Strain-**sampling** verification at mid-span vs `M*c/(E*I)`. Centroidal `ETABLE` undersamples by `(1-1/n)` - 0.0195 / 0.511 / 0.681 / 0.767 at n = 1/2/3/4. **Nodal sampling is exact (1.000000) at every layer count, including n = 1.** This is why `s5_3_loadsteps` must move off `ETABLE` for strain. |
