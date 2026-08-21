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
| `s5_3_crimp_post.txt` | S5.3 (Part 2), 2026-08-21 | Crimp run that FAILED at LS1 substep 32, read back with the POST1-only deck. One data set (the unconverged debug set), so EPS_MAX 0.176 is a diverging value, not a measurement. What matters: **P_CRIMPER = 486.3 MPa proves the crimper contact carried load** (TARGE170 prescribed-motion crimper, CONTA173 orientation and FKN all correct), while **P_SELF = 0 shows self-contact never engaged** - the mesh failed at 4.88 mm outer diameter, before struts came close enough to touch. |
| `s5_3_loadsteps_amp.txt` | S5.4, 2026-08-21 | Per-node strain RANGE between LS3 and LS4, one value per line by APDL node number (12,385 = the sector mesh node count). Python halves it to an amplitude. Cross-check: the field's own max reproduces the deck's EPSAMP to 8 digits, which is what proves the right array was exported. |
| `s5_4_scalars.json` | S5.4, 2026-08-21 | **DERIVED, not raw Ansys output.** The three 3D scalars for the reference cell: `K_radial_3D` 27.128 N/mm3, `eps_a_max_3D` 0.034046, `A_over_lim_3D` 0.18317. Regenerate with `python -c "from sim3d import loadsteps; loadsteps.extract_scalars()"`. |
