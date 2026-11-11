# Shared DC-bus braking experiments

Reproduction code for **Coordinated Braking of Shared DC-Bus PMSM Drives With Energy Scheduling and Safety Filtering**.

Three surface-mounted permanent-magnet synchronous motors share an isolated DC bus. The simulation integrates their electrical and mechanical dynamics, capacitor energy, winding losses and external absorption in Python. No external simulator or network service is needed after installation.

## Installation

Tested on Linux with Python 3.11. Install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

Check the installation with 34 short experiments covering all eight configurations and both ablations:

```bash
python reproduce.py --stage smoke --workers 2
```

Run the complete study:

```bash
python reproduce.py --workers 4
```

The full run performs 3,816 closed-loop experiments, numerical checks, serial timing measurements, schedule regeneration and plotting. Results are written to `results/`. Numerical kernels compile on first use. Use `--workers 1` if memory is limited.

Stages can also be run separately:

```bash
python reproduce.py --stage experiments --workers 4
python reproduce.py --stage checks
python reproduce.py --stage timing
python reproduce.py --stage planning
python reproduce.py --stage plots
```

The `checks` and `plots` stages require the complete experiment outputs. Use `--output another_directory` to keep a separate run, and supply that directory to subsequent stages. Timing runs are serial; avoid other computational jobs during those measurements.

## Experiments

| Folder | Runs | Conditions |
|---|---:|---|
| `friction` | 576 | Three friction profiles, 24 state/absorption combinations, eight configurations |
| `parameters` | 1,152 | Six parameter perturbations, 24 combinations, eight configurations |
| `noisy` | 1,152 | The same perturbations with errors in identification readings |
| `newstates` | 768 | Six perturbations, 16 additional combinations, eight configurations |
| `auxiliary` | 24 | Stationary-axis dissipation removed |
| `no_identification` | 144 | Electrical-parameter updates removed |

| Output name | Allocation | Current control | Execution filter |
|---|---|---|---|
| `conic` | Proposed schedule | Sampled regulator | Proposed |
| `dual` | Dual AOV | Sampled regulator | Proposed |
| `mcl` | Shared MCL | Sampled regulator | Proposed |
| `cvo` | CVO | Sampled regulator | Proposed |
| `mcl_pdo` | Shared MCL | PDO-SMPCC | Proposed |
| `hocbf` | Proposed schedule | Sampled regulator | HOCBF |
| `pdo_smpcc` | Proposed schedule | PDO-SMPCC | Proposed |
| `dpcc` | Proposed schedule | Conventional DPCC | Proposed |

The first five configurations compare braking allocation. The final three replace one execution component under the same schedule. Conventional DPCC is a diagnostic reference. MCL/PDO combines two published components for the shared-bus plant.

All comparisons use a 50-microsecond voltage-update period, a 10-microsecond RK4 plant step and supervisory holds of at most 1 millisecond. Current references use 10% headroom for the first 5 milliseconds and 5% thereafter. Voltage commands have 1% radial headroom. The refinement check halves the plant and identification step while retaining both control periods.

The PDO-SMPCC controller uses nominal inductance and its own disturbance observer. The other current controllers use the common electrical estimates. PDO references satisfy `dr/dt = (command - r)/0.001`; the reference and its derivative enter the published law. Observer initialization uses nominal electrical dynamics at the measured initial state. Selected gains appear in `config/baseline_gains.json`.

Identification-noise tests perturb estimator readings only. Local current feedback and supervisory measurements remain exact. The continuous-feedback boundary checks test the analytical certificate separately; physical-limit checks in the sampled experiments use the integrated plant trajectories.

## Outputs

- `summary.csv`: completion counts, mean cost and travel, and physical-limit measurements by condition and configuration.
- `paired_comparisons.json`: paired cost reductions, success counts, excluded incomplete pairs and descriptive intervals clustered by initial state.
- Experiment folders: metrics for every run and compressed trajectories for the first and last main/additional states. Arrays contain plant samples, supervisory references and electrical estimates.
- `figures/`: six PDF time-history figures covering braking, interaxis allocation, noisy identification, additional states, auxiliary-axis ablation and current-loop substitutions. Curves use saved samples without smoothing or interpolation.
- `timing/`: 192 serial runs, supervisory update samples, and separate inner-loop timing summaries.
- `planning_timing.json` and `schedules.json`: planning measurements and regenerated schedules.
- `mathematical_checks.json`, `compiled_check.json`, `implementation_checks.json`, `certificate_check.json` and `refinement.json`: numerical identities, published-law algebra, filter implementation checks, boundary checks and 48 step-refinement runs.

Incomplete trajectories remain in the metric records. Means use completed trajectories; paired reductions use the same completed cases for both configurations. All runs contribute to success counts. HOCBF and parameter-correction ablations can terminate when their constraint checks reject a command. Removing auxiliary dissipation can leave a physically safe trajectory that misses the stopping deadline. These outcomes are part of the experiment, not installation errors.

## Schedules

`config/` contains initial states, selected gains and the numerical schedules used for the comparisons. The controller uses reactive braking for the first 25 milliseconds, then activates the stored schedule. The schedule is available before the stop. Its activation time is separate from the planning computation time.

The scheduling algorithm is included. To evaluate newly generated schedules:

```bash
python reproduce.py --stage planning
python reproduce.py --stage experiments --schedules results/schedules.json --output regenerated_results
python reproduce.py --stage checks --schedules results/schedules.json --output regenerated_results
python reproduce.py --stage plots --output regenerated_results
```

## Comparator sources

These are independent implementations adapted to the shared-bus plant, not the original authors' software.

- Dual AOV: Ding et al., *IEEE Transactions on Power Electronics*, 2020. [DOI](https://doi.org/10.1109/TPEL.2019.2932030).
- Shared MCL: Marcetic and Matic, *IEEE Transactions on Industrial Electronics*, 2020. [DOI](https://doi.org/10.1109/TIE.2019.2946569).
- CVO: Ding et al., *IEEE Transactions on Industrial Electronics*, 2020. [DOI](https://doi.org/10.1109/TIE.2019.2918497).
- PDO-SMPCC: He et al., *IEEE Transactions on Industrial Electronics*, 72(2), 1292–1302, 2025. [DOI](https://doi.org/10.1109/TIE.2024.3423357).
- HOCBF: Xiao and Belta, *IEEE Transactions on Automatic Control*, 2022. [DOI](https://doi.org/10.1109/TAC.2021.3105491); Ames et al., *IEEE Transactions on Automatic Control*, 2017. [DOI](https://doi.org/10.1109/TAC.2016.2638961).

Shared MCL uses common winding-loss allowances and shared-bus power balance. CVO uses completed-hold current-vector projection in place of the source phase-locked loop. HOCBF imposes hard energy, current, copper-energy and inverter-voltage constraints.

The PDO-SMPCC implementation follows source Eqs. (16), (19), (38), (43) and (44). It implements the published variant without the additional harmonic disturbance observer; the averaged plant contains no inverter dead-time harmonics. Its selected gains were obtained from 12 candidates on three independent stiff-bus tracking conditions and retained across the braking tests. `src/sampled_drive.py` contains the voltage-driven plant and current laws; `scripts/check_sampled.py` independently checks their algebra and energy balance.
