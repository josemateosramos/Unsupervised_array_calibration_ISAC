# Unsupervised End-to-End Array Calibration for Multi-Target ISAC

Code to reproduce the results of *J. M. Mateos-Ramos, B. Chatelier, L. Le Magoarou, N. Shlezinger, H. Wymeersch, C. Häger, "Unsupervised End-to-End Array Calibration for Multi-Target Integrated Sensing and Communication", IEEE Transactions on Wireless Communications, 2026*.

An ISAC base station with two impaired ULAs (gain-phase and antenna displacement impairments) is calibrated without labeled data: the steering vectors are parameterized by Ψ = [ξ, ω] and learned from the OMP residual (sensing) and the received energy at the UE (communication), with a gradient-free (GF) approximation of the channel gradient at the TX.

## Repository structure

```
lib/
  functions.py                Channels, precoders, OMP, losses, training loop (train), testing and
                              experiment helpers (file names, evaluation phases) used by methods/
  simulation_parameters.py    Table II parameters, true impairments Ξ, learnable Ψ, optimizers, CLI flags
  learnable_parameters.py     Ψ = [ξ_tx, ω_tx, ξ_rx, ω_rx]
methods/                      One script per experiment (see table below)
plotting/
  plot_beam_ad_map_isac.ipynb Figs. 1 and 2: beam pattern of the ISAC precoder and ADMs over the OMP iterations
```

Code names follow the paper: `gamma_x`/`p_x` true GPIs/positions (Ξ), `xi_x`/`omega_x` learned ones (Ψ), `f_s`, `f_c`, `f_tx` precoders, `Y_s`, `y_c` observations, `omega_r`, `eta_r` trade-offs, `sigma` precoder perturbation, suffixes `_s`/`_c` for sensing/communication.

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.0, NumPy (Matplotlib for the notebook).
- A CUDA GPU. Training is not practical on CPU with the parameters of Table II.

Seeds are fixed and `torch.use_deterministic_algorithms` is enabled, so results are reproducible for the same seed, hardware and software versions.

## Running

Run every script as a module from the repository root. Outputs are written to `results/` (`.npz`) and `models/` (learned Ψ), which must exist:

```bash
mkdir -p results models
python -m methods.gf_ul --refSensor 1 -s 3 --sensLoss residual -n <seed>
```

The plotting notebook `plotting/plot_beam_ad_map_isac.ipynb` reproduces Figs. 1 and 2 with and without impairment knowledge. It is self-contained (it does not import anything from this repository), runs on CPU, and saves the beam patterns of Fig. 1 (`beam_ideal.txt`, `beam_hwi.txt`) and the ADMs of Fig. 2 (`first_it_*.pdf`, `residual_*.pdf`) in its working directory. The batch samples shown are set by `idx_beam` and `idx_adm`.

Command-line flags (defaults in brackets):

| Flag | Meaning |
|---|---|
| `-i` [1] | 1: no impairment knowledge (Ψ initialized to the ideal ULA). 0: Ψ = Ξ |
| `-s` [0] | Index of σ in `sigma_list` = λ·{5, 0.01, 2, 10, 50, 100, 199} = {0.025, 5e-5, 0.01, **0.05**, 0.25, 0.5, 0.995} |
| `-n` [10] | Seed of the impairment realization |
| `--sensLoss` [adm] | GF UL sensing loss: `adm` (eq. 23) or `residual` (eq. 24) |
| `--ompIt` [1] | OMP iterations *I* of the residual loss |
| `--refSensor` [0] | 1: first element of each ULA is a known reference sensor (eq. 35) |
| `--etaFixed` [–] | `gf_ul` only: fixed η_r (0.0: only communication, 1.0: only sensing). If not given, η_r = ω_r ~ U[0, 1] |
| `--knownTx` [0] | `gf_ul` only: 1: the TX impairments are known (Ψ_tx = Ξ_tx) and only Ψ_rx is learned |
| `--clutterVar` [–] | `gf_ul` only: clutter variance σ_c² of the training sensing channel (up to 5 scatterers). If not given, training is clutter-free |

**Paper settings:** `--refSensor 1 -s 3 --sensLoss residual --ompIt 1` for every GF UL run, `--refSensor 1 -s 3` for `slcb_pert` (same precoder perturbation σ as GF UL) and `--refSensor 1` for SLCB and baselines, repeated for five impairment realizations (`-n`). Model-based baselines are run with `-i 1` (no impairment knowledge) and `-i 0` (impairment knowledge). Settings specific to the other experiments are constants at the top of each script, edited by hand.

## Reproducing the figures

| Fig. | Scripts (settings) | Saved results |
|---|---|---|
| 1, 2 | `plotting/plot_beam_ad_map_isac.ipynb` (ISAC precoder with ω_r = 0.75, P = 0.1 W) | `beam_ideal.txt`, `beam_hwi.txt`, `first_it_*.pdf`, `residual_*.pdf` |
| 4 | `gf_ul --etaFixed 1.0 --knownTx 1` with `--sensLoss adm`, and with `--sensLoss residual --ompIt 1` / `--ompIt 5`; `baseline_sensing_known_tx` | `p_d_roc`, `p_fa_roc`, `gospa_roc` |
| 5 | `gf_ul --clutterVar <σ_c²>` (sweep σ_c²; `--ompIt 1/3/6`); `baseline_sensing` | `scnr_s`, `gospa_roc` / `gospa` |
| 6 | `gf_ul --etaFixed 0.0` and `--etaFixed 1.0`, `gf_ul`, `slcb`, `baseline_isac` | `p_d_isac`, `gospa_isac`, `ser_isac` |
| 7 | `gf_ul`, `gf_ul_alternate` (`tx_first = True/False`); 2500 it.: `train_it = 2500` in `simulation_parameters.py` | as Fig. 6 |
| 8 | Learned `xi_tx`, `omega_tx` of `gf_ul` | `xi_tx`, `omega_tx` |
| 9 | `gf_ul -s 0..6`; `baseline_comm` | `ser_sweep`, `ser_isac` |
| 10 | `gf_ul_sector_pert` (`sigma_theta` in rad); `baseline_sensing`, `baseline_comm` | `gospa_isac`, `ser_isac` |
| 11 | `gf_ul_train_sector` (`phi_deg`; `delta_theta_deg = (10, 20)` or `(5, 5)`); `baseline_sensing` | `gospa_roc` |
| 12 | Set `P = 1e-4` in `simulation_parameters.py` (training SNR −33 dB), run `gf_ul`, `slcb`, `slcb_pert`, then `load_gf_ul_snr`, `load_slcb_snr`, `load_slcb_pert_snr` with the same flags and edit; `baseline_snr` | `p_d_snr`, `P_list` |

Every trained method saves `omega_r`, `eta_r`, the training losses `loss_r`, `loss_c`, the training SCNR `scnr_s` [dB], the true (`gamma_*`, `p_*`) and learned (`xi_*`, `omega_*`) impairments, and the test results: sensing ROC (`*_roc`), communication SNR/SER for increasing UE range `R_max_c_sweep` (`snr_c_sweep`, `ser_sweep`) and the ISAC trade-off over ω_r (`*_isac`, at p_fa = 10⁻²). Sensing metrics are the detection probability `p_d` (= 1 − p_md), `p_fa`, `gospa` and the mean position error `pos_err`. `gf_ul --clutterVar` also saves `C_max` and `sigma_c2`.
