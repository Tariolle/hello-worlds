# Running on Dalia (IDRIS) — team cheat-sheet

Everything we learned bringing the EEG-JEPA up on Dalia, so nobody re-fights it.

## 0. TL;DR quickstart
```bash
# 1. connect — MUST be on the OpenGate wired network (4G/other wifi is firewalled)
ssh -i ~/.ssh/htw_dalia <user>@dalia.idris.fr      # Windows: read §1 first

# 2. get the code into $WORK  ($WORK = /lustre/work/vivatech-helloworlds/$USER)
mkdir -p $WORK/logs && cd $WORK
git clone <repo>  hello-worlds                     # HTTPS + PAT (SSH is blocked, see §7)
cd hello-worlds

# 3. one-time: build the aarch64 venv on a COMPUTE node (~10 min, torch+cu128)
sbatch cluster/build_venv.sbatch                   # wait for VENV_DONE in $WORK/logs/venv_*.out

# 4. smoke test a forward pass on a B200
sbatch cluster/smoke.sbatch                         # check $WORK/logs/smoke_*.out -> SMOKE_DONE

# monitor anything
squeue -u $USER ;  sacct -j <id> --format=JobID,State,ExitCode ;  cat $WORK/logs/<job>.out
```

## 1. SSH connection — the Windows traps (cost us an hour)
- Dalia is reachable **only from the OpenGate network** (wired at your table). 4G / hotspot / other wifi → silently rejected, no error.
- Host fingerprint (ed25519): `SHA256:ZBphkLNp7aeZemAKK9/BjT47Ii2UnigG9hqQDmieczo`.
- **Linux/macOS:** `chmod 600 <key>` then `ssh -i <key> <user>@dalia.idris.fr`.
- **Windows — two gotchas:**
  1. The emailed key file has **no trailing newline** → OpenSSH rejects it (`error in libcrypto` / `invalid format`). Fix:
     `content=$(tr -d '\r' < key); printf '%s\n' "$content" > ~/.ssh/htw_dalia`
  2. **Git Bash's ssh (OpenSSL 3.1.2) cannot load ed25519 keys.** Use **Windows native OpenSSH** (`C:\Windows\System32\OpenSSH\ssh.exe`, LibreSSL). The key is valid — Python `cryptography` loads it fine; it's a client bug.

## 2. Cluster shape
- Login nodes (`dalia1/2`): **x86_64**. Compute nodes (`dalianvl01-18`): **aarch64 (Grace) + 4× B200 each** (`gpu:b200:4`) → 72 GPUs, partition `defq`, reservation `Vivatech`, 2-day max.
- Compiled wheels (torch…) are **not portable** across arches → build the venv **on a compute node** (so pip grabs aarch64 wheels), never on the login node.

## 3. Storage — never use $HOME
- `$HOME` (`/lustre/home/...`) = **3 GB only** → git/venvs/caches overflow it instantly.
- Put everything under **`$WORK = /lustre/work/vivatech-helloworlds/$USER`** (520 TB). Team scratch: `/lustre/work/vivatech-helloworlds/shared`.
- Redirect caches: `export PIP_CACHE_DIR=$WORK/.cache/pip XDG_CACHE_HOME=$WORK/.cache TORCH_HOME=$WORK/.cache/torch TRITON_CACHE_DIR=$WORK/.cache/triton`.

## 4. SLURM — the recipe that actually runs (the signal-53 fix)
Jobs that omit `--account` (or `--nodes/--ntasks`) **die instantly with `RaisedSignal:53(Real-time_signal_19)` and an empty log.** Always submit the full recipe:
```
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-helloworlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1        # GPU jobs only — use --gpus-per-node, NOT --gres=gpu:1
#SBATCH --time=HH:MM:SS
```
- **Never use `--mem`** — Dalia's scheduler rejects the job. Memory is auto-allocated ≈ `cpus-per-task × 10.7 GB` (16 cores ≈ 171 GB).
- **Don't use `--export=NONE`** — it strips the proxy env vars and breaks pip/git inside jobs.
- Monitor: `squeue -u $USER`, `sacct -j <id> --format=JobID,State,ExitCode,Reason`, logs in `$WORK/logs/` (SLURM buffers stdout — an empty log on a RUNNING job is normal).

## 5. The venv (torch + CUDA on B200)
`cluster/build_venv.sbatch` builds `$WORK/venvs/hw_aarch64` with **torch 2.11+cu128 (aarch64)** + numpy/scipy/scikit-learn/omegaconf/pyedflib/pyriemann/matplotlib/tqdm/wandb. Activate in any job:
```
source $WORK/venvs/hw_aarch64/bin/activate
```
Verified on a B200 (NVIDIA GB200): forward+backward of all 6 `reg_type × reg_space` cells, with the SPD-tangent `eigh`/logm on GPU (finite, no NaN).

## 6. Data — TUAB is already staged (no download)
`/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED`
- `train/{normal,abnormal}` = 2717 `.edf`, `eval/{normal,abnormal}` = 276 `.edf`, 19 ch @ 200 Hz.
- World-readable; it's already the `data.data_root` in `examples/eeg/cfgs/train.yaml`.

## 7. Internet — HTTP proxy only, SSH blocked
- Outbound is via `http://platprox.idris.fr:3128` (`http_proxy`/`https_proxy` set in the env). `wget` / `pip` / `git HTTPS` / `wandb` all work through it.
- **SSH outbound is blocked** (the proxy is HTTP-only) → `git@github.com` does **not** work, on port 22 or 443. Clone/pull over **HTTPS**. For a private repo, use a **fine-grained read-only PAT** (Contents: Read) stored via `git config --global credential.helper store`.
