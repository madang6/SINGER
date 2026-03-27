# SINGER: An Onboard Generalist Vision-Language Navigation Policy for Drones

## Installation

SINGER requires [FiGS-Standalone](https://github.com/StanfordMSL/FiGS-Standalone) as its base environment.

### Prerequisites

- NVIDIA GPU with drivers installed
- Docker Engine ([install guide](https://docs.docker.com/engine/install/ubuntu/))
- NVIDIA Container Toolkit (for GPU access inside containers):

```bash
# Add the NVIDIA container toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

- FiGS-Standalone repository (for the base Docker image)

### 1) Clone repositories

```bash
git clone https://github.com/StanfordMSL/FiGS-Standalone.git
git clone https://github.com/StanfordMSL/SINGER.git
```

### 2) Build the FiGS base image (one-time setup)

```bash
cd FiGS-Standalone
CUDA_ARCHITECTURES=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.') docker compose build
```

### 3) Build the SINGER image (layers on top of FiGS)

```bash
cd SINGER
docker compose build
```

### 4) Run SINGER development environment

```bash
docker compose run --rm singer
```

This will:
- Use the `singer:latest` image (extends `figs:latest` with SINGER-specific deps)
- Mount SINGER, FiGS-Standalone, and coverage_view_selection for editable development
- Auto-install editable packages on startup

### Configuration

You can customize paths via environment variables or a `.env` file:

```bash
# Default values
FIGS_PATH=../FiGS-Standalone
DATA_PATH=/media/admin/data/StanfordMSL/nerf_data
```

### Rebuilding

After changes to FiGS-Standalone's Dockerfile:

```bash
cd FiGS-Standalone
CUDA_ARCHITECTURES=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.') docker compose build

# Rebuild SINGER image on top of updated FiGS
cd ../SINGER
docker compose build
docker compose run --rm singer
```

## Usage

All commands run inside the Docker container. The main entry point is `notebooks/ssv_multi3dgs_campaign.py`.

```bash
# Generate RRT rollout data
python notebooks/ssv_multi3dgs_campaign.py generate-rollouts --config-file configs/experiment/ssv_multi3dgs.yml

# Generate observations from rollouts
python notebooks/ssv_multi3dgs_campaign.py generate-observations --config-file configs/experiment/ssv_multi3dgs.yml

# Train policy (two stages)
python notebooks/ssv_multi3dgs_campaign.py train-history --config-file configs/experiment/ssv_multi3dgs.yml
python notebooks/ssv_multi3dgs_campaign.py train-command --config-file configs/experiment/ssv_multi3dgs.yml

# Simulate trained policy vs expert
python notebooks/ssv_multi3dgs_campaign.py simulate --config-file configs/experiment/ssv_multi3dgs.yml
```

See `notebooks/simulating_trajectories.md` for details on configuring single-trajectory runs and using pre-generated trajectories.

### Key config files

| File | Controls |
|------|----------|
| `configs/experiment/ssv_multi3dgs.yml` | Cohort, method, flights, roster |
| `configs/scenes/{scene}.yml` | Semantic queries, radii, altitudes, RRT branches |
| `configs/method/rrt.json` | Rollout params, perturbation bounds, policy |

