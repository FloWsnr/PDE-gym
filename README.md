# PDE-Gym

A Gymnasium RL environment for Partial Differential Equation (PDE) simulations using WebGL shaders.

## Overview

PDE-Gym wraps the [VisualPDE](https://visualpde.com/) WebGL simulation as a standard Gymnasium environment, enabling reinforcement learning agents to interact with PDE dynamics. The simulation runs in a headless browser, preserving GPU-accelerated shader-based computation while being fully compatible with cloud and HPC environments.

### Key Features

- **GPU-Accelerated Simulation**: WebGL shaders for fast PDE solving
- **Headless Execution**: Runs without display using SwiftShader (CPU) or EGL (GPU)
- **230+ PDE Presets**: Reaction-diffusion, waves, fluids, biological models
- **Standard RL Interface**: Full Gymnasium API compatibility
- **Flexible Actions**: Continuous brush strokes to modify concentrations
- **Customizable Rewards**: Built-in reward functions for various tasks

## Installation

```bash
# Clone the repository
git clone https://github.com/FloWsnr/PDE-gym.git
cd PDE-gym

# Install the package
pip install -e .

# Install Playwright browsers (required)
playwright install chromium
```

### For Training (optional)

```bash
pip install -e ".[train]"  # Includes stable-baselines3, torch
```

## Quick Start

```python
from pde_gym import PDEEnv

# Create environment
env = PDEEnv(
    preset="GrayScott",       # PDE type
    obs_width=128,            # Observation resolution
    obs_height=128,
    timesteps_per_step=50,    # PDE steps per env step
    headless=True,            # Required for servers
    use_swiftshader=True,     # CPU-only WebGL
)

# Standard Gymnasium interface
obs, info = env.reset(seed=42)

for _ in range(100):
    # Action: [x, y, species, value, radius]
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Using with Gymnasium Registry

```python
import gymnasium as gym
import pde_gym  # Register environments

# Pre-configured environments
env = gym.make("PDEEnv-GrayScott-v0")
env = gym.make("PDEEnv-Wave-v0")
env = gym.make("PDEEnv-Discrete-v0")  # Discrete action space
```

## Action Space

Continuous actions control brush strokes that modify chemical concentrations:

| Index | Parameter | Range | Description |
|-------|-----------|-------|-------------|
| 0 | x | [0, 1] | Horizontal position |
| 1 | y | [0, 1] | Vertical position |
| 2 | species | [0, 1] | Field selector (u if <0.5, v if >=0.5) |
| 3 | value | [0, 1] | Concentration to apply |
| 4 | radius | [0, 1] | Brush size (maps to 0.01-0.2) |

## Available Presets

### Reaction-Diffusion
- `GrayScott` - Classic Turing patterns
- `BrusselatorPDE` - Chemical oscillations
- `FHN` - FitzHugh-Nagumo neuronal model
- `GiererMeinhardt` - Morphogenesis patterns

### Wave Equations
- `waveEquation` - 2D wave equation
- `dampedWaveEquation` - With dissipation
- `KdV` - Korteweg-de Vries solitons

### Biological Models
- `SIR`, `SEIR` - Epidemic models
- `LotkaVolterra` - Predator-prey dynamics
- `fisherKPP` - Population spread

### Fluid Dynamics
- `NavierStokes` - Fluid flow
- `BurgersPDE` - Shock waves

## Custom Rewards

```python
from pde_gym import PDEEnv
from pde_gym.rewards import VarianceReward, CompositeReward, EntropyReward

# Single reward function
reward_fn = VarianceReward(maximize=True, scale=100.0)

# Composite reward
reward_fn = CompositeReward([
    (VarianceReward(target_variance=0.05), 0.5),
    (EntropyReward(maximize=True), 0.5),
])

env = PDEEnv(preset="GrayScott", reward_fn=reward_fn)
```

## Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from pde_gym import PDEEnv

env = PDEEnv(
    preset="GrayScott",
    obs_width=64,
    obs_height=64,
    timesteps_per_step=50,
)

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## Architecture

```
Python RL Agent
       |
       v
+------------------+
|     PDEEnv       |  <- Gymnasium Environment
|  (pde_gym/env.py)|
+--------+---------+
         | Playwright (Python)
         v
+------------------+
|  Headless Chrome |
|  +------------+  |
|  | window.VPDE|  |  <- JavaScript API
|  |    API     |  |
|  +-----+------+  |
|        |         |
|  +-----v------+  |
|  |   WebGL    |  |  <- GPU Shaders
|  |  Shaders   |  |
|  +------------+  |
+------------------+
```

## System Requirements

PDE-Gym requires WebGL support in the headless browser. This is provided by:

- **GPU Mode**: NVIDIA/AMD GPU with proper drivers and EGL support
- **CPU Mode**: SwiftShader software renderer (bundled with Chromium)

### Tested Configurations

| Environment | Configuration | Status |
|-------------|--------------|--------|
| Linux + NVIDIA GPU | `use_swiftshader=False` | Recommended |
| Linux + SwiftShader | `use_swiftshader=True` | Works, slower |
| Docker + SwiftShader | Requires proper GL libs | Works with setup |
| WSL2 + SwiftShader | May need X server | Limited support |

### Troubleshooting WebGL

If WebGL fails to initialize:

```bash
# Check WebGL support
python -c "from pde_gym.browser import _find_chromium_executable; print(_find_chromium_executable())"

# Verify browser can create WebGL context
python test_webgl.py  # Included in repo
```

## Headless Server Setup

For cloud/HPC environments without displays:

```python
# SwiftShader (CPU) - works on most systems
env = PDEEnv(headless=True, use_swiftshader=True)

# EGL (GPU) - requires NVIDIA/AMD drivers
env = PDEEnv(headless=True, use_swiftshader=False)
```

## API Reference

### PDEEnv

```python
PDEEnv(
    preset: str = "GrayScott",        # PDE preset name
    obs_width: int = 128,             # Observation width
    obs_height: int = 128,            # Observation height
    sim_width: int = 512,             # Simulation resolution
    sim_height: int = 512,
    timesteps_per_step: int = 50,     # PDE steps per env step
    max_episode_steps: int = 500,     # Episode length
    render_mode: str = None,          # "rgb_array" or "human"
    headless: bool = True,            # Headless browser mode
    use_swiftshader: bool = True,     # CPU WebGL rendering
    reward_fn: Callable = None,       # Custom reward function
    seed: int = None,                 # Random seed
    options: dict = None,             # PDE parameter overrides
)
```

### Methods

- `reset(seed=None, options=None)` -> `(obs, info)`
- `step(action)` -> `(obs, reward, terminated, truncated, info)`
- `render()` -> `np.ndarray` (if render_mode="rgb_array")
- `close()` - Cleanup browser resources

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run quick tests (no browser)
python tests/test_env.py

# Format code
black pde_gym/ tests/
```

## License

MIT License - See [LICENSE](LICENSE) for details.

Based on [VisualPDE](https://visualpde.com/) by Benjamin Walker, Adam Sherwood-Smith, and colleagues.
