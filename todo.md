# PDE-Gym: Converting Visual-PDE to an RL Environment

## Overview

Convert the [visual-pde](https://github.com/FloWsnr/visual-pde) WebGL-based PDE simulator into a Python-based Gymnasium RL environment that:
- Runs **headless** (cloud/HPC compatible)
- Preserves the **shader-based simulation** for GPU-accelerated PDE solving
- Provides a standard **Gymnasium API** for RL agents
- Supports **parallel environments** for distributed training

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python RL Training Loop                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Agent     │───>│   PDEEnv    │───>│  Vectorized │          │
│  │  (Stable    │<───│ (Gymnasium) │<───│    Envs     │          │
│  │  Baselines) │    └─────────────┘    └─────────────┘          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Playwright (Python)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Headless Chrome Browser                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  window.VPDE API                                             │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │ │
│  │  │ stepN() │  │ reset() │  │applyBrush│ │ render()│          │ │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘          │ │
│  │       │            │            │            │                │ │
│  │       ▼            ▼            ▼            ▼                │ │
│  │  ┌─────────────────────────────────────────────────┐         │ │
│  │  │           WebGL Shader Simulation               │         │ │
│  │  │  (Three.js + GLSL PDE Solvers)                  │         │ │
│  │  │  - GPU-accelerated finite differences           │         │ │
│  │  │  - 230+ preset PDE configurations               │         │ │
│  │  │  - SwiftShader fallback for CPU-only            │         │ │
│  │  └─────────────────────────────────────────────────┘         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology | Reason |
|-----------|------------|--------|
| Python RL API | Gymnasium | Standard RL interface |
| Browser Automation | Playwright (Python) | Better async support, maintained |
| Headless Browser | Chromium | WebGL support via SwiftShader |
| Simulation Engine | VisualPDE (WebGL/Three.js) | Preserve shader-based solving |
| Parallel Envs | Gymnasium VectorEnv | Async SubprocVecEnv support |

## Implementation Plan

### Phase 1: Project Setup & Core Bridge

- [ ] **1.1 Initialize Python project structure**
  - Create `pde_gym/` package directory
  - Set up `pyproject.toml` with dependencies (gymnasium, playwright, numpy, pillow)
  - Create `__init__.py` and entry points
  - Add development dependencies (pytest, black, mypy)

- [ ] **1.2 Copy and adapt VisualPDE simulation files**
  - Copy `sim/` directory with shaders and main.js
  - Copy `headless/index.html` as base template
  - Modify paths for local serving
  - Create minimal HTTP server or use file:// protocol

- [ ] **1.3 Implement BrowserManager class**
  - Playwright browser lifecycle management
  - Chromium launch with WebGL flags (SwiftShader for CPU mode)
  - Page creation and cleanup
  - Async context management for proper resource handling

- [ ] **1.4 Implement PDEBridge class**
  - JavaScript evaluation wrapper for VPDE API
  - Methods: `step(n)`, `reset()`, `apply_brush()`, `render()`, `get_state()`
  - Frame capture and conversion to numpy array
  - Error handling and timeout management

### Phase 2: Gymnasium Environment

- [ ] **2.1 Define observation space**
  - Image-based: `Box(shape=(H, W, C), dtype=np.uint8)`
  - Optional: Raw state tensor from `getRawState()`
  - Configurable resolution (default 128x128 for faster training)

- [ ] **2.2 Define action space**
  - Continuous brush actions: `Box(shape=(5,), dtype=np.float32)`
    - `x, y`: position [0, 1]
    - `species`: which field to modify [0, 1] → u/v
    - `value`: concentration to apply [0, 1]
    - `radius`: brush size [0.01, 0.2]
  - Discrete variant: Predefined action set for simpler tasks

- [ ] **2.3 Implement PDEEnv class**
  - `__init__`: Configure preset, resolution, timesteps_per_step
  - `reset()`: Initialize browser page, load preset, return initial obs
  - `step(action)`: Apply brush, evolve PDE, return (obs, reward, done, truncated, info)
  - `render()`: Return current frame as RGB array
  - `close()`: Cleanup browser resources

- [ ] **2.4 Implement reward functions**
  - Base class with abstract `compute_reward()` method
  - Example rewards:
    - `PatternMatchReward`: Match target pattern (e.g., Turing spots)
    - `StabilityReward`: Minimize variance over time
    - `ReachTargetReward`: Guide concentration to target value
    - `CustomReward`: User-defined function

### Phase 3: Headless & Performance Optimization

- [ ] **3.1 SwiftShader configuration for CPU-only execution**
  - Chrome flags: `--use-gl=angle --use-angle=swiftshader-webgl`
  - Test on headless Linux server
  - Document GPU vs CPU performance tradeoffs

- [ ] **3.2 Optimize observation pipeline**
  - Direct WebGL pixel readback (avoid canvas.toDataURL overhead)
  - Implement downsampling in shader if needed
  - Cache numpy array allocation

- [ ] **3.3 Benchmark single environment**
  - Measure steps/second at various resolutions
  - Profile browser communication overhead
  - Identify bottlenecks

### Phase 4: Vectorized Environments

- [ ] **4.1 Implement AsyncVectorEnv wrapper**
  - Multiple browser pages in single browser instance
  - Async stepping with `asyncio.gather()`
  - Shared BrowserManager for resource efficiency

- [ ] **4.2 Implement SubprocVectorEnv support**
  - One browser per subprocess
  - Pickle-safe environment configuration
  - Proper cleanup on process termination

- [ ] **4.3 Benchmark parallel performance**
  - Test scaling: 1, 2, 4, 8, 16 parallel environments
  - Measure throughput vs resource usage
  - Document recommended configurations

### Phase 5: Preset Library & Task Suite

- [ ] **5.1 Create task wrapper classes**
  - `GrayScottPatternTask`: Form Turing patterns
  - `WaveControlTask`: Guide wave propagation
  - `ReactionControlTask`: Stabilize unstable equilibria
  - `HeatDissipationTask`: Efficient heat spreading

- [ ] **5.2 Implement curriculum support**
  - Difficulty progression (e.g., larger domains, longer horizons)
  - Parameter randomization for generalization
  - Automatic curriculum based on success rate

- [ ] **5.3 Add standard RL baselines**
  - Integration examples with Stable-Baselines3
  - PPO, SAC, TD3 training scripts
  - Hyperparameter recommendations

### Phase 6: Documentation & Testing

- [ ] **6.1 Write comprehensive documentation**
  - Installation guide (including headless setup)
  - Quick start tutorial
  - API reference
  - Example notebooks

- [ ] **6.2 Implement test suite**
  - Unit tests for bridge and environment
  - Integration tests for full episodes
  - CI/CD with headless browser testing

- [ ] **6.3 Create example training scripts**
  - Single environment training
  - Vectorized training
  - Custom reward function examples

## Key Design Decisions

### Why Playwright over Pyppeteer?
- More actively maintained
- Better async support
- Automatic browser management
- Cross-browser support (useful for future)

### Why keep WebGL simulation?
- GPU acceleration on capable hardware
- SwiftShader provides CPU fallback
- Preserves numerical accuracy of original
- Avoids reimplementing complex PDE solvers

### Observation Space Trade-offs

| Type | Pros | Cons |
|------|------|------|
| RGB Image (128x128) | Standard CNN input, fast | Lossy, colormap-dependent |
| Grayscale (128x128) | Simpler, faster | Single channel only |
| Raw State (64x64x4) | Full precision, all species | Larger, needs custom network |

### Action Space Options

| Type | Dims | Description |
|------|------|-------------|
| Continuous | 5 | (x, y, species, value, radius) |
| Discrete | N | Predefined brush locations/values |
| Multi-Discrete | 3 | (position_id, species, value_level) |

## File Structure

```
pde_gym/
├── pde_gym/
│   ├── __init__.py              # Package exports
│   ├── env.py                   # PDEEnv Gymnasium environment
│   ├── bridge.py                # PDEBridge JavaScript interface
│   ├── browser.py               # BrowserManager Playwright wrapper
│   ├── rewards.py               # Reward function library
│   ├── tasks/                   # Pre-configured task environments
│   │   ├── __init__.py
│   │   ├── gray_scott.py
│   │   ├── wave_control.py
│   │   └── ...
│   ├── vector/                  # Vectorized environment support
│   │   ├── __init__.py
│   │   └── async_vector.py
│   └── sim/                     # VisualPDE simulation files (copied)
│       ├── headless.html
│       └── scripts/
│           └── RD/
│               ├── main.js
│               ├── presets.js
│               └── *_shaders.js
├── tests/
│   ├── test_env.py
│   ├── test_bridge.py
│   └── test_vector.py
├── examples/
│   ├── quickstart.py
│   ├── train_ppo.py
│   ├── custom_reward.py
│   └── notebooks/
│       └── tutorial.ipynb
├── pyproject.toml
├── README.md
└── todo.md
```

## Dependencies

```toml
[project]
dependencies = [
    "gymnasium>=0.29.0",
    "playwright>=1.40.0",
    "numpy>=1.24.0",
    "Pillow>=10.0.0",
]

[project.optional-dependencies]
train = [
    "stable-baselines3>=2.0.0",
    "torch>=2.0.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]
```

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Browser overhead slows training | High | Optimize observation pipeline, use raw state |
| SwiftShader too slow on CPU | Medium | Provide GPU mode documentation, batch environments |
| Memory leaks in long training | Medium | Periodic browser restart, memory monitoring |
| WebGL context loss | Low | Automatic page reload, state recovery |

## Success Criteria

1. **Functional**: Environment passes Gymnasium compatibility tests
2. **Headless**: Runs on Linux server without display (X11/Wayland)
3. **Performance**: >100 steps/second on GPU, >10 steps/second on CPU
4. **Scalable**: Linear speedup up to 8 parallel environments
5. **Documented**: Complete API docs and training examples

## Next Steps

1. Start with Phase 1.1-1.4 to establish the core bridge
2. Build minimal PDEEnv in Phase 2
3. Validate headless execution in Phase 3
4. Iterate based on performance benchmarks
