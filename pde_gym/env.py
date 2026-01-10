"""
Gymnasium environment for PDE simulations.

This module provides the main PDEEnv class that wraps the VisualPDE
WebGL simulation as a standard Gymnasium environment for RL training.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any, Tuple, SupportsFloat, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from pde_gym.browser import SyncBrowserManager
from pde_gym.bridge import SyncPDEBridge


# Default presets available in VisualPDE
AVAILABLE_PRESETS = [
    # Reaction-Diffusion
    "GrayScott",
    "BrusselatorPDE",
    "SchnakenbergPDE",
    "GiererMeinhardt",
    "FHN",  # FitzHugh-Nagumo
    "Oregonator",
    # Wave equations
    "waveEquation",
    "waveEquation1D",
    "dampedWaveEquation",
    "KdV",  # Korteweg-de Vries
    "sineGordon",
    # Diffusion
    "heatEquation",
    "heatEquation1D",
    "nonlinearDiffusion",
    "chemotaxis",
    # Biological models
    "SIR",
    "SIS",
    "SEIR",
    "LotkaVolterra",
    "predatorPrey",
    "fisherKPP",
    # Fluid dynamics
    "NavierStokes",
    "BurgersPDE",
    "KuramotoSivashinsky",
]


class PDEEnv(gym.Env):
    """
    Gymnasium environment for PDE simulations using WebGL shaders.

    This environment allows RL agents to interact with PDE simulations
    by applying brush strokes that modify chemical concentrations.
    The simulation runs in a headless browser with GPU-accelerated
    (or SwiftShader CPU) WebGL rendering.

    Observation Space:
        Box(shape=(obs_height, obs_width, 3), dtype=np.uint8)
        RGB image of the current simulation state.

    Action Space:
        Box(shape=(5,), dtype=np.float32)
        Continuous brush parameters: [x, y, species, value, radius]
        - x, y: position in [0, 1]
        - species: which field, mapped from [0,1] to u/v
        - value: concentration [0, 1]
        - radius: brush radius [0.01, 0.2]

    Rewards:
        By default, returns 0 each step. Override `compute_reward()` or
        pass a custom reward function to customize.

    Example:
        >>> env = PDEEnv(preset="GrayScott")
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
        >>> env.close()
    """

    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 30,
    }

    def __init__(
        self,
        preset: str = "GrayScott",
        obs_width: int = 128,
        obs_height: int = 128,
        sim_width: int = 512,
        sim_height: int = 512,
        timesteps_per_step: int = 50,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
        headless: bool = True,
        use_swiftshader: bool = True,
        reward_fn: Optional[Callable[[np.ndarray, np.ndarray, Dict], float]] = None,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the PDE environment.

        Args:
            preset: Name of the PDE preset (e.g., "GrayScott", "BrusselatorPDE")
            obs_width: Width of observation images
            obs_height: Height of observation images
            sim_width: Internal simulation canvas width
            sim_height: Internal simulation canvas height
            timesteps_per_step: Number of PDE timesteps per environment step
            max_episode_steps: Maximum steps per episode (truncation)
            render_mode: "rgb_array" or "human" (for visualization)
            headless: Run browser in headless mode (required for HPC)
            use_swiftshader: Use CPU WebGL via SwiftShader (set False if GPU available)
            reward_fn: Custom reward function(obs, action, info) -> float
            seed: Random seed for reproducibility
            options: Override simulation parameters
        """
        super().__init__()

        self.preset = preset
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.sim_width = sim_width
        self.sim_height = sim_height
        self.timesteps_per_step = timesteps_per_step
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.headless = headless
        self.use_swiftshader = use_swiftshader
        self.reward_fn = reward_fn
        self._seed = seed
        self._options = options or {}

        # Episode tracking
        self._step_count = 0
        self._episode_count = 0

        # Define spaces
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_height, obs_width, 3),
            dtype=np.uint8,
        )

        # Action: [x, y, species, value, radius]
        # All normalized to [0, 1], we'll map them appropriately
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

        # Browser and bridge (lazy initialization)
        self._browser_manager: Optional[SyncBrowserManager] = None
        self._bridge: Optional[SyncPDEBridge] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize browser and simulation."""
        if self._initialized:
            return

        # Create event loop
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        # Initialize browser
        self._browser_manager = SyncBrowserManager(
            headless=self.headless,
            use_swiftshader=self.use_swiftshader,
            width=self.sim_width,
            height=self.sim_height,
        )
        self._browser_manager.start()

        # Create page and bridge
        page = self._browser_manager.new_page()
        self._bridge = SyncPDEBridge(page, self._loop)

        self._initialized = True

    def _get_observation(self) -> np.ndarray:
        """Capture and resize the current frame as observation."""
        frame = self._bridge.get_frame(as_array=True)

        # Resize if needed
        if frame.shape[0] != self.obs_height or frame.shape[1] != self.obs_width:
            pil_image = Image.fromarray(frame)
            pil_image = pil_image.resize(
                (self.obs_width, self.obs_height),
                Image.Resampling.BILINEAR,
            )
            frame = np.array(pil_image)

        return frame

    def _decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Decode normalized action vector to brush parameters.

        Args:
            action: [x, y, species, value, radius] in [0, 1]

        Returns:
            Dictionary with brush parameters
        """
        x, y, species_val, value, radius_norm = action

        # Map species from [0, 1] to "u" or "v"
        species = "u" if species_val < 0.5 else "v"

        # Map radius from [0, 1] to [0.01, 0.2]
        radius = 0.01 + radius_norm * 0.19

        return {
            "x": float(x),
            "y": float(y),
            "species": species,
            "value": float(value),
            "radius": float(radius),
        }

    def compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """
        Compute reward for the current step.

        Override this method or provide reward_fn in __init__ to customize.

        Args:
            observation: Current observation
            action: Action taken
            info: Additional information

        Returns:
            Reward value
        """
        if self.reward_fn is not None:
            return self.reward_fn(observation, action, info)

        # Default: no reward (use for exploration or custom wrappers)
        return 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (can override preset)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Use provided seed or stored seed
        reset_seed = seed if seed is not None else self._seed

        # Merge options
        reset_options = {**self._options}
        if options:
            reset_options.update(options)

        # Initialize if needed
        self._ensure_initialized()

        # Initialize simulation with preset
        preset = reset_options.pop("preset", self.preset)
        self._bridge.initialize(
            preset=preset,
            seed=reset_seed,
            options=reset_options if reset_options else None,
        )

        # Reset episode tracking
        self._step_count = 0
        self._episode_count += 1

        # Get initial observation
        self._bridge.render()
        observation = self._get_observation()

        info = {
            "episode": self._episode_count,
            "preset": preset,
            "sim_time": self._bridge.get_time(),
        }

        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Action to take [x, y, species, value, radius]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Decode and apply action
        brush_params = self._decode_action(action)
        self._bridge.apply_brush(**brush_params)

        # Evolve simulation
        self._bridge.step_and_render(self.timesteps_per_step)

        # Get observation
        observation = self._get_observation()

        # Update step count
        self._step_count += 1

        # Check termination
        terminated = False  # PDEs don't naturally terminate
        truncated = self._step_count >= self.max_episode_steps

        # Build info
        info = {
            "step": self._step_count,
            "sim_time": self._bridge.get_time(),
            "action_params": brush_params,
        }

        # Compute reward
        reward = self.compute_reward(observation, action, info)

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if not self._initialized:
            return None

        if self.render_mode == "rgb_array":
            return self._bridge.get_frame(as_array=True)

        return None

    def close(self) -> None:
        """Clean up resources."""
        if self._browser_manager is not None:
            self._browser_manager.stop()
            self._browser_manager = None

        self._bridge = None
        self._initialized = False

    def get_simulation_time(self) -> float:
        """Get the current simulation time."""
        if not self._initialized:
            return 0.0
        return self._bridge.get_time()

    def get_preset_options(self) -> Dict[str, Any]:
        """Get all current simulation options."""
        if not self._initialized:
            return {}
        return self._bridge.get_options()


class DiscreteActionPDEEnv(PDEEnv):
    """
    PDE Environment with discrete action space.

    Instead of continuous brush parameters, actions are selected from
    a predefined grid of positions and a set of value options.

    Action space is Discrete(n_positions * n_values * n_species).
    """

    def __init__(
        self,
        grid_size: int = 8,
        n_values: int = 4,
        n_species: int = 2,
        brush_radius: float = 0.05,
        **kwargs,
    ):
        """
        Initialize discrete action environment.

        Args:
            grid_size: Number of positions per dimension (total = grid_size^2)
            n_values: Number of discrete concentration values
            n_species: Number of species to control (1 or 2)
            brush_radius: Fixed brush radius
            **kwargs: Additional arguments passed to PDEEnv
        """
        super().__init__(**kwargs)

        self.grid_size = grid_size
        self.n_values = n_values
        self.n_species = n_species
        self.brush_radius = brush_radius

        # Override action space
        n_actions = grid_size * grid_size * n_values * n_species
        self.action_space = spaces.Discrete(n_actions)

        # Pre-compute action grid
        self._positions = np.linspace(0.1, 0.9, grid_size)
        self._values = np.linspace(0.0, 1.0, n_values)
        self._species_list = ["u", "v"][:n_species]

    def _decode_action(self, action: int) -> Dict[str, Any]:
        """Decode discrete action to brush parameters."""
        # Decode indices
        n_pos = self.grid_size * self.grid_size
        n_val = self.n_values

        species_idx = action // (n_pos * n_val)
        remainder = action % (n_pos * n_val)
        value_idx = remainder // n_pos
        pos_idx = remainder % n_pos

        x_idx = pos_idx % self.grid_size
        y_idx = pos_idx // self.grid_size

        return {
            "x": float(self._positions[x_idx]),
            "y": float(self._positions[y_idx]),
            "species": self._species_list[species_idx],
            "value": float(self._values[value_idx]),
            "radius": self.brush_radius,
        }


# Register environments with Gymnasium
gym.register(
    id="PDEEnv-v0",
    entry_point="pde_gym.env:PDEEnv",
    max_episode_steps=500,
)

gym.register(
    id="PDEEnv-GrayScott-v0",
    entry_point="pde_gym.env:PDEEnv",
    kwargs={"preset": "GrayScott"},
    max_episode_steps=500,
)

gym.register(
    id="PDEEnv-Wave-v0",
    entry_point="pde_gym.env:PDEEnv",
    kwargs={"preset": "waveEquation"},
    max_episode_steps=500,
)

gym.register(
    id="PDEEnv-Discrete-v0",
    entry_point="pde_gym.env:DiscreteActionPDEEnv",
    max_episode_steps=500,
)
