"""
Bridge between Python and the VisualPDE JavaScript API.

This module provides a Python interface to the window.VPDE API exposed by
the VisualPDE simulation running in the browser.
"""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from typing import Optional, Dict, Any, Tuple, Literal

import numpy as np
from PIL import Image
from playwright.async_api import Page

from pde_gym.browser import get_sim_html_path


# Type aliases
Species = Literal["u", "v", "w", "q"]
BrushShape = Literal["circle", "square"]
BrushAction = Literal["replace", "add"]


class PDEBridge:
    """
    Python bridge to the VisualPDE JavaScript API.

    This class wraps the window.VPDE API exposed by the simulation,
    providing Python-friendly methods for controlling the PDE simulation.

    Attributes:
        page: The Playwright page running the simulation
        preset: The current PDE preset name
        ready: Whether the simulation is initialized and ready

    Example:
        >>> bridge = PDEBridge(page)
        >>> await bridge.initialize("GrayScott")
        >>> await bridge.step(100)  # Evolve 100 timesteps
        >>> frame = await bridge.render()  # Get current state as image
    """

    def __init__(self, page: Page):
        """
        Initialize the bridge with a browser page.

        Args:
            page: A Playwright page instance
        """
        self.page = page
        self.preset: Optional[str] = None
        self.ready = False
        self._html_path = get_sim_html_path()

    async def initialize(
        self,
        preset: str = "GrayScott",
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the simulation with a preset configuration.

        Args:
            preset: Name of the PDE preset (e.g., "GrayScott", "BrusselatorPDE")
            seed: Random seed for reproducibility
            options: Optional parameter overrides
            timeout: Maximum time to wait for initialization (seconds)

        Raises:
            TimeoutError: If simulation doesn't initialize within timeout
        """
        # Build URL with preset and optional seed
        url = f"file://{self._html_path}?preset={preset}"
        if seed is not None:
            url += f"&seed={seed}"

        # Navigate to simulation page
        await self.page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)

        # Wait for VPDE to be ready
        await self.page.wait_for_function(
            "window.VPDE_READY === true",
            timeout=timeout * 1000,
        )

        # Apply option overrides if provided
        if options:
            await self._apply_options(options)

        # Reset to apply all settings
        await self.reset()

        self.preset = preset
        self.ready = True

    async def _apply_options(self, options: Dict[str, Any]) -> None:
        """Apply option overrides to the simulation."""
        await self.page.evaluate(
            """(opts) => {
                for (const [key, value] of Object.entries(opts)) {
                    window.VPDE.setOption(key, value);
                }
                window.VPDE.updateProblem();
            }""",
            options,
        )

    async def reset(self) -> None:
        """
        Reset the simulation to initial conditions.

        This clears all concentrations and restarts from the initial
        conditions defined in the current preset.
        """
        await self.page.evaluate("window.VPDE.reset()")

    async def step(self, n: int = 1) -> None:
        """
        Evolve the PDE simulation by n timesteps.

        Args:
            n: Number of timesteps to evolve (default: 1)
        """
        await self.page.evaluate(f"window.VPDE.stepN({n})")

    async def render(self) -> None:
        """
        Render the current simulation state to the canvas.

        Call this after step() to update the visual representation.
        """
        await self.page.evaluate("window.VPDE.render()")

    async def step_and_render(self, n: int = 1) -> None:
        """
        Evolve simulation and render in one call.

        More efficient than calling step() and render() separately.

        Args:
            n: Number of timesteps to evolve
        """
        await self.page.evaluate(
            f"""() => {{
                window.VPDE.stepN({n});
                window.VPDE.render();
            }}"""
        )

    async def apply_brush(
        self,
        x: float,
        y: float,
        species: Species = "u",
        value: float = 1.0,
        radius: float = 0.05,
        shape: BrushShape = "circle",
        action: BrushAction = "replace",
    ) -> None:
        """
        Apply a brush stroke to modify concentrations.

        This is the primary action mechanism for RL agents.

        Args:
            x: X coordinate (0 to 1, normalized)
            y: Y coordinate (0 to 1, normalized)
            species: Which chemical species to modify ("u", "v", "w", "q")
            value: Concentration value to apply
            radius: Brush radius in simulation units
            shape: Brush shape ("circle" or "square")
            action: How to apply ("replace" overwrites, "add" adds to current)
        """
        # Clamp coordinates to valid range
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        await self.page.evaluate(
            f"window.VPDE.applyBrush({x}, {y}, '{species}', {value}, {radius})"
        )

    async def get_frame(self, as_array: bool = True) -> np.ndarray | str:
        """
        Capture the current frame from the simulation canvas.

        Args:
            as_array: If True, return numpy array; otherwise return base64 PNG

        Returns:
            RGB image as numpy array (H, W, 3) or base64-encoded PNG string
        """
        # Get base64 PNG from canvas
        data_url = await self.page.evaluate("window.VPDE.captureFrame()")

        if not as_array:
            # Return just the base64 data without prefix
            return data_url.split(",")[1]

        # Convert to numpy array
        base64_data = data_url.split(",")[1]
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))

        # Convert to RGB numpy array
        return np.array(image.convert("RGB"))

    async def get_grayscale_frame(self) -> np.ndarray:
        """
        Capture the current frame as a grayscale image.

        Returns:
            Grayscale image as numpy array (H, W)
        """
        rgb_frame = await self.get_frame(as_array=True)
        # Convert to grayscale using standard weights
        return np.dot(rgb_frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    async def get_time(self) -> float:
        """
        Get the current simulation time.

        Returns:
            Current simulation time value
        """
        return await self.page.evaluate("window.VPDE.getTime()")

    async def get_options(self) -> Dict[str, Any]:
        """
        Get all current simulation options.

        Returns:
            Dictionary of all simulation parameters
        """
        return await self.page.evaluate("window.VPDE.getOptions()")

    async def set_option(self, key: str, value: Any) -> None:
        """
        Set a single simulation option.

        Args:
            key: Option name
            value: Option value
        """
        await self.page.evaluate(
            f"window.VPDE.setOption('{key}', {repr(value) if isinstance(value, str) else value})"
        )

    async def load_preset(self, preset: str) -> None:
        """
        Load a different PDE preset.

        Args:
            preset: Name of the preset to load
        """
        await self.page.evaluate(f"window.VPDE.loadPreset('{preset}')")
        self.preset = preset
        await self.reset()

    async def get_canvas_size(self) -> Tuple[int, int]:
        """
        Get the current canvas dimensions.

        Returns:
            Tuple of (width, height) in pixels
        """
        size = await self.page.evaluate(
            """() => {
                const canvas = document.getElementById('simCanvas');
                return [canvas.width, canvas.height];
            }"""
        )
        return tuple(size)


class SyncPDEBridge:
    """
    Synchronous wrapper around PDEBridge for non-async code.

    Provides the same interface as PDEBridge but with synchronous methods,
    suitable for use with Gymnasium environments.
    """

    def __init__(self, page: Page, loop: asyncio.AbstractEventLoop):
        """
        Initialize synchronous bridge.

        Args:
            page: Playwright page instance
            loop: Event loop to use for async operations
        """
        self._async_bridge = PDEBridge(page)
        self._loop = loop

    def _run(self, coro):
        """Run coroutine synchronously."""
        return self._loop.run_until_complete(coro)

    def initialize(
        self,
        preset: str = "GrayScott",
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the simulation."""
        self._run(self._async_bridge.initialize(preset, seed, options, timeout))

    def reset(self) -> None:
        """Reset to initial conditions."""
        self._run(self._async_bridge.reset())

    def step(self, n: int = 1) -> None:
        """Evolve simulation by n timesteps."""
        self._run(self._async_bridge.step(n))

    def render(self) -> None:
        """Render current state to canvas."""
        self._run(self._async_bridge.render())

    def step_and_render(self, n: int = 1) -> None:
        """Step and render in one call."""
        self._run(self._async_bridge.step_and_render(n))

    def apply_brush(
        self,
        x: float,
        y: float,
        species: Species = "u",
        value: float = 1.0,
        radius: float = 0.05,
        shape: BrushShape = "circle",
        action: BrushAction = "replace",
    ) -> None:
        """Apply brush stroke."""
        self._run(
            self._async_bridge.apply_brush(x, y, species, value, radius, shape, action)
        )

    def get_frame(self, as_array: bool = True) -> np.ndarray | str:
        """Capture current frame."""
        return self._run(self._async_bridge.get_frame(as_array))

    def get_grayscale_frame(self) -> np.ndarray:
        """Capture grayscale frame."""
        return self._run(self._async_bridge.get_grayscale_frame())

    def get_time(self) -> float:
        """Get simulation time."""
        return self._run(self._async_bridge.get_time())

    def get_options(self) -> Dict[str, Any]:
        """Get all options."""
        return self._run(self._async_bridge.get_options())

    def set_option(self, key: str, value: Any) -> None:
        """Set option."""
        self._run(self._async_bridge.set_option(key, value))

    def load_preset(self, preset: str) -> None:
        """Load preset."""
        self._run(self._async_bridge.load_preset(preset))

    def get_canvas_size(self) -> Tuple[int, int]:
        """Get canvas size."""
        return self._run(self._async_bridge.get_canvas_size())

    @property
    def preset(self) -> Optional[str]:
        """Current preset name."""
        return self._async_bridge.preset

    @property
    def ready(self) -> bool:
        """Whether simulation is ready."""
        return self._async_bridge.ready
