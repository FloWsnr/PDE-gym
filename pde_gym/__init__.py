"""
PDE-Gym: Gymnasium RL environment for PDE simulations using WebGL shaders.

This package provides a Gymnasium-compatible environment for reinforcement learning
on partial differential equation (PDE) simulations. The simulation is powered by
WebGL shaders running in a headless browser, preserving GPU acceleration while
being fully compatible with cloud and HPC environments.
"""

from pde_gym.env import PDEEnv
from pde_gym.browser import BrowserManager
from pde_gym.bridge import PDEBridge

__version__ = "0.1.0"
__all__ = ["PDEEnv", "BrowserManager", "PDEBridge"]
