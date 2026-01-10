"""
Reward functions for PDE-Gym environments.

This module provides reusable reward functions that can be used with
PDEEnv to define various RL tasks on PDE simulations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import numpy as np


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def __call__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """
        Compute reward for the current step.

        Args:
            observation: Current observation (H, W, 3) RGB image
            action: Action taken
            info: Additional information from the environment

        Returns:
            Reward value
        """
        pass

    def reset(self) -> None:
        """Reset any internal state. Called at episode start."""
        pass


class PatternMatchReward(RewardFunction):
    """
    Reward for matching a target pattern.

    Computes negative MSE between current observation and target,
    encouraging the agent to produce specific patterns.
    """

    def __init__(
        self,
        target: np.ndarray,
        scale: float = 1.0,
        normalize: bool = True,
    ):
        """
        Initialize pattern matching reward.

        Args:
            target: Target pattern (H, W) or (H, W, 3)
            scale: Reward scaling factor
            normalize: Whether to normalize images before comparison
        """
        self.target = target.astype(np.float32)
        if len(self.target.shape) == 3:
            # Convert to grayscale
            self.target = np.dot(self.target[..., :3], [0.299, 0.587, 0.114])
        if normalize:
            self.target = self.target / 255.0
        self.scale = scale
        self.normalize = normalize

    def __call__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        # Convert observation to grayscale
        obs = np.dot(observation[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        if self.normalize:
            obs = obs / 255.0

        # Resize if needed
        if obs.shape != self.target.shape:
            from PIL import Image

            pil_obs = Image.fromarray((obs * 255).astype(np.uint8))
            pil_obs = pil_obs.resize(
                (self.target.shape[1], self.target.shape[0]),
                Image.Resampling.BILINEAR,
            )
            obs = np.array(pil_obs).astype(np.float32) / 255.0

        # Compute negative MSE
        mse = np.mean((obs - self.target) ** 2)
        return -mse * self.scale


class VarianceReward(RewardFunction):
    """
    Reward based on spatial variance of the observation.

    Can be used to encourage pattern formation (high variance)
    or stability (low variance).
    """

    def __init__(
        self,
        target_variance: Optional[float] = None,
        maximize: bool = True,
        scale: float = 1.0,
    ):
        """
        Initialize variance reward.

        Args:
            target_variance: If set, reward approaches 0 as variance approaches target
            maximize: If True, reward high variance; if False, reward low variance
            scale: Reward scaling factor
        """
        self.target_variance = target_variance
        self.maximize = maximize
        self.scale = scale

    def __call__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        # Convert to grayscale
        gray = np.dot(observation[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        gray = gray / 255.0

        variance = np.var(gray)

        if self.target_variance is not None:
            # Reward for approaching target variance
            return -abs(variance - self.target_variance) * self.scale
        elif self.maximize:
            return variance * self.scale
        else:
            return -variance * self.scale


class EntropyReward(RewardFunction):
    """
    Reward based on spatial entropy of the observation.

    High entropy indicates diverse patterns; low entropy indicates
    uniform regions.
    """

    def __init__(
        self,
        n_bins: int = 32,
        maximize: bool = True,
        scale: float = 1.0,
    ):
        """
        Initialize entropy reward.

        Args:
            n_bins: Number of histogram bins for entropy calculation
            maximize: If True, reward high entropy
            scale: Reward scaling factor
        """
        self.n_bins = n_bins
        self.maximize = maximize
        self.scale = scale

    def __call__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        # Convert to grayscale
        gray = np.dot(observation[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

        # Compute histogram
        hist, _ = np.histogram(gray, bins=self.n_bins, range=(0, 255))
        hist = hist.astype(np.float32)
        hist = hist / hist.sum()  # Normalize

        # Compute entropy (avoid log(0))
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))

        # Normalize by max entropy
        max_entropy = np.log2(self.n_bins)
        normalized_entropy = entropy / max_entropy

        if self.maximize:
            return normalized_entropy * self.scale
        else:
            return -normalized_entropy * self.scale


class SparseReward(RewardFunction):
    """
    Sparse reward at episode end based on final state.

    Useful for goal-conditioned tasks where intermediate rewards
    might be misleading.
    """

    def __init__(
        self,
        final_reward_fn: RewardFunction,
        episode_length: int = 500,
    ):
        """
        Initialize sparse reward.

        Args:
            final_reward_fn: Reward function to apply at episode end
            episode_length: Expected episode length
        """
        self.final_reward_fn = final_reward_fn
        self.episode_length = episode_length

    def __call__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        step = info.get("step", 0)

        if step >= self.episode_length:
            return self.final_reward_fn(observation, action, info)
        return 0.0


class DeltaReward(RewardFunction):
    """
    Reward based on change from previous observation.

    Useful for detecting when patterns are forming or changing.
    """

    def __init__(
        self,
        reward_increase: bool = True,
        scale: float = 1.0,
    ):
        """
        Initialize delta reward.

        Args:
            reward_increase: If True, reward increased change; else decreased
            scale: Reward scaling factor
        """
        self.reward_increase = reward_increase
        self.scale = scale
        self._prev_obs: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset previous observation."""
        self._prev_obs = None

    def __call__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        if self._prev_obs is None:
            self._prev_obs = observation.copy()
            return 0.0

        # Compute change
        delta = np.mean(np.abs(observation.astype(float) - self._prev_obs.astype(float)))
        delta = delta / 255.0  # Normalize

        self._prev_obs = observation.copy()

        if self.reward_increase:
            return delta * self.scale
        else:
            return -delta * self.scale


class CompositeReward(RewardFunction):
    """
    Combine multiple reward functions.

    Useful for multi-objective tasks or curriculum learning.
    """

    def __init__(
        self,
        rewards: list[Tuple[RewardFunction, float]],
    ):
        """
        Initialize composite reward.

        Args:
            rewards: List of (reward_function, weight) tuples
        """
        self.rewards = rewards

    def reset(self) -> None:
        """Reset all sub-rewards."""
        for reward_fn, _ in self.rewards:
            reward_fn.reset()

    def __call__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        total = 0.0
        for reward_fn, weight in self.rewards:
            total += weight * reward_fn(observation, action, info)
        return total


class RegionTargetReward(RewardFunction):
    """
    Reward for achieving target concentration in a specific region.

    Useful for control tasks where the agent must guide concentrations
    to specific areas.
    """

    def __init__(
        self,
        region: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
        target_value: float = 0.8,
        scale: float = 1.0,
    ):
        """
        Initialize region target reward.

        Args:
            region: Target region as (x1, y1, x2, y2) in pixel coordinates
            target_value: Target normalized intensity [0, 1]
            scale: Reward scaling factor
        """
        self.region = region
        self.target_value = target_value
        self.scale = scale

    def __call__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        x1, y1, x2, y2 = self.region

        # Extract region
        region_obs = observation[y1:y2, x1:x2]

        # Convert to grayscale and normalize
        gray = np.dot(region_obs[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        gray = gray / 255.0

        # Compute mean intensity in region
        mean_intensity = np.mean(gray)

        # Reward for approaching target
        return -abs(mean_intensity - self.target_value) * self.scale


def create_pattern_task_reward(
    target_variance: float = 0.05,
    entropy_weight: float = 0.5,
    variance_weight: float = 0.5,
) -> CompositeReward:
    """
    Create a reward function for pattern formation tasks.

    Encourages the agent to create interesting patterns with
    appropriate variance and entropy.

    Args:
        target_variance: Target spatial variance
        entropy_weight: Weight for entropy component
        variance_weight: Weight for variance component

    Returns:
        Configured CompositeReward
    """
    return CompositeReward([
        (VarianceReward(target_variance=target_variance), variance_weight),
        (EntropyReward(maximize=True), entropy_weight),
    ])


def create_control_task_reward(
    target_region: Tuple[int, int, int, int],
    target_value: float = 0.8,
) -> RegionTargetReward:
    """
    Create a reward function for concentration control tasks.

    Encourages the agent to guide concentrations to a target region.

    Args:
        target_region: (x1, y1, x2, y2) pixel coordinates
        target_value: Target normalized intensity

    Returns:
        Configured RegionTargetReward
    """
    return RegionTargetReward(
        region=target_region,
        target_value=target_value,
    )
