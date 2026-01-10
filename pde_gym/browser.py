"""
Browser management for PDE-Gym using Playwright.

This module handles the lifecycle of headless Chromium browsers with WebGL support,
including both GPU-accelerated and SwiftShader (CPU) rendering modes.
"""

from __future__ import annotations

import asyncio
import glob
import os
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from playwright.async_api import async_playwright, Browser, Page, Playwright


def _find_chromium_executable() -> Optional[str]:
    """
    Find an available Chromium executable in the Playwright cache.

    Returns:
        Path to chromium executable, or None if not found.
    """
    cache_dir = Path.home() / ".cache" / "ms-playwright"

    # Try to find any chromium installation
    patterns = [
        "chromium-*/chrome-linux/chrome",
        "chromium_headless_shell-*/chrome-linux/headless_shell",
        "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell",
    ]

    for pattern in patterns:
        matches = list(cache_dir.glob(pattern))
        if matches:
            # Sort by version (higher is newer) and return the newest
            matches.sort(reverse=True)
            return str(matches[0])

    return None


class BrowserManager:
    """
    Manages headless Chromium browser instances with WebGL support.

    This class handles browser lifecycle, page creation, and provides both
    GPU-accelerated and CPU-only (SwiftShader) rendering modes for headless
    execution on cloud/HPC systems.

    Attributes:
        headless: Whether to run in headless mode
        use_swiftshader: Whether to use SwiftShader for CPU-only WebGL
        width: Canvas width in pixels
        height: Canvas height in pixels

    Example:
        >>> async with BrowserManager() as manager:
        ...     page = await manager.new_page()
        ...     # Use page for simulation
        ...     await manager.close_page(page)
    """

    def __init__(
        self,
        headless: bool = True,
        use_swiftshader: bool = True,
        width: int = 512,
        height: int = 512,
        max_pages_per_browser: int = 50,
    ):
        """
        Initialize the browser manager.

        Args:
            headless: Run browser in headless mode (required for HPC/cloud)
            use_swiftshader: Use SwiftShader for CPU-only WebGL rendering.
                           Set to False if GPU is available.
            width: Default canvas width in pixels
            height: Default canvas height in pixels
            max_pages_per_browser: Maximum pages before browser restart
                                  (prevents memory leaks)
        """
        self.headless = headless
        self.use_swiftshader = use_swiftshader
        self.width = width
        self.height = height
        self.max_pages_per_browser = max_pages_per_browser

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._pages: List[Page] = []
        self._page_count = 0
        self._lock = asyncio.Lock()

    def _get_browser_args(self) -> List[str]:
        """Get Chromium launch arguments for WebGL support."""
        args = [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-accelerated-2d-canvas",
            "--disable-gpu-sandbox",
            "--disable-web-security",  # Allow file:// protocol
            "--allow-file-access-from-files",
            f"--window-size={self.width},{self.height}",
        ]

        if self.use_swiftshader:
            # Use SwiftShader for CPU-only WebGL rendering
            # This is essential for headless execution without GPU
            args.extend([
                "--use-gl=angle",
                "--use-angle=swiftshader-webgl",
            ])
        else:
            # Use system GPU (EGL)
            args.extend([
                "--use-gl=egl",
                "--enable-gpu-rasterization",
            ])

        if self.headless:
            args.append("--disable-extensions")

        return args

    async def start(self) -> None:
        """Start the Playwright browser instance."""
        if self._browser is not None:
            return

        self._playwright = await async_playwright().start()

        # Try to find an explicit chromium executable
        executable_path = _find_chromium_executable()

        launch_kwargs = {
            "headless": self.headless,
            "args": self._get_browser_args(),
        }

        if executable_path:
            launch_kwargs["executable_path"] = executable_path

        self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        self._page_count = 0

    async def stop(self) -> None:
        """Stop the browser and cleanup resources."""
        # Close all pages
        for page in self._pages:
            try:
                await page.close()
            except Exception:
                pass
        self._pages.clear()

        # Close browser
        if self._browser is not None:
            await self._browser.close()
            self._browser = None

        # Stop playwright
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def _maybe_restart_browser(self) -> None:
        """Restart browser if page count exceeds limit (prevents memory leaks)."""
        if self._page_count >= self.max_pages_per_browser:
            await self.stop()
            await self.start()

    async def new_page(self) -> Page:
        """
        Create a new browser page.

        Returns:
            A new Playwright Page instance configured for WebGL rendering.
        """
        async with self._lock:
            if self._browser is None:
                await self.start()

            await self._maybe_restart_browser()

            context = await self._browser.new_context(
                viewport={"width": self.width, "height": self.height},
                device_scale_factor=1,
            )
            page = await context.new_page()
            self._pages.append(page)
            self._page_count += 1

            return page

    async def close_page(self, page: Page) -> None:
        """
        Close a browser page.

        Args:
            page: The page to close
        """
        if page in self._pages:
            self._pages.remove(page)

        try:
            context = page.context
            await page.close()
            await context.close()
        except Exception:
            pass

    async def __aenter__(self) -> "BrowserManager":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


class SyncBrowserManager:
    """
    Synchronous wrapper around BrowserManager for use in non-async code.

    This wrapper maintains an event loop and provides synchronous methods
    for browser management, suitable for use with Gymnasium environments.

    Example:
        >>> with SyncBrowserManager() as manager:
        ...     page = manager.new_page()
        ...     # Use page for simulation
        ...     manager.close_page(page)
    """

    def __init__(
        self,
        headless: bool = True,
        use_swiftshader: bool = True,
        width: int = 512,
        height: int = 512,
        max_pages_per_browser: int = 50,
    ):
        """Initialize synchronous browser manager."""
        self._async_manager = BrowserManager(
            headless=headless,
            use_swiftshader=use_swiftshader,
            width=width,
            height=height,
            max_pages_per_browser=max_pages_per_browser,
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._owns_loop = False

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._owns_loop = True
        return self._loop

    def _run(self, coro):
        """Run a coroutine synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def start(self) -> None:
        """Start the browser."""
        self._run(self._async_manager.start())

    def stop(self) -> None:
        """Stop the browser."""
        self._run(self._async_manager.stop())
        if self._owns_loop and self._loop is not None:
            self._loop.close()
            self._loop = None

    def new_page(self) -> Page:
        """Create a new page."""
        return self._run(self._async_manager.new_page())

    def close_page(self, page: Page) -> None:
        """Close a page."""
        self._run(self._async_manager.close_page(page))

    def __enter__(self) -> "SyncBrowserManager":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def get_sim_html_path() -> str:
    """
    Get the absolute path to the headless simulation HTML file.

    Returns:
        Absolute file path to headless.html
    """
    module_dir = Path(__file__).parent
    html_path = module_dir / "headless" / "index.html"
    return str(html_path.resolve())
