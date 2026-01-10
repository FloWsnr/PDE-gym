#!/usr/bin/env python3
"""Test WebGL availability in headless Chrome."""

from playwright.sync_api import sync_playwright
import os

WEBGL_CHECK_JS = '''() => {
    const canvas = document.createElement("canvas");
    let gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    if (!gl) {
        gl = canvas.getContext("webgl2");
    }
    if (!gl) {
        return { available: false, error: "No WebGL context" };
    }
    const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
    return {
        available: true,
        vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : "Unknown",
        renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : "Unknown",
    };
}'''

def test_webgl_with_args(args_list, name):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path="/root/.cache/ms-playwright/chromium-1194/chrome-linux/chrome",
            args=args_list
        )
        page = browser.new_page()
        webgl_info = page.evaluate(WEBGL_CHECK_JS)
        print(f"{name}: {webgl_info}")
        browser.close()

if __name__ == "__main__":
    # Test 1: Basic no-sandbox
    test_webgl_with_args([
        "--no-sandbox",
        "--disable-setuid-sandbox",
    ], "Basic")

    # Test 2: With enable-webgl
    test_webgl_with_args([
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--enable-webgl",
        "--ignore-gpu-blocklist",
    ], "Enable WebGL")

    # Test 3: With SwiftShader
    test_webgl_with_args([
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--use-gl=angle",
        "--use-angle=swiftshader-webgl",
    ], "SwiftShader ANGLE")

    # Test 4: With virtual display approach
    test_webgl_with_args([
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--use-gl=swiftshader",
        "--enable-webgl",
    ], "SwiftShader Direct")

    # Test 5: Disable GPU completely
    test_webgl_with_args([
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-gpu",
        "--enable-unsafe-swiftshader",
    ], "Unsafe SwiftShader")
