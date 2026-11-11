"""Screenshots of the results site (assets/screens/*.png) with Playwright.

    python -m kernels.assets.screens
"""

from __future__ import annotations

from kernels.assets.style import ROOT

OUT = ROOT / "assets" / "screens"
PAGES = ["index", "rmsnorm", "swiglu", "w4a16", "e2e", "cost"]


def capture(width: int = 1280, full_page: bool = True) -> list:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": 800}, device_scale_factor=1)
        for name in PAGES:
            page.goto((ROOT / "site" / f"{name}.html").as_uri())
            page.wait_for_load_state("networkidle")
            page.evaluate("document.fonts.ready")
            page.wait_for_timeout(500)
            path = OUT / f"{name}.png"
            page.screenshot(path=str(path), full_page=full_page)
            paths.append(path)
        # Mobile check: the overview at phone width.
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto((ROOT / "site" / "index.html").as_uri())
        mobile.wait_for_load_state("networkidle")
        mobile.wait_for_timeout(500)
        path = OUT / "index_mobile.png"
        mobile.screenshot(path=str(path), full_page=False)
        paths.append(path)
        browser.close()
    return paths


if __name__ == "__main__":
    for p in capture():
        print(p.relative_to(ROOT))
