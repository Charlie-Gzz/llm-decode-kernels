"""Export site/explainer.html to assets/explainer.gif (and a few PNG stills) with Playwright.

The page exposes render(t) in ?capture mode, so frames are exact, not timing-dependent.
"""

from __future__ import annotations

import io

from PIL import Image

from kernels.assets.style import ROOT

FRAMES = 100  # 10 s loop at 10 fps
WIDTH, HEIGHT = 1200, 675


def capture(out_dir=ROOT / "assets", frames: int = FRAMES) -> list:
    from playwright.sync_api import sync_playwright

    page_url = (ROOT / "site" / "explainer.html").as_uri() + "?capture"
    images = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(page_url)
        page.wait_for_load_state("networkidle")
        page.evaluate("document.fonts.ready")
        svg = page.locator("#stage")
        for i in range(frames):
            page.evaluate(f"render({i / frames})")
            images.append(Image.open(io.BytesIO(svg.screenshot())).convert("RGB"))
        browser.close()

    stills = {"explainer_load.png": 0.15, "explainer_unpack.png": 0.35,
              "explainer_dequant.png": 0.55, "explainer_mac.png": 0.8}
    for name, t in stills.items():
        images[int(t * frames)].save(out_dir / name)

    small = [im.resize((960, 540), Image.LANCZOS) for im in images]
    pal = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
           for im in small]
    pal[0].save(out_dir / "explainer.gif", save_all=True, append_images=pal[1:],
                duration=100, loop=0, optimize=True, disposal=1)
    return [out_dir / "explainer.gif", *(out_dir / n for n in stills)]


if __name__ == "__main__":
    for path in capture():
        print(path, path.stat().st_size // 1024, "KB")
