"""Regenerate every chart, the explainer GIF and the site from results JSON. No GPU needed.

    python -m kernels.assets            # charts + site
    python -m kernels.assets --gif      # also re-record the explainer GIF (needs Playwright)
"""

import sys

from kernels import ledger
from kernels.assets import charts, readme, site
from kernels.assets.style import ROOT

ledger.render()
made = charts.build_all()
if "--gif" in sys.argv:
    from kernels.assets import gif

    made += gif.capture()
made += site.build()
readme.build()
made.append(ROOT / "README.md")
for p in made:
    print(p.relative_to(ROOT))
