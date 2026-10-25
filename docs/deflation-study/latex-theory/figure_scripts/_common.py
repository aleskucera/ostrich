"""Shared figure output: figures/<script stem>.pdf, stamped with the script and git commit."""

import subprocess
from pathlib import Path

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def save_fig(fig, script):
    """Save `fig` as figures/<stem of script>.pdf with the script path and git state in the PDF Subject."""
    script = Path(script).resolve()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=script.parent, text=True).strip())
    rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", str(script)], cwd=root, text=True).strip()
    out = FIGURES / f"{script.stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Subject": f"{script.relative_to(root)} @ {rev}{'+dirty' if dirty else ''}"})
    return out
