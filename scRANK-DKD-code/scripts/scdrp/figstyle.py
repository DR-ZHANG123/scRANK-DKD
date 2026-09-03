"""Publication figure style."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
           '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']

MODEL_COLORS = {
    "scPair_LASSO": "#C44E52", "scDRP_DKD": "#4C72B0",
    "DeepPair": "#64B5CD", "DRGpair_LASSO": "#8C8C8C",
    "DEG_LASSO": "#55A868", "DEG_LogReg": "#937860",
    "DEG_RF": "#CCB974", "DEG_XGB": "#DA8BC3",
    "Module_LogReg": "#8172B3",
}

DISPLAY = {
    "scPair_LASSO": "scPair-LASSO", "scDRP_DKD": "DeepPair (prog)",
    "DeepPair": "DeepPair (no prog)", "DRGpair_LASSO": "DRGpair-LASSO",
    "DEG_LogReg": "DEG-LogReg", "DEG_LASSO": "DEG-LASSO", "DEG_RF": "DEG-RF",
    "DEG_XGB": "DEG-XGB", "Module_LogReg": "Module-LogReg",
}


def disp(m: str) -> str:
    return DISPLAY.get(m, m.replace("_", "-"))


COHORT_DISPLAY = {
    "GLOM_GSE30528": "GSE30528", "GLOM_GSE96804": "GSE96804",
    "GLOM_ERCB1": "ERCB-G1", "GLOM_ERCB2": "ERCB-G2",
    "GLOM_GSE1009": "GSE1009", "TUB_GSE30529": "GSE30529",
    "TUB_ERCB1": "ERCB-T1", "TUB_ERCB2": "ERCB-T2",
}


def cohort(c: str) -> str:
    return COHORT_DISPLAY.get(c, c.replace("GLOM_", "").replace("TUB_", ""))


MODEL_ORDER = ["DEG_LogReg", "DEG_LASSO", "DEG_RF", "DEG_XGB", "Module_LogReg",
               "DRGpair_LASSO", "scPair_LASSO", "DeepPair", "scDRP_DKD"]


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica",
                            "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "mathtext.sf": "Liberation Sans",
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def panel(ax, letter: str) -> None:
    ax.set_title(letter, fontweight="bold", fontsize=11, loc="left", pad=8)


def legend(ax, **kw):
    kw.setdefault("frameon", True)
    kw.setdefault("fancybox", False)
    kw.setdefault("edgecolor", "#CCCCCC")
    kw.setdefault("fontsize", 7)
    return ax.legend(**kw)


def save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  [fig] {name}.pdf / .png")
