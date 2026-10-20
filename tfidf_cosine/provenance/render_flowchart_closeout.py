from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_materials"
FONT_PATH = ROOT / "report_materials/echarts_v6/ReportCJK.font"
FONT = font_manager.FontProperties(fname=str(FONT_PATH))

NODES = [
    (0.11, 0.69, 0.16, 0.13, "内部 train\n句子1、句子2", "input"),
    (0.32, 0.69, 0.19, 0.13, "分别拟合三个 TF-IDF\n词表与 IDF", "box"),
    (0.11, 0.35, 0.18, 0.16, "内部 validation\n仅用于比较与\n选择", "input"),
    (0.32, 0.35, 0.19, 0.13, "使用已拟合向量器\ntransform", "box"),
    (0.55, 0.35, 0.21, 0.13, "306 组验证比较\n选择 char_1_2 / 0.28", "box"),
    (0.55, 0.69, 0.21, 0.13, "冻结获选模型\nthreshold = 0.28", "freeze"),
    (0.81, 0.35, 0.17, 0.13, "最终 evaluation\n句子对", "input"),
    (0.81, 0.69, 0.17, 0.13, "固定配置预测\n评分与错误分析", "freeze"),
]

STYLE = {
    "input": ("#f5f5f4", "#57534e"),
    "box": ("#eef4ff", "#2563eb"),
    "freeze": ("#ecfdf5", "#059669"),
}


def add_node(ax, x, y, w, h, label, kind):
    face, edge = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.012",
                                facecolor=face, edgecolor=edge, linewidth=1.6))
    ax.text(x, y, label, ha="center", va="center", fontproperties=FONT,
            fontsize=11, color="#111827", linespacing=1.35)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color="#374151",
                                 connectionstyle="arc3,rad=0"))


def render(path, fonttype):
    matplotlib.rcParams["svg.fonttype"] = fonttype
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, "TF-IDF + 余弦相似度分析流程", ha="center", va="center",
            fontproperties=FONT, fontsize=18, color="#111827")
    for node in NODES:
        add_node(ax, *node)
    arrow(ax, (0.19, 0.69), (0.225, 0.69))
    arrow(ax, (0.20, 0.35), (0.225, 0.35))
    arrow(ax, (0.32, 0.62), (0.32, 0.43))
    arrow(ax, (0.415, 0.35), (0.445, 0.35))
    arrow(ax, (0.55, 0.42), (0.55, 0.62))
    arrow(ax, (0.655, 0.69), (0.725, 0.69))
    arrow(ax, (0.81, 0.42), (0.81, 0.62))
    arrow(ax, (0.76, 0.69), (0.81, 0.69))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12, dpi=220,
                transparent=False)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render method flowcharts into a new output directory.")
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--font', type=Path, default=FONT_PATH)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    FONT_PATH = args.font.resolve()
    FONT = font_manager.FontProperties(fname=str(FONT_PATH))
    if not FONT_PATH.exists():
        raise FileNotFoundError(FONT_PATH)
    render(OUT / "方法流程图.svg", "none")
    render(OUT / "方法流程图_跨设备路径版.svg", "path")
    render(OUT / "方法流程图.png", "none")
    print({"font": str(FONT_PATH), "png": str(OUT / "方法流程图.png")})
