import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def parse_log(file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    pattern = re.compile(
        r"time=([\d.]+)s\s+llm_patch=([\d.]+)\s+img_patch=([\d.]+)"
    )
    matches = pattern.findall(text)
    if not matches:
        raise ValueError("No matching lines found.")
    arr = np.array(matches, dtype=np.float64)
    times, llm, img = arr[:, 0], arr[:, 1], arr[:, 2]
    print(f"Parsed {len(times):,} entries.")
    return times, llm, img


def fit_linear(img, llm, times):
    X = np.column_stack([img, llm])
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X, times)
    pred = model.predict(X)
    r2 = r2_score(times, pred)
    rmse = np.sqrt(np.mean((times - pred) ** 2))
    a, b = model.coef_
    c = model.intercept_
    print(f"\nf(img, llm) = {a:.6e} * img + {b:.6e} * llm + {c:.6f}")
    print(f"R² = {r2:.4f}  RMSE = {rmse:.4f}s")
    return model, a, b, c


def plot(img, llm, times, model, output_path):
    X = np.column_stack([img, llm])
    pred = model.predict(X)

    fig = plt.figure(figsize=(18, 5))

    # 图1：3D 散点图
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    sc = ax0.scatter(img, llm, times, c=times, cmap="viridis", s=8, alpha=0.6)
    fig.colorbar(sc, ax=ax0, shrink=0.5).set_label("Actual time (s)")
    ax0.set_title("Ground truth (3D)")
    ax0.set_xlabel("Image patch")
    ax0.set_ylabel("LLM patch")
    ax0.set_zlabel("Time (s)")

    # 图2：预测 vs 实际
    ax1 = fig.add_subplot(1, 3, 2)
    lim = [min(times.min(), pred.min()) * 0.97, max(times.max(), pred.max()) * 1.03]
    ax1.scatter(times, pred, s=10, alpha=0.5, edgecolors="none")
    ax1.plot(lim, lim, "r--", lw=1.5, label="ideal")
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_title(f"Linear fit  R²={r2_score(times, pred):.4f}")
    ax1.set_xlabel("Actual time (s)")
    ax1.set_ylabel("Predicted time (s)")
    ax1.legend()

    # 图3：预测曲面等高线
    ax2 = fig.add_subplot(1, 3, 3)
    ig = np.linspace(img.min(), img.max(), 80)
    lg = np.linspace(llm.min(), llm.max(), 80)
    IG, LG = np.meshgrid(ig, lg)
    ZG = model.predict(np.column_stack([IG.ravel(), LG.ravel()])).reshape(IG.shape)
    cs = ax2.contourf(IG, LG, ZG, levels=25, cmap="viridis")
    fig.colorbar(cs, ax=ax2).set_label("Predicted time (s)")
    ax2.set_title("Predicted surface")
    ax2.set_xlabel("Image patch")
    ax2.set_ylabel("LLM patch")

    residuals = times - pred
    plt.suptitle(f"MAE={np.abs(residuals).mean():.4f}s  |  RMSE={residuals.std():.4f}s", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    input_file   = sys.argv[1] if len(sys.argv) > 1 else "log.txt"
    output_image = sys.argv[2] if len(sys.argv) > 2 else "fit_result.png"

    times, llm, img = parse_log(input_file)
    model, a, b, c = fit_linear(img, llm, times)
    plot(img, llm, times, model, output_image)

    print(f"\nExample: f(img=12000, llm=15000) = {a*12000 + b*15000 + c:.3f}s")