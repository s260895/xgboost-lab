"""Inspect a real XGBoost tree: split conditions, leaf weights, gain, cover.

This is a *learning* tool, not part of the training pipeline. It trains a
deliberately tiny, shallow model (so the first tree is legible) and shows you:

  1. A text rendering of tree #0 — every split as `feature < threshold`, every
     leaf as its weight w* = -G/(H+lambda) (notes 02/04).
  2. XGBoost's own dump (with stats), where `gain` is the split's Gain and `cover`
     is the sum of hessians H in that node (note 02).
  3. A feature-importance plot saved to disk.
  4. (Best effort) a graphviz-rendered tree image, if graphviz is installed.

How to read it, tied to the math:
  * a node `[mean radius < 16.8] (gain=..., cover=...)` chose that split because it
    maximized Gain = 1/2[ G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G^2/(H+λ) ] - γ.
  * `cover` is H = Σ h_i for the rows reaching the node (for logloss h = p(1-p)).
  * a `leaf = +0.21` is w* = -G/(H+λ): the score that leaf adds to F(x), before
    the learning-rate shrinkage.

Run:  python scripts/inspect_tree.py            (tree 0, depth 3)
      python scripts/inspect_tree.py --tree 1 --max-depth 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xgboost as xgb

from xgblearn.config import PROJECT_ROOT, ModelConfig, set_global_seed
from xgblearn.data.loaders import load_dataset
from xgblearn.data.splits import train_val_test_split
from xgblearn.models.train import train_native


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tree", type=int, default=0, help="Which tree to inspect.")
    p.add_argument("--max-depth", type=int, default=3, help="Tree depth (keep small).")
    p.add_argument("--rounds", type=int, default=10, help="Number of boosting rounds.")
    p.add_argument("--out-dir", default="models", help="Where to save figures.")
    return p.parse_args()


def render_tree(df: pd.DataFrame, root_id: str) -> None:
    """Pretty-print one tree from booster.trees_to_dataframe()."""
    by_id = {row.ID: row for row in df.itertuples(index=False)}

    def walk(node_id: str, prefix: str, branch: str) -> None:
        row = by_id[node_id]
        if row.Feature == "Leaf":
            # For leaf rows, the `Gain` column holds the leaf weight w*.
            print(f"{prefix}{branch}leaf = {row.Gain:+.4f}   (cover/H={row.Cover:.2f})")
            return
        cond = f"{row.Feature} < {row.Split:.4g}"
        print(
            f"{prefix}{branch}[{cond}]   (gain={row.Gain:.3f}, cover/H={row.Cover:.2f})"
        )
        child_prefix = prefix + ("    " if branch else "")
        # `Yes` is taken when the condition is TRUE (feature < split).
        walk(row.Yes, child_prefix, "yes-> ")
        walk(row.No, child_prefix, "no--> ")

    walk(root_id, "", "")


def main() -> int:
    args = parse_args()
    set_global_seed(42)

    ds = load_dataset("breast_cancer")
    splits = train_val_test_split(
        ds.X, ds.y, test_size=0.2, val_size=0.2, stratify=True, seed=42
    )

    # Tiny, shallow, CPU model so the tree is small and reproducible.
    cfg = ModelConfig(
        objective="binary:logistic",
        eval_metric=["logloss"],
        device="cpu",
        tree_method="hist",
        num_boost_round=args.rounds,
        early_stopping_rounds=None,
        seed=42,
        params={"learning_rate": 0.3, "max_depth": args.max_depth},
    )
    booster: xgb.Booster = train_native(splits, cfg).model

    all_trees = booster.trees_to_dataframe()
    n_trees = int(all_trees["Tree"].max()) + 1
    if not 0 <= args.tree < n_trees:
        raise SystemExit(f"--tree must be in [0, {n_trees - 1}]; got {args.tree}.")

    print(f"\nModel: {n_trees} trees, max_depth={args.max_depth}, lr=0.3 (CPU)\n")
    print(f"=== Tree #{args.tree}: readable rendering ===")
    tree_df = all_trees[all_trees["Tree"] == args.tree]
    render_tree(tree_df, root_id=f"{args.tree}-0")

    print(f"\n=== Tree #{args.tree}: XGBoost dump (with stats) ===")
    print(booster.get_dump(with_stats=True)[args.tree])

    # Feature-importance plot (pure matplotlib; no graphviz needed).
    out_dir = (
        PROJECT_ROOT / args.out_dir
        if not Path(args.out_dir).is_absolute()
        else Path(args.out_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")  # headless backend; we save to file
    import matplotlib.pyplot as plt

    ax = xgb.plot_importance(
        booster, importance_type="gain", max_num_features=10, height=0.5
    )
    ax.set_title("Feature importance (gain) — top 10")
    plt.tight_layout()
    imp_path = out_dir / "feature_importance.png"
    plt.savefig(imp_path, dpi=120)
    plt.close()
    print(f"\nSaved feature-importance plot -> {imp_path}")

    # Best-effort graphviz tree image.
    try:
        graph = xgb.to_graphviz(booster, num_trees=args.tree)
        tree_path = out_dir / f"tree_{args.tree}"
        graph.render(filename=str(tree_path), format="png", cleanup=True)
        print(f"Saved tree image          -> {tree_path}.png")
    except Exception as exc:  # graphviz (python pkg or `dot` binary) not present
        print(
            f"\n(Skipped graphviz tree image: {type(exc).__name__}. "
            "The text rendering above is the same tree.)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
