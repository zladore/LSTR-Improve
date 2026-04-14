import os
import sys
import argparse
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from metrics import perframe_average_precision
from postprocessing import postprocessing


THUMOS_CLASS_NAMES = [
    "Background",
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking",
    "Ambiguous",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True,
                        help="Path to best_test_predictions.csv")
    parser.add_argument("--ignore_index", type=int, default=21)
    parser.add_argument("--metrics", type=str, default="AP", choices=["AP", "cAP"])
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)

    score_cols = [f"score_cls_{i}" for i in range(22)]
    for c in ["target"] + score_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in pred csv: {c}")

    targets = df["target"].astype(int).to_numpy()
    prediction = df[score_cols].to_numpy(dtype=np.float32)

    # one-hot GT, shape [N, 22]
    ground_truth = np.zeros((len(targets), 22), dtype=np.int32)
    valid = (targets >= 0) & (targets < 22)
    ground_truth[np.arange(len(targets))[valid], targets[valid]] = 1

    result = perframe_average_precision(
        ground_truth=ground_truth,
        prediction=prediction,
        class_names=THUMOS_CLASS_NAMES,
        ignore_index=args.ignore_index,
        metrics=args.metrics,
        postprocessing=postprocessing("THUMOS"),
    )

    print("=" * 80)
    print("LSTR official-style evaluation on THUMOS")
    print(f"metrics   : {args.metrics}")
    print(f"ignore_idx: {args.ignore_index}")
    print(f"mean_AP   : {result['mean_AP']:.6f}")
    print("-" * 80)
    print("per_class_AP:")
    for k, v in result["per_class_AP"].items():
        print(f"{k:20s}: {v:.6f}")


if __name__ == "__main__":
    main()