"""Command-line training.

    python train.py                 train on data/StudentPerformanceFactors.csv
    python train.py --cv 5          also run 5-fold cross-validation
    python train.py --data my.csv   train on another CSV with the same columns
"""
import argparse

from ml import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", default=str(pipeline.DEFAULT_DATASET))
    parser.add_argument("--cv", type=int, default=0,
                        help="number of cross-validation folds (0 to skip)")
    args = parser.parse_args()

    def progress(stage, percent, message):
        print(f"[{percent:3d}%] {message}", flush=True)

    _, m = pipeline.train(args.data, progress=progress, cv_folds=args.cv)
    d = m["dataset"]
    print()
    print(f"Clean records        {d['cleaning']['rows_clean']:,}")
    print(f"Training (balanced)  {d['rows_train_balanced']:,} "
          f"({d['rows_synthetic']:,} synthetic)")
    print(f"Held-out test        {d['rows_test']:,}")
    print()
    print(f"{'Model':<24}{'Accuracy':>9}{'Precision':>11}{'Recall':>8}{'F1':>8}")
    for key, label in [("single_decision_tree", "Single decision tree"),
                       ("random_forest", "Random Forest"),
                       ("gradient_boosting", "Gradient Boosting"),
                       ("ensemble", "Voting ensemble")]:
        r = m["results"][key]
        print(f"{label:<24}{r['accuracy']:>9.4f}{r['precision_macro']:>11.4f}"
              f"{r['recall_macro']:>8.4f}{r['f1_macro']:>8.4f}")
    if m["cross_validation"]:
        cv = m["cross_validation"]
        print(f"\n{cv['folds']}-fold CV accuracy  {cv['accuracy_mean']:.4f} "
              f"(+/- {cv['accuracy_std']:.4f})")
    print(f"\nSaved to {pipeline.MODEL_PATH}")


if __name__ == "__main__":
    main()
