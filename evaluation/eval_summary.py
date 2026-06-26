import json
import os

def load_json(path):
    if not os.path.exists(path):
        print(f"  [WARNING] File tidak ditemukan: {path}")
        return None
    with open(path) as f:
        return json.load(f)

def main():
    os.makedirs("eval_output", exist_ok=True)

    clf = load_json("eval_output/classifier_overall.json")
    reg = load_json("eval_output/regressor_overall.json")

    lines = []

    header = f"""
{'=' * 50}
  EVALUATION SUMMARY
{'=' * 50}"""
    print(header)
    lines.append(header.strip())

    # ===== CLASSIFIER =====
    print("\n" + "-" * 48)
    print("  DESTINATION CLASSIFIER (XGBoost)")
    print("-" * 48)
    lines.append("")
    lines.append("-" * 48)
    lines.append("  DESTINATION CLASSIFIER (XGBoost)")
    lines.append("-" * 48)

    if clf:
        items = [
            ("Accuracy", f"{clf['accuracy']*100:.2f}%"),
            ("Top-3 Accuracy", f"{clf['top3_accuracy']*100:.2f}%"),
            ("Top-5 Accuracy", f"{clf['top5_accuracy']*100:.2f}%"),
            ("Macro Avg Precision", f"{clf['macro_avg_precision']:.4f}"),
            ("Macro Avg Recall", f"{clf['macro_avg_recall']:.4f}"),
            ("Macro Avg F1", f"{clf['macro_avg_f1']:.4f}"),
            ("Weighted Avg Precision", f"{clf['weighted_avg_precision']:.4f}"),
            ("Weighted Avg Recall", f"{clf['weighted_avg_recall']:.4f}"),
            ("Weighted Avg F1", f"{clf['weighted_avg_f1']:.4f}"),
            ("Samples Tested", f"{clf['test_samples']:,}"),
            ("Classes", str(clf['n_classes'])),
            ("Confidence (Correct)", f"{clf['confidence_correct']:.4f}"),
            ("Confidence (Wrong)", f"{clf['confidence_wrong']:.4f}"),
        ]
    else:
        items = [("Status", "Belum dijalankan")]

    for key, value in items:
        padding = 26 - len(key)
        line = f"  {key}{'.' * padding} {value}"
        print(line)
        lines.append(line)

    # ===== REGRESSOR =====
    print("\n" + "-" * 48)
    print("  ETA REGRESSOR (XGBoost)")
    print("-" * 48)
    lines.append("")
    lines.append("-" * 48)
    lines.append("  ETA REGRESSOR (XGBoost)")
    lines.append("-" * 48)

    if reg:
        items = [
            ("MAE", f"{reg['mae_minutes']:.2f} menit"),
            ("RMSE", f"{reg['rmse_minutes']:.2f} menit"),
            ("R2 Score", f"{reg['r2_score']:.4f}"),
            ("Error < 1 menit", f"{reg['error_lt_1min_pct']:.1f}%"),
            ("Error < 5 menit", f"{reg['error_lt_5min_pct']:.1f}%"),
            ("Error < 10 menit", f"{reg['error_lt_10min_pct']:.1f}%"),
            ("Error < 15 menit", f"{reg['error_lt_15min_pct']:.1f}%"),
            ("Error < 30 menit", f"{reg['error_lt_30min_pct']:.1f}%"),
            ("Samples Tested", f"{reg['test_samples']:,}"),
        ]
    else:
        items = [("Status", "Belum dijalankan")]

    for key, value in items:
        padding = 26 - len(key)
        line = f"  {key}{'.' * padding} {value}"
        print(line)
        lines.append(line)

    footer = f"""
{'=' * 50}
  Output files:
    eval_output/overall_summary.txt
    eval_output/classifier_overall.json
    eval_output/regressor_overall.json
{'=' * 50}"""
    print(footer)
    lines.append("")
    lines.append(footer.strip())

    with open("eval_output/overall_summary.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\n  Summary saved: eval_output/overall_summary.txt")

if __name__ == "__main__":
    main()
