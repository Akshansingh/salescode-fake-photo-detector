import argparse
import os
import time
from glob import glob
import numpy as np
from predict import predict

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def collect(folder):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob(os.path.join(folder, ext)))
        paths.extend(glob(os.path.join(folder, ext.upper())))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=".", help="Folder containing real/ and screen/")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    items = []
    for p in collect(os.path.join(args.data, "real")):
        items.append((p, 0))
    for p in collect(os.path.join(args.data, "screen")):
        items.append((p, 1))

    if not items:
        raise SystemExit("No images found. Add images to real/ and screen/ first.")

    correct = 0
    latencies = []
    mistakes = []

    # warm-up
    try:
        predict(items[0][0])
    except Exception:
        pass

    for path, label in items:
        t0 = time.perf_counter()
        score = predict(path)
        dt = (time.perf_counter() - t0) * 1000
        pred = 1 if score >= args.threshold else 0
        latencies.append(dt)
        correct += int(pred == label)
        if pred != label:
            mistakes.append((path, label, score, pred))

    acc = correct / len(items) * 100
    print(f"Images evaluated: {len(items)}")
    print(f"Accuracy @ threshold {args.threshold:.2f}: {acc:.2f}%")
    print(f"Average latency: {np.mean(latencies):.2f} ms/image")
    print(f"P95 latency: {np.percentile(latencies, 95):.2f} ms/image")
    print("Cost per image: ₹0 / $0 when run on-device; no cloud API required.")

    if mistakes:
        print("\nMistakes:")
        for p, label, score, pred in mistakes[:20]:
            print(f"{p} | true={label} pred={pred} score={score:.4f}")
    else:
        print("No mistakes on this dataset.")


if __name__ == "__main__":
    main()
