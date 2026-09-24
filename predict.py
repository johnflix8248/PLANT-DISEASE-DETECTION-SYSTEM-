"""Command-line prediction (optional). Usage: python predict.py path/to/leaf.jpg"""
from __future__ import annotations

import sys
from predict_gui import interpret, load_artifacts, preprocess


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    model, class_names = load_artifacts()
    probs = model.predict(preprocess(sys.argv[1]), verbose=0)[0]
    pretty, status, conf, affected, _ = interpret(class_names, probs)
    print(f"Class      : {pretty}")
    print(f"Status     : {status}")
    print(f"Confidence : {conf:.2f}%")
    print(f"Affected % : {affected:.2f}%")
    for name, p in zip(class_names, probs):
        print(f"  {name}: {p*100:.2f}%")


if __name__ == "__main__":
    main()
