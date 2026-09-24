"""
Plant Disease Detection GUI (Spyder-friendly, Tkinter).

Run AFTER train_cnn.py so models/plant_disease_cnn.keras exists.

Upload a leaf image -> disease class, healthy vs affected,
affected/confidence percentage, and training accuracy graph.
"""
from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
IMG_SIZE = (128, 128)
HEALTHY_TOKENS = ("healthy", "health")


def load_artifacts():
    model_path = MODEL_DIR / "plant_disease_cnn.keras"
    names_path = MODEL_DIR / "class_names.json"
    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train_cnn.py first.\n"
            f"Expected: {model_path}"
        )
    model = tf.keras.models.load_model(model_path)
    class_names = json.loads(names_path.read_text())
    return model, class_names


def preprocess(path: str) -> np.ndarray:
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)


def interpret(class_names, probs):
    idx = int(np.argmax(probs))
    label = class_names[idx]
    confidence = float(probs[idx]) * 100.0
    is_healthy = any(tok in label.lower() for tok in HEALTHY_TOKENS)
    # Affected %: probability mass on non-healthy classes
    diseased_mass = 0.0
    for i, name in enumerate(class_names):
        if not any(tok in name.lower() for tok in HEALTHY_TOKENS):
            diseased_mass += float(probs[i])
    affected_pct = diseased_mass * 100.0
    status = "Healthy" if is_healthy else "Affected (diseased)"
    pretty = label.replace("_", " ").title()
    return pretty, status, confidence, affected_pct, idx


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Plant Disease Detection — CNN")
        self.geometry("980x640")
        self.configure(bg="#f4f7f4")
        self.model = None
        self.class_names = None
        self.photo = None
        self._build()
        self.after(100, self._load_model)

    def _build(self):
        header = tk.Frame(self, bg="#2e7d32")
        header.pack(fill="x")
        tk.Label(
            header,
            text="Plant Disease Detection Using Convolutional Neural Networks",
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg="#2e7d32",
            pady=12,
        ).pack()

        body = tk.Frame(self, bg="#f4f7f4")
        body.pack(fill="both", expand=True, padx=16, pady=12)

        left = tk.Frame(body, bg="#ffffff", bd=1, relief="solid")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(left, text="Leaf image", font=("Segoe UI", 12, "bold"), bg="#ffffff").pack(
            pady=(10, 4)
        )
        self.preview = tk.Label(
            left, text="No image selected", bg="#e8f5e9", width=48, height=18
        )
        self.preview.pack(padx=12, pady=8, fill="both", expand=True)
        btns = tk.Frame(left, bg="#ffffff")
        btns.pack(pady=10)
        ttk.Button(btns, text="Upload plant image", command=self.choose_image).pack(
            side="left", padx=6
        )
        ttk.Button(btns, text="Detect disease", command=self.detect).pack(side="left", padx=6)

        right = tk.Frame(body, bg="#ffffff", bd=1, relief="solid")
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))
        tk.Label(right, text="Result", font=("Segoe UI", 12, "bold"), bg="#ffffff").pack(
            pady=(10, 4)
        )
        self.result = tk.Text(right, height=10, font=("Consolas", 11), wrap="word")
        self.result.pack(fill="x", padx=12, pady=6)
        self.result.insert("end", "Load a trained model, then upload a leaf photo.")
        self.result.configure(state="disabled")

        tk.Label(
            right, text="Training accuracy graph", font=("Segoe UI", 12, "bold"), bg="#ffffff"
        ).pack(pady=(8, 4))
        self.graph_label = tk.Label(right, bg="#e8f5e9", text="Graph appears after training")
        self.graph_label.pack(padx=12, pady=8, fill="both", expand=True)
        self._show_graph()

        self.status = tk.Label(self, text="Starting...", anchor="w", bg="#c8e6c9")
        self.status.pack(fill="x", side="bottom")

    def _load_model(self):
        try:
            self.model, self.class_names = load_artifacts()
            self.status.configure(
                text=f"Model ready. Classes: {', '.join(self.class_names)}"
            )
        except Exception as exc:
            self.status.configure(text=str(exc))
            messagebox.showwarning("Model missing", str(exc))

    def _show_graph(self):
        path = MODEL_DIR / "accuracy_graph.png"
        if not path.exists():
            return
        img = Image.open(path)
        img.thumbnail((460, 280))
        self.graph_im = ImageTk.PhotoImage(img)
        self.graph_label.configure(image=self.graph_im, text="")

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Select plant leaf image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")],
        )
        if not path:
            return
        self.image_path = path
        img = Image.open(path).convert("RGB")
        img.thumbnail((420, 420))
        self.photo = ImageTk.PhotoImage(img)
        self.preview.configure(image=self.photo, text="")
        self.status.configure(text=f"Selected: {path}")

    def detect(self):
        if self.model is None:
            messagebox.showerror("Error", "Train the model first (run train_cnn.py).")
            return
        if not getattr(self, "image_path", None):
            messagebox.showinfo("Upload", "Please upload a plant image first.")
            return
        x = preprocess(self.image_path)
        probs = self.model.predict(x, verbose=0)[0]
        pretty, status, conf, affected, idx = interpret(self.class_names, probs)

        lines = [
            f"Predicted class : {pretty}",
            f"Plant status    : {status}",
            f"Model confidence: {conf:.2f}%",
            f"Affected %      : {affected:.2f}%  (probability of disease classes)",
            "",
            "Class probabilities:",
        ]
        for name, p in zip(self.class_names, probs):
            bar = "#" * int(round(p * 20))
            lines.append(f"  {name:22s} {p*100:6.2f}%  {bar}")

        self.result.configure(state="normal")
        self.result.delete("1.0", "end")
        self.result.insert("end", "\n".join(lines))
        self.result.configure(state="disabled")
        self.status.configure(text=f"{status} | {pretty} | confidence {conf:.1f}%")
        self._show_graph()


if __name__ == "__main__":
    App().mainloop()
