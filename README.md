# Plant Disease Detection Using Convolutional Neural Networks

Spyder / Python project: download a public plant-disease dataset, train a CNN, then upload a leaf image to see whether the plant is healthy or affected, the disease name, confidence / affected percentage, and the training accuracy graph.

## Dataset (downloaded automatically)

**iBean** — Makerere AI Lab bean leaf images (public Google Cloud storage, no Kaggle login):

| Class | Meaning |
| --- | --- |
| `angular_leaf_spot` | Diseased |
| `bean_rust` | Diseased |
| `healthy` | Not affected |

URLs used by `download_dataset.py`:

- https://storage.googleapis.com/ibean/train.zip
- https://storage.googleapis.com/ibean/validation.zip
- https://storage.googleapis.com/ibean/test.zip

After download, files stay in the local `dataset/` folder (kept for reuse).

## Spyder steps

1. Open this folder as the working directory.
2. Install packages (IPython console or system terminal):

```text
pip install -r requirements.txt
```

3. Run **`download_dataset.py`** (or skip — training calls it automatically).
4. Run **`train_cnn.py`**. This trains the CNN and writes:

   - `models/plant_disease_cnn.keras`
   - `models/accuracy_graph.png`
   - `models/loss_graph.png`
   - `models/class_names.json`

5. Run **`predict_gui.py`**. Click **Upload plant image**, then **Detect disease**.

Command-line check:

```text
python predict.py dataset/test/healthy/<some_file>.jpg
```

## Outputs shown in the GUI

- Predicted disease class
- **Healthy** vs **Affected**
- Model confidence (%)
- **Affected %** = total probability of the two disease classes
- Training **accuracy graph** (from `models/accuracy_graph.png`)

## Project files

| File | Role |
| --- | --- |
| `download_dataset.py` | Fetch and extract the online dataset |
| `train_cnn.py` | Build/train CNN, save model + graphs |
| `predict_gui.py` | Tkinter upload + results window |
| `predict.py` | CLI prediction |
| `requirements.txt` | TensorFlow, Pillow, matplotlib |

Training on CPU typically takes a few minutes (small 128×128 CNN, 12 epochs with early stopping).
