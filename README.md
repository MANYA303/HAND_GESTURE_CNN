# Hand Gesture Recognition Using CNN

A complete computer-vision project that recognises **hand gestures from images** using a
Convolutional Neural Network (CNN) built with TensorFlow/Keras, with an interactive
**Streamlit** web app for real-time predictions.

---

## Features

| Feature | Detail |
|---|---|
| CNN Architecture | 4 conv blocks (32→64→128→256 filters) + GAP + Dense head |
| Input | 64 × 64 RGB images |
| Supported gestures | 10 classes from the LeapGestRecog dataset |
| Data augmentation | Rotation, shift, zoom, horizontal flip |
| Training callbacks | Early stopping, model checkpointing, LR reduction on plateau |
| Evaluation | Accuracy, precision, recall, F1-score, confusion matrix |
| Streamlit UI | Image upload, gesture prediction, confidence bar chart |

---

## Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras** – CNN model training and inference
- **OpenCV** – image loading and preprocessing
- **NumPy** – numerical operations
- **Matplotlib / Seaborn** – training curves and confusion matrix
- **Streamlit** – interactive web application
- **Scikit-learn** – metrics (classification report, confusion matrix)

---

## Dataset: LeapGestRecog

| Property | Value |
|---|---|
| Source | [Kaggle – gti-upm/leapgestrecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) |
| Samples | ~20 000 grayscale frames across 10 subjects |
| Classes | 10 gesture categories |
| License | CC BY-SA 4.0 |

### Gesture Classes

| Folder name | Gesture |
|---|---|
| `01_palm` | Open palm |
| `02_l` | L-shape |
| `03_fist` | Fist |
| `04_fist_moved` | Moved fist |
| `05_thumb` | Thumbs up |
| `06_index` | Index finger |
| `07_ok` | OK / okay |
| `08_palm_moved` | Moved palm |
| `09_c` | C-shape |
| `10_down` | Thumbs down |

---

## Project Folder Structure

```
hand-gesture-recognition-cnn/
├── app.py                    # Streamlit application
├── train.py                  # Training entry-point
├── preprocess.py             # Image loading, augmentation, split utilities
├── model.py                  # CNN architecture factory
├── evaluate.py               # Full evaluation (report + confusion matrix)
├── download_dataset.py       # Kaggle dataset downloader / organiser
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── dataset/                  # <class_name>/<image> structure goes here
├── models/
│   ├── gesture_model.keras   # Best saved model (after training)
│   ├── label_map.json        # Ordered list of class names
│   ├── X_test.npy            # Cached test images (for evaluate.py)
│   └── y_test.npy            # Cached test labels
├── outputs/
│   ├── accuracy.png          # Training/validation accuracy curve
│   ├── loss.png              # Training/validation loss curve
│   ├── confusion_matrix.png  # Confusion matrix heatmap
│   └── classification_report.txt
└── utils/
    ├── __init__.py
    └── prediction.py         # Single-image inference helper
```

---

## Installation (Windows)

### 1. Clone / download the project

```
git clone <repo-url>
cd hand-gesture-recognition-cnn
```

Or simply place all files in a folder called `hand-gesture-recognition-cnn`.

### 2. Create a virtual environment

```
python -m venv .venv
```

### 3. Activate the virtual environment

```
.venv\Scripts\activate
```

### 4. Install dependencies

```
pip install -r requirements.txt
```

> **GPU note:** To use a CUDA GPU, install the matching `tensorflow[and-cuda]` package
> after the step above:
> ```
> pip install tensorflow[and-cuda]
> ```

---

## Dataset Setup

### Option A – Automatic download (recommended)

1. Create a free account at [https://www.kaggle.com](https://www.kaggle.com).
2. Go to **Settings → Account → API → Create New Token** – this downloads `kaggle.json`.
3. Place `kaggle.json` at `C:\Users\<YourName>\.kaggle\kaggle.json`.
4. Run:
   ```
   python download_dataset.py
   ```
   This downloads, extracts, and organises all images automatically.

### Option B – Manual download

1. Download the dataset from:
   [https://www.kaggle.com/datasets/gti-upm/leapgestrecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
2. Extract the zip file.
3. Inside the extracted folder you will find sub-folders like
   `00/01_palm/`, `01/01_palm/`, … (10 subjects × 10 classes).
4. Create a `dataset/` folder in the project root and copy the images in this structure:
   ```
   dataset/
     01_palm/
       frame_00_...png
       frame_01_...png
       ...
     02_l/
       ...
     03_fist/
       ...
     (etc.)
   ```
   Each class folder should contain all images for that gesture (across all subjects).

---

## Training

```
python train.py
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `dataset` | Path to dataset root |
| `--epochs` | `30` | Maximum training epochs |
| `--batch-size` | `32` | Mini-batch size |

Example with custom settings:
```
python train.py --epochs 50 --batch-size 64
```

After training, you will find:
- `models/gesture_model.keras` – best model weights
- `models/label_map.json` – class name list
- `outputs/accuracy.png` – accuracy curve
- `outputs/loss.png` – loss curve

---

## Evaluation

```
python evaluate.py
```

Generates:
- A full classification report (precision / recall / F1 per class)
- `outputs/confusion_matrix.png`
- `outputs/classification_report.txt`

---

## Streamlit Application

```
python -m streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501).

**Usage:**
1. Upload a hand gesture image (JPG / JPEG / PNG).
2. Click **Predict Gesture**.
3. See the predicted gesture name, confidence percentage, and a probability bar chart.

---

## Troubleshooting

### `FileNotFoundError: Dataset directory not found`
→ Run `python download_dataset.py` or manually place images in `dataset/<class>/`.

### `FileNotFoundError: Model not found`
→ Run `python train.py` before launching the app or running evaluate.py.

### `ModuleNotFoundError: No module named 'tensorflow'`
→ Make sure you activated the virtual environment:
```
.venv\Scripts\activate
pip install -r requirements.txt
```

### `kaggle.json` missing when running `download_dataset.py`
→ Follow the Kaggle API token setup steps in the **Dataset Setup** section above.

### GPU not detected / slow training
→ Install CUDA-enabled TensorFlow:
```
pip install tensorflow[and-cuda]
```
Check CUDA availability with:
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Streamlit: `OSError: [Errno 10048]` (port already in use)
→ Run on a different port:
```
python -m streamlit run app.py --server.port 8502
```

### Images look too dark / washed out in the app
→ The model expects RGB images normalised to [0, 1]. The app handles this automatically;
   ensure you upload standard JPG/PNG photos.

---

## Quick-Start Summary

```
# 1. Set up environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Download dataset
python download_dataset.py

# 3. Train the model
python train.py

# 4. Evaluate (optional)
python evaluate.py

# 5. Launch the app
python -m streamlit run app.py
```
