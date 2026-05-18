# Sign Language Detector (PyTorch)

A real-time American Sign Language (ASL) recognition system that uses a webcam to detect hand gestures and convert them into text. The project ships with a Flask web app that streams the camera feed in the browser, predicts the signed letter live, and lets the user build full sentences with the help of word auto-completion.

---

## ✨ Features

- **Real-time gesture recognition** of ASL alphabet letters (A–Y, excluding J and Z which require motion) using a Convolutional Neural Network built with PyTorch.
- **Two ways to run it:**
  - `main.py` – a lightweight OpenCV desktop window.
  - `app.py` – a Flask web application with a browser UI.
- **Sentence builder** – press a trigger to append the current letter to a sentence.
- **Word auto-completion / suggestions** – the next word is predicted using the `autocomplete` library to speed up communication.
- **Pre-trained model included** (`model_trained.pt`), so no training is required to try it out.

---

## 🧠 How It Works

1. The webcam captures frames using OpenCV.
2. A fixed Region of Interest (a 230×230 green box in the top-left of the frame) is cropped from each frame — this is where the user places their hand.
3. The crop is resized to **28×28** and converted to **grayscale**, matching the format of the Sign Language MNIST dataset.
4. The tensor is passed through a **CNN** (`model.py` → `Net`):
   - Conv2d (1 → 80, 5×5) → BatchNorm → ReLU → MaxPool 2×2
   - Conv2d (80 → 80, 5×5) → BatchNorm → ReLU → MaxPool 2×2
   - Fully connected (1280 → 250) → ReLU
   - Fully connected (250 → 25) → LogSoftmax
5. The class with the highest softmax probability is mapped to a letter. If confidence is below 40%, "Nothing detected" is shown.
6. In the web app, hitting the trigger appends the predicted letter to a sentence, and `autocomplete.predict()` suggests the next word.

### Dataset
The model was trained on the [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist) dataset — 28×28 grayscale images of hand gestures, labelled 0–24 for the letters A–Y.

---

## 📁 Project Structure

```
Sign_Language_Detector-PyTorch-master/
├── app.py                 # Flask web application (browser UI + live stream)
├── main.py                # Standalone OpenCV desktop demo
├── model.py               # CNN architecture (PyTorch nn.Module)
├── model_trained.pt       # Pre-trained model weights
├── requirements.txt       # Python dependencies
├── templates/
│   ├── index.html         # Web UI
│   └── favicon.png
├── Images/                # Screenshots used in docs
└── AI Experiments/        # Jupyter notebooks (training/experiments)
```

---

## 🛠️ Setup Instructions

These steps work on **Windows**, **Linux**, and **macOS**. The only differences are how you create and activate the virtual environment.

### 1. Prerequisites

- **Python 3.8 – 3.10** recommended (the pinned `torch==1.4.0` in `requirements.txt` is older; on newer Python you may want a newer PyTorch — see the note below).
- **pip** (comes with Python).
- **Git** (to clone the repo).
- A **webcam** connected to your machine.

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/Sign_Language_Detector-PyTorch.git
cd Sign_Language_Detector-PyTorch
```

### 3. Create and activate a virtual environment

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on versions:** `requirements.txt` pins older versions (`torch==1.4.0`, `numpy==1.22.0`, `Flask==1.1.1`, `opencv_python==4.2.0.32`). If installation fails on your Python version, install the latest compatible versions instead:
>
> ```bash
> pip install torch torchvision numpy opencv-python Flask imutils autocomplete
> ```
>
> The code uses standard PyTorch / OpenCV APIs and works fine with current releases.

### 5. (macOS only) Grant camera permission

macOS will prompt for camera access the first time the script runs. Allow it under *System Settings → Privacy & Security → Camera*.

### 6. (Linux only) Make sure your webcam device exists

```bash
ls /dev/video*
```
If nothing shows up, plug in a webcam or check your drivers.

---

## ▶️ Running the Project

### Option A — Desktop OpenCV demo

```bash
python main.py
```
A window opens showing the webcam feed. Place your hand inside the green box; the predicted letter is drawn underneath. Press **`q`** to quit.

### Option B — Flask web application (recommended)

```bash
python app.py -i 0.0.0.0 -o 8080
```

Arguments:
- `-i / --ip`   IP to bind to (use `0.0.0.0` to expose on your network, or `127.0.0.1` for local only).
- `-o / --port` Port number (e.g. `8080`).

Then open your browser to:

```
http://localhost:8080/
```

You will see the live camera feed, the detected letter, the sentence being built, and a row of word suggestions.

---

## 🌐 Web App Endpoints

| Route | Purpose |
|-------|---------|
| `/` | Main UI |
| `/video_feed` | MJPEG stream of the annotated webcam |
| `/trigger` | Appends the currently detected letter to the sentence |
| `/char?character=<n>` | Picks suggestion number `n` (or `space` for a space) |
| `/suggestions` | JSON list of the current top-5 word suggestions |
| `/sentence` | JSON of the sentence built so far |

---

## 🧪 Notebooks

The `AI Experiments/` folder contains the Jupyter notebooks used while building and training the model:

- `sign_language_detector.ipynb` – main training pipeline.
- `Gesture_Recognition.ipynb`, `asl_detector.ipynb`, `Input_processing.ipynb` – exploratory work.

You can open them with:
```bash
pip install jupyter
jupyter notebook
```

---

## 🩹 Troubleshooting

- **"Could not open camera" / black frame** – another application is using the webcam, or the wrong device index. Edit `VideoStream(src=0)` in `app.py` (or `cv2.VideoCapture(0)` in `main.py`) and try `src=1`, `src=2`, …
- **`torch.load` warning / error about `weights_only`** – you are on a much newer PyTorch. The code already passes `weights_only=False` in `app.py`; do the same in `main.py` if needed: `torch.load('model_trained.pt', map_location='cpu', weights_only=False)`.
- **`autocomplete` install fails** – on some Python versions you may need: `pip install autocomplete --no-build-isolation`.
- **Port already in use** – pick a different port via `-o`.
- **Low accuracy** – make sure your hand fills the green box, the background is plain, and lighting is even. The model was trained on clean 28×28 grayscale crops, so it is sensitive to noise.

---

## 💡 Use Cases

- Helping deaf and mute individuals communicate with people who don't know sign language.
- Inclusive classrooms where signers can ask questions in real time.
- Educational tool for learning the ASL alphabet.

---

## 📜 License

This project is released for educational purposes. Please credit the original authors if you reuse it.
