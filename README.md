# 🎯 Cup and Ball Tracker using OpenCV

This project detects and tracks a colored cup in a video using **OpenCV** and **NumPy**. It includes two versions:
- A simple core script (`without-ui/`) using only OpenCV
- A full-featured GUI version (`with-ui/`) built with Tkinter

---

## 📂 Project Structure

```
cup-ball-tracker/
├── README.md
├── LICENSE
│
├── without-ui/
│   └── core_tracker.py
│
├── with-ui/
│   └── color_tracker_gui.py
│
└── example_video/
    └── warraaathoooose.mp4  (optional, for demonstration)
```

---

## 📥 Requirements

Python 3.8+

Install dependencies:

```bash
pip install opencv-python numpy Pillow
```

> For the GUI version, ensure `tkinter` is installed (usually comes with Python).

---

## 🎞️ Input Video Instructions

- Your video **must be resized to 640x360** (width x height).
- The video must contain **exactly 3 cups**, each one with a **different color**.
  - For example: red, green, and blue cups on a white table.
  - This allows the app to track the selected cup accurately.

- Place the video in the project folder or provide the path when using the GUI.

---

## 🚀 How to Run

### 🔧 Core version (OpenCV only)

```bash
python without-ui/core_tracker.py
```

### 🖥 GUI version (with Tkinter)

```bash
python with-ui/color_tracker_gui.py
```

You can select the video, adjust HSV sliders, and track a cup by clicking on it in the first frame.

---

## 🧠 How It Works

- Detects a colored cup using HSV filtering and user click
- Tracks the largest contour of that color across frames
- Displays the last known position of the cup: **Left**, **Middle**, or **Right**
- Supports real-time preview, logging, and simple interaction

---

## 🙏 Credits

**Author:** GarablueX  
- OpenCV and NumPy logic: ✅ built from scratch  
- UI framework: 🧱 adapted from a generic Tkinter template and integrated with the detection logic

---

## 📜 License

This project is licensed under the **MIT License**.
