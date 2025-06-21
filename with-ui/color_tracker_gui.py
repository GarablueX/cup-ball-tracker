import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import threading
import queue
import os
import time
import json


class ColorTrackerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Color ditection 0.0.-1")
        self.root.geometry("1400x900")

        # Video processing variables
        self.width = 640
        self.height = 360
        self.X = 0
        self.Y = 0
        self.Lhue = 0
        self.Hhue = 0
        self.Lsat = 0
        self.Hsat = 0
        self.Lval = 0
        self.Hval = 0
        self.click = False
        self.cap = None
        self.running = False
        self.paused = False

        # Enhanced features
        self.tracking_history = []
        self.confidence_threshold = 500  # Minimum area for reliable detection
        self.smoothing_factor = 0.7  # For position smoothing
        self.last_position = None

        # Thread communication
        self.message_queue = queue.Queue()
        self.frame_count = 0
        self.total_frames = 0
        self.fps_counter = 0
        self.last_time = time.time()

        # Recording functionality
        self.recording = False
        self.video_writer = None
        self.output_path = "tracked_output.mp4"

        self.setup_ui()
        self.process_messages()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Video path and controls
        path_frame = ttk.LabelFrame(main_frame, text="Video Controls", padding="5")
        path_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        path_frame.columnconfigure(1, weight=1)

        ttk.Label(path_frame, text="Path:").grid(row=0, column=0, padx=(0, 5))
        self.path_var = tk.StringVar(value="warraaathoooose.mp4")
        self.path_entry = ttk.Entry(path_frame, textvariable=self.path_var)
        self.path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))

        # Control buttons
        ttk.Button(path_frame, text="Browse", command=self.browse_video).grid(row=0, column=2, padx=(0, 5))
        self.start_btn = ttk.Button(path_frame, text="Start", command=self.start_tracking)
        self.start_btn.grid(row=0, column=3, padx=(0, 5))
        self.pause_btn = ttk.Button(path_frame, text="Pause", command=self.toggle_pause, state='disabled')
        self.pause_btn.grid(row=0, column=4, padx=(0, 5))
        self.stop_btn = ttk.Button(path_frame, text="Stop", command=self.stop_tracking, state='disabled')
        self.stop_btn.grid(row=0, column=5, padx=(0, 5))

        # Second row of controls
        ttk.Button(path_frame, text="Clear Output", command=self.clear_output).grid(row=1, column=0, pady=(5, 0))
        self.record_btn = ttk.Button(path_frame, text="Start Recording", command=self.toggle_recording)
        self.record_btn.grid(row=1, column=1, pady=(5, 0), padx=(5, 0), sticky=tk.W)
        ttk.Button(path_frame, text="Save Settings", command=self.save_settings).grid(row=1, column=2, pady=(5, 0),
                                                                                      padx=(5, 0))
        ttk.Button(path_frame, text="Load Settings", command=self.load_settings).grid(row=1, column=3, pady=(5, 0),
                                                                                      padx=(5, 0))
        ttk.Button(path_frame, text="Export Data", command=self.export_tracking_data).grid(row=1, column=4, pady=(5, 0),
                                                                                           padx=(5, 0))

        # Video display area
        video_frame = ttk.LabelFrame(main_frame, text="Video Display", padding="5")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        # Video canvas with scrollbar for seeking
        canvas_frame = ttk.Frame(video_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(canvas_frame, bg="black", width=640, height=360)
        self.video_canvas.grid(row=0, column=0, pady=(0, 10))
        self.video_canvas.bind("<Button-1>", self.on_video_click)

        # Seek bar
        seek_frame = ttk.Frame(canvas_frame)
        seek_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        seek_frame.columnconfigure(0, weight=1)

        ttk.Label(seek_frame, text="Seek:").grid(row=0, column=0, padx=(0, 5))
        self.seek_var = tk.DoubleVar()
        self.seek_scale = ttk.Scale(seek_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                    variable=self.seek_var, command=self.on_seek)
        self.seek_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        # Bottom frame for obj and mask
        bottom_frame = ttk.Frame(canvas_frame)
        bottom_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)

        # obj canvas
        obj_label = ttk.Label(bottom_frame, text="Filtered Object")
        obj_label.grid(row=0, column=0, pady=(0, 5))
        self.obj_canvas = tk.Canvas(bottom_frame, bg="black", width=320, height=180)
        self.obj_canvas.grid(row=1, column=0, padx=(0, 5))

        # mask canvas
        mask_label = ttk.Label(bottom_frame, text="Binary Mask")
        mask_label.grid(row=0, column=1, pady=(0, 5))
        self.mask_canvas = tk.Canvas(bottom_frame, bg="black", width=320, height=180)
        self.mask_canvas.grid(row=1, column=1, padx=(5, 0))

        # Right panel for controls and output
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_frame.columnconfigure(0, weight=1)
        control_frame.rowconfigure(2, weight=1)

        # HSV Controls
        hsv_frame = ttk.LabelFrame(control_frame, text="HSV Color Range", padding="5")
        hsv_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        hsv_frame.columnconfigure(1, weight=1)

        # Create HSV sliders
        self.create_slider(hsv_frame, "Lhue", 0, 179, 0, self.T1)
        self.create_slider(hsv_frame, "Hhue", 1, 179, 0, self.T2)
        self.create_slider(hsv_frame, "Lsat", 2, 255, 0, self.T3)
        self.create_slider(hsv_frame, "Hsat", 3, 255, 0, self.T4)
        self.create_slider(hsv_frame, "Lval", 4, 255, 0, self.T5)
        self.create_slider(hsv_frame, "Hval", 5, 255, 0, self.T6)

        # Buttons for HSV controls
        button_frame = ttk.Frame(hsv_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=(10, 0))
        ttk.Button(button_frame, text="Reset", command=self.reset_hsv).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Blue", command=self.blue_preset).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Red", command=self.red_preset).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Green", command=self.green_preset).pack(side=tk.LEFT)

        # Detection Settings
        detection_frame = ttk.LabelFrame(control_frame, text="Detection Settings", padding="5")
        detection_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        detection_frame.columnconfigure(1, weight=1)

        ttk.Label(detection_frame, text="Min Area:").grid(row=0, column=0, sticky=tk.W)
        self.min_area_var = tk.IntVar(value=500)
        ttk.Scale(detection_frame, from_=100, to=2000, orient=tk.HORIZONTAL,
                  variable=self.min_area_var, command=self.update_min_area).grid(row=0, column=1, sticky=(tk.W, tk.E),
                                                                                 padx=(5, 5))
        self.min_area_label = ttk.Label(detection_frame, text="500")
        self.min_area_label.grid(row=0, column=2)

        ttk.Label(detection_frame, text="Smoothing:").grid(row=1, column=0, sticky=tk.W)
        self.smoothing_var = tk.DoubleVar(value=0.7)
        ttk.Scale(detection_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                  variable=self.smoothing_var, command=self.update_smoothing).grid(row=1, column=1, sticky=(tk.W, tk.E),
                                                                                   padx=(5, 5))
        self.smoothing_label = ttk.Label(detection_frame, text="0.7")
        self.smoothing_label.grid(row=1, column=2)

        # Output text area with enhanced info
        output_frame = ttk.LabelFrame(control_frame, text="Status & Output", padding="5")
        output_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(2, weight=1)

        # Frame and status info
        self.frame_info = ttk.Label(output_frame, text="Frame: 0/0 | FPS: 0")
        self.frame_info.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        self.status_info = ttk.Label(output_frame, text="Status: Ready", foreground="green")
        self.status_info.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))

        self.output_text = scrolledtext.ScrolledText(output_frame, height=12, state='disabled')
        self.output_text.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def create_slider(self, parent, label, row, max_val, initial_val, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 10))
        slider = ttk.Scale(parent, from_=0, to=max_val, orient=tk.HORIZONTAL, command=command)
        slider.set(initial_val)
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        value_label = ttk.Label(parent, text=str(initial_val), width=4)
        value_label.grid(row=row, column=2)

        setattr(self, f"{label.lower()}_slider", slider)
        setattr(self, f"{label.lower()}_label", value_label)

    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        )
        if filename:
            self.path_var.set(filename)

    def start_tracking(self):
        if not self.running:
            video_path = self.path_var.get()
            if not os.path.exists(video_path):
                messagebox.showerror("Error", f"Video file '{video_path}' not found!")
                return

            self.running = True
            self.paused = False
            self.tracking_history = []
            self.start_btn.config(state='disabled')
            self.pause_btn.config(state='normal')
            self.stop_btn.config(state='normal')
            self.status_info.config(text="Status: Processing...", foreground="blue")

            self.log_message("🎬 Starting color tracking analysis...")
            self.log_message(f"📁 Video: {os.path.basename(video_path)}")

            self.video_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
            self.video_thread.start()

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.config(text="Resume")
            self.status_info.config(text="Status: Paused", foreground="orange")
            self.log_message("⏸️ Tracking paused")
        else:
            self.pause_btn.config(text="Pause")
            self.status_info.config(text="Status: Processing...", foreground="blue")
            self.log_message("▶️ Tracking resumed")

    def stop_tracking(self):
        self.running = False
        self.paused = False
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled', text="Pause")
        self.stop_btn.config(state='disabled')
        self.status_info.config(text="Status: Stopped", foreground="red")

        if self.cap:
            self.cap.release()
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            self.record_btn.config(text="Start Recording")

        self.log_message("⏹️ Tracking stopped")

    def toggle_recording(self):
        if not self.recording:
            self.output_path = filedialog.asksaveasfilename(
                title="Save Recorded Video",
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")]
            )
            if self.output_path:
                self.recording = True
                self.record_btn.config(text="Stop Recording")
                self.log_message(f"🔴 Recording started: {os.path.basename(self.output_path)}")
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.record_btn.config(text="Start Recording")
            self.log_message("⏹️ Recording stopped")

    def on_seek(self, value):
        if self.cap and not self.running:
            frame_pos = int(float(value) * self.total_frames / 100)
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_pos)

    def update_min_area(self, value):
        self.confidence_threshold = int(float(value))
        self.min_area_label.config(text=str(self.confidence_threshold))

    def update_smoothing(self, value):
        self.smoothing_factor = float(value)
        self.smoothing_label.config(text=f"{self.smoothing_factor:.1f}")

    def save_settings(self):
        settings = {
            'hsv_values': {
                'Lhue': self.Lhue, 'Hhue': self.Hhue,
                'Lsat': self.Lsat, 'Hsat': self.Hsat,
                'Lval': self.Lval, 'Hval': self.Hval
            },
            'detection_settings': {
                'min_area': self.confidence_threshold,
                'smoothing': self.smoothing_factor
            }
        }

        filename = filedialog.asksaveasfilename(
            title="Save Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=2)
            self.log_message(f"💾 Settings saved: {os.path.basename(filename)}")

    def load_settings(self):
        filename = filedialog.askopenfilename(
            title="Load Settings",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    settings = json.load(f)

                hsv = settings['hsv_values']
                self.Lhue, self.Hhue = hsv['Lhue'], hsv['Hhue']
                self.Lsat, self.Hsat = hsv['Lsat'], hsv['Hsat']
                self.Lval, self.Hval = hsv['Lval'], hsv['Hval']

                detect = settings['detection_settings']
                self.confidence_threshold = detect['min_area']
                self.smoothing_factor = detect['smoothing']

                self.root.after(0, self.update_all_sliders)
                self.log_message(f"📂 Settings loaded: {os.path.basename(filename)}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load settings: {str(e)}")

    def export_tracking_data(self):
        if not self.tracking_history:
            messagebox.showwarning("Warning", "No tracking data to export!")
            return

        filename = filedialog.asksaveasfilename(
            title="Export Tracking Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if filename:
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Frame', 'X', 'Y', 'Area', 'Confidence', 'Section'])
                writer.writerows(self.tracking_history)
            self.log_message(f"📊 Tracking data exported: {os.path.basename(filename)}")

    def clear_output(self):
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')

    def reset_hsv(self):
        for slider_name in ["lhue", "hhue", "lsat", "hsat", "lval", "hval"]:
            slider = getattr(self, f"{slider_name}_slider")
            slider.set(0)

    def blue_preset(self):
        self.lhue_slider.set(100)
        self.hhue_slider.set(130)
        self.lsat_slider.set(100)
        self.hsat_slider.set(255)
        self.lval_slider.set(50)
        self.hval_slider.set(255)
        self.log_message("🔵 Blue preset applied")

    def red_preset(self):
        self.lhue_slider.set(160)
        self.hhue_slider.set(179)
        self.lsat_slider.set(100)
        self.hsat_slider.set(255)
        self.lval_slider.set(50)
        self.hval_slider.set(255)
        self.log_message("🔴 Red preset applied")

    def green_preset(self):
        self.lhue_slider.set(50)
        self.hhue_slider.set(80)
        self.lsat_slider.set(100)
        self.hsat_slider.set(255)
        self.lval_slider.set(50)
        self.hval_slider.set(255)
        self.log_message("🟢 Green preset applied")

    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.message_queue.put(f"[{timestamp}] {message}")

    def process_messages(self):
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.output_text.config(state='normal')
                self.output_text.insert(tk.END, message + "\n")
                self.output_text.see(tk.END)
                self.output_text.config(state='disabled')
        except queue.Empty:
            pass

        self.root.after(100, self.process_messages)

    def on_video_click(self, event):
        self.X = event.x
        self.Y = event.y
        self.log_message(f"🎯 Click coordinates: ({self.X}, {self.Y})")
        self.click = True

    # HSV slider callbacks
    def T1(self, x):
        self.Lhue = int(float(x))
        self.lhue_label.config(text=str(self.Lhue))

    def T2(self, x):
        self.Hhue = int(float(x))
        self.hhue_label.config(text=str(self.Hhue))

    def T3(self, x):
        self.Lsat = int(float(x))
        self.lsat_label.config(text=str(self.Lsat))

    def T4(self, x):
        self.Hsat = int(float(x))
        self.hsat_label.config(text=str(self.Hsat))

    def T5(self, x):
        self.Lval = int(float(x))
        self.lval_label.config(text=str(self.Lval))

    def T6(self, x):
        self.Hval = int(float(x))
        self.hval_label.config(text=str(self.Hval))

    def cv_to_tkinter(self, cv_image):
        if len(cv_image.shape) == 3:
            cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        elif len(cv_image.shape) == 2:
            cv_image = cv.cvtColor(cv_image, cv.COLOR_GRAY2RGB)

        pil_image = Image.fromarray(cv_image)
        return ImageTk.PhotoImage(pil_image)

    def update_canvas(self, canvas, image, width, height):
        if image is not None:
            resized_image = cv.resize(image, (width, height))
            tk_image = self.cv_to_tkinter(resized_image)
            canvas.delete("all")
            canvas.create_image(width // 2, height // 2, image=tk_image)
            canvas.image = tk_image

    def update_all_sliders(self):
        slider_mappings = [
            ("Lhue", "lhue_slider"), ("Hhue", "hhue_slider"), ("Lsat", "lsat_slider"),
            ("Hsat", "hsat_slider"), ("Lval", "lval_slider"), ("Hval", "hval_slider")
        ]

        for attr, slider_name in slider_mappings:
            value = getattr(self, attr)
            slider = getattr(self, slider_name)
            slider.set(value)

        # Update detection settings
        self.min_area_var.set(self.confidence_threshold)
        self.smoothing_var.set(self.smoothing_factor)
        self.min_area_label.config(text=str(self.confidence_threshold))
        self.smoothing_label.config(text=f"{self.smoothing_factor:.1f}")

    def video_processing_loop(self):
        self.cap = cv.VideoCapture(self.path_var.get())
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv.CAP_PROP_FPS)

        # Initialize video writer if recording
        if self.recording and self.output_path:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv.VideoWriter(self.output_path, fourcc, fps, (self.width, self.height))

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue

            # Update frame info
            self.frame_count = int(self.cap.get(cv.CAP_PROP_POS_FRAMES))
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps_counter = int(1.0 / (current_time - self.last_time + 0.001))
                self.last_time = current_time
                self.root.after(0, self.update_frame_info)

                # Update seek bar
                progress = (self.frame_count / self.total_frames) * 100
                self.seek_var.set(progress)

            framee = cv.resize(frame, (self.width, self.height))
            framehsv = cv.cvtColor(framee, cv.COLOR_BGR2HSV)
            lowerb = np.array([self.Lhue, self.Lsat, self.Lval])
            highrb = np.array([self.Hhue, self.Hsat, self.Hval])
            mask = cv.inRange(framehsv, lowerb, highrb)

            # Morphological operations for noise reduction
            kernel = np.ones((3, 3), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

            smallmask = cv.resize(mask, (self.width // 2, self.height // 2))
            obj = cv.bitwise_and(framehsv, framehsv, mask=mask)
            smlallobj = cv.resize(obj, (self.width // 2, self.height // 2))

            # Handle click for color selection
            if self.click:
                if 0 <= self.X <= self.width and 0 <= self.Y <= self.height:
                    hsvclicked = framehsv[self.Y, self.X]
                    self.log_message(f"🎨 HSV at click: {hsvclicked}")
                    h, s, v = int(hsvclicked[0]), int(hsvclicked[1]), int(hsvclicked[2])

                    # Auto-adjust with better ranges
                    self.Lhue = max(h - 15, 0)
                    self.Hhue = min(h + 15, 179)
                    self.Lsat = max(s - 60, 0)
                    self.Hsat = min(s + 60, 255)
                    self.Lval = max(v - 60, 0)
                    self.Hval = min(v + 60, 255)

                    self.root.after(0, self.update_all_sliders)
                    self.log_message(f"🔧 Auto-adjusted HSV ranges")
                    self.click = False

            # Find and process contours
            cnt, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if cnt:
                # Get largest contour
                largest_cnt = max(cnt, key=cv.contourArea)
                area = cv.contourArea(largest_cnt)

                # Only process if area meets threshold
                if area >= self.confidence_threshold:
                    x, y, w, h = cv.boundingRect(largest_cnt)
                    cx = x + w // 2
                    cy = y + h // 2

                    # Apply position smoothing
                    if self.last_position is not None:
                        smooth_cx = int(
                            self.smoothing_factor * self.last_position[0] + (1 - self.smoothing_factor) * cx)
                        smooth_cy = int(
                            self.smoothing_factor * self.last_position[1] + (1 - self.smoothing_factor) * cy)
                        cx, cy = smooth_cx, smooth_cy

                    self.last_position = (cx, cy)

                    # Determine section
                    section = "LEFT" if cx <= self.width // 3 else "MIDDLE" if cx <= 2 * self.width // 3 else "RIGHT"

                    # Calculate confidence based on area
                    max_area = self.width * self.height * 0.1  # Assume max 10% of frame
                    confidence = min(100, (area / max_area) * 100)

                    # Store tracking data
                    self.tracking_history.append([
                        self.frame_count, cx, cy, int(area), f"{confidence:.1f}%", section
                    ])

                    # Draw tracking visuals
                    cv.rectangle(framee, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.circle(framee, (cx, cy), 8, (255, 0, 0), -1)
                    cv.circle(framee, (cx, cy), 15, (255, 255, 255), 2)

                    # Draw info text
                    info_text = f"Area: {int(area)} | Conf: {confidence:.1f}%"
                    cv.putText(framee, info_text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv.putText(framee, section, (cx - 30, cy + 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # Update smaller displays
                    cv.rectangle(smallmask, (x // 2, y // 2), ((x + w) // 2, (y + h) // 2), (255), 2)
                    cv.rectangle(smlallobj, (x // 2, y // 2), ((x + w) // 2, (y + h) // 2), (255, 255, 255), 2)

                    # Log periodic updates
                    if self.frame_count % 60 == 0:  # Every 60 frames (roughly 2 seconds at 30fps)
                        self.log_message(
                            f"📍 Frame {self.frame_count}: {section} section, Area: {int(area)}, Confidence: {confidence:.1f}%")

                    # Final result detection
                    if self.frame_count == self.total_frames - 1:
                        self.log_message("=" * 50)
                        self.log_message(f"🏆 FINAL RESULT: Object is in the {section} section")
                        self.log_message(f"📊 Final position: ({cx}, {cy}) with {confidence:.1f}% confidence")
                        self.log_message("=" * 50)

                        # Show result dialog
                        self.root.after(0, lambda: messagebox.showinfo(
                            "Tracking Complete",
                            f"🎯 Final Result: The tracked object is in the {section} section!\n\n"
                            f"Position: ({cx}, {cy})\n"
                            f"Confidence: {confidence:.1f}%\n"
                            f"Total frames processed: {self.frame_count}"
                        ))
                else:
                    # Object too small, show low confidence
                    if self.frame_count % 120 == 0:  # Less frequent logging for low confidence
                        self.log_message(
                            f"⚠️ Frame {self.frame_count}: Low confidence detection (area: {int(area)} < {self.confidence_threshold})")
            else:
                # No object detected
                if self.frame_count % 180 == 0:  # Even less frequent for no detection
                    self.log_message(f"❌ Frame {self.frame_count}: No object detected")

            # Draw section dividers and labels
            cv.line(framee, (self.width // 3, 0), (self.width // 3, self.height), (255, 255, 255), 2)
            cv.line(framee, (2 * self.width // 3, 0), (2 * self.width // 3, self.height), (255, 255, 255), 2)

            # Enhanced section labels with backgrounds
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # LEFT label
            text = "LEFT"
            text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
            cv.rectangle(framee, (self.width // 6 - text_size[0] // 2 - 5, 15),
                         (self.width // 6 + text_size[0] // 2 + 5, 45), (0, 0, 0), -1)
            cv.putText(framee, text, (self.width // 6 - text_size[0] // 2, 35), font, font_scale, (255, 255, 255),
                       thickness)

            # MIDDLE label
            text = "MIDDLE"
            text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
            cv.rectangle(framee, (self.width // 2 - text_size[0] // 2 - 5, 15),
                         (self.width // 2 + text_size[0] // 2 + 5, 45), (0, 0, 0), -1)
            cv.putText(framee, text, (self.width // 2 - text_size[0] // 2, 35), font, font_scale, (255, 255, 255),
                       thickness)

            # RIGHT label
            text = "RIGHT"
            text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
            cv.rectangle(framee, (5 * self.width // 6 - text_size[0] // 2 - 5, 15),
                         (5 * self.width // 6 + text_size[0] // 2 + 5, 45), (0, 0, 0), -1)
            cv.putText(framee, text, (5 * self.width // 6 - text_size[0] // 2, 35), font, font_scale, (255, 255, 255),
                       thickness)

            # Record frame if recording
            if self.recording and self.video_writer:
                self.video_writer.write(cv.resize(framee, (self.width, self.height)))

            # Update UI displays
            self.root.after(0, lambda: self.update_canvas(self.video_canvas, framee, 640, 360))
            self.root.after(0, lambda: self.update_canvas(self.obj_canvas, smlallobj, 320, 180))
            self.root.after(0, lambda: self.update_canvas(self.mask_canvas, smallmask, 320, 180))

            # Control frame rate (25 FPS = 40ms delay)
            cv.waitKey(40)

        # Cleanup
        if self.cap:
            self.cap.release()
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.recording = False
            self.record_btn.config(text="Start Recording")

        self.root.after(0, lambda: self.status_info.config(text="Status: Complete", foreground="green"))
        self.log_message("✅ Video processing completed!")

    def update_frame_info(self):
        progress = (self.frame_count / self.total_frames) * 100 if self.total_frames > 0 else 0
        self.frame_info.config(
            text=f"Frame: {self.frame_count}/{self.total_frames} ({progress:.1f}%) | FPS: {self.fps_counter}")


if __name__ == "__main__":
    root = tk.Tk()

    # Set application icon and styling
    try:
        root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(data='''
            iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==
        '''))
    except:
        pass  # Icon loading failed, continue without it

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_reqwidth()
    height = root.winfo_reqheight()
    pos_x = (root.winfo_screenwidth() // 2) - (width // 2)
    pos_y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")

    app = ColorTrackerUI(root)


    # Handle window closing
    def on_closing():
        if app.running:
            app.stop_tracking()
        root.destroy()


    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()