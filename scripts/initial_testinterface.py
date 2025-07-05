import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ultralytics import YOLO

class YOLODetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("YOLO Object Detection")
        master.geometry("800x600")

        # Find the latest model
        try:
            self.model_path = self.find_latest_model()
            self.model = YOLO(self.model_path)
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load model: {e}")
            self.model = None

        # Create UI elements
        self.create_widgets()

    def find_latest_model(self):
        """
        Find the most recent trained model (copied from model_verification.py)
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        runs_dir = os.path.join(project_root, 'runs', 'detect')
        
        training_dirs = [
            d for d in os.listdir(runs_dir) 
            if d.startswith('train_')
        ]
        
        if not training_dirs:
            raise FileNotFoundError("No training runs found!")
        
        latest_run = "train_20241214_143213"
        model_path = os.path.join(
            runs_dir, 
            latest_run, 
            'yolov8_training',
            'weights', 
            'best.pt'
        )
        
        return model_path

    def create_widgets(self):
        # Select Image Button
        self.select_button = tk.Button(
            self.master, 
            text="Select Image", 
            command=self.select_image
        )
        self.select_button.pack(pady=10)

        # Frame for displaying results
        self.result_frame = tk.Frame(self.master)
        self.result_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def select_image(self):
        # Check if model is loaded
        if not self.model:
            messagebox.showerror("Model Error", "No model loaded!")
            return

        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        # Clear previous results
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        # Predict and visualize
        self.predict_and_show(file_path)

    def predict_and_show(self, image_path):
        # Run prediction
        results = self.model.predict(
            source=image_path, 
            conf=0.25,
            max_det=300
        )

        # Create a figure for matplotlib
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot results
        for r in results:
            im_array = r.plot()
            ax.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
            ax.set_title("Object Detection Results")
            ax.axis('off')

        # Embed matplotlib figure in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill=tk.BOTH)

        # Add class information
        class_info = "\n".join([
            f"Class {idx}: {name}" 
            for idx, name in enumerate(self.model.names)
        ])
        
        info_label = tk.Label(
            self.result_frame, 
            text=f"Model: {os.path.basename(self.model_path)}\n\nClasses:\n{class_info}",
            justify=tk.LEFT,
            font=("Courier", 10)
        )
        info_label.pack(pady=10)

def main():
    root = tk.Tk()
    app = YOLODetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
