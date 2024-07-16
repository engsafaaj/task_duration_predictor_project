import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from joblib import load

# Load the trained model
model_path = 'task_duration_predictor_new.joblib'
model = load(model_path)

# Function to validate input
def validate_input():
    try:
        float(project_size_var.get())
        int(project_duration_var.get())
        if not project_type_var.get() or not task_name_var.get() or not task_description_var.get("1.0", tk.END).strip():
            raise ValueError
        return True
    except ValueError:
        messagebox.showerror("Invalid input", "Please ensure all fields are filled correctly.")
        return False

# Function to predict task duration
def predict_duration():
    if validate_input():
        project_type = project_type_var.get()
        project_size = float(project_size_var.get())
        project_duration = int(project_duration_var.get())
        task_name = task_name_var.get()
        task_description = task_description_var.get("1.0", tk.END).strip()

        # Create a DataFrame for the model input to match the training features
        input_data = pd.DataFrame([[project_type, project_size, project_duration, task_name, task_description]],
                                  columns=['project_type', 'project_size', 'project_duration', 'task_name', 'task_description'])
        
        # Predict task duration
        predicted_duration = model.predict(input_data)[0]
        result_var.set(f"Predicted Task Duration: {predicted_duration:.2f} days")

# Create the main window
root = tk.Tk()
root.title("Task Duration Predictor")
root.geometry("600x450")  # Set the window size

# Set a larger font
font = ("Times New Roman (Headings CS)", 12)

# Define options for comboboxes
project_types = ['Infrastructure', 'Residential', 'Commercial', 'Industrial']
task_names = ['Roofing', 'Framing', 'Plumbing', 'Electrical', 'Painting']

# Create input fields
tk.Label(root, text="Project Type:", font=font).grid(row=0, column=0, padx=10, pady=10, sticky='w')
project_type_var = tk.StringVar()
ttk.Combobox(root, textvariable=project_type_var, values=project_types, font=font).grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Project Size (m^2):", font=font).grid(row=1, column=0, padx=10, pady=10, sticky='w')
project_size_var = tk.StringVar()
ttk.Entry(root, textvariable=project_size_var, font=font).grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="Project Duration (days):", font=font).grid(row=2, column=0, padx=10, pady=10, sticky='w')
project_duration_var = tk.StringVar()
ttk.Entry(root, textvariable=project_duration_var, font=font).grid(row=2, column=1, padx=10, pady=10)

tk.Label(root, text="Task Name:", font=font).grid(row=3, column=0, padx=10, pady=10, sticky='w')
task_name_var = tk.StringVar()
ttk.Combobox(root, textvariable=task_name_var, values=task_names, font=font).grid(row=3, column=1, padx=10, pady=10)

tk.Label(root, text="Task Description:", font=font).grid(row=4, column=0, padx=10, pady=10, sticky='w')
task_description_var = tk.Text(root, height=5, width=40, font=font)
task_description_var.grid(row=4, column=1, padx=10, pady=10)

# Create a button to predict task duration
ttk.Button(root, text="Predict Duration", command=predict_duration).grid(row=5, column=0, columnspan=2, pady=20)
# Label to display the result
result_var = tk.StringVar()
tk.Label(root, textvariable=result_var, font=('Times New Roman (Headings CS)', 14, 'bold')).grid(row=6, column=0, columnspan=2, pady=10)

# Run the application
root.mainloop()
