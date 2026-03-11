import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import os

from utils.theme import COLORS, FONTS
from utils.state import get_state, set_state


class DataLoadingFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        
        # Header
        self.header = ctk.CTkLabel(self, text="DATA LOADING 📂", font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(30, 20), sticky="w", padx=40)
        
        # Upload Button
        self.upload_btn = ctk.CTkButton(
            self, text="Upload Dataset (.xlsx, .csv)", 
            font=FONTS["header"], height=50,
            command=self.browse_file
        )
        self.upload_btn.grid(row=1, column=0, pady=10, padx=40, sticky="ew")

        # Status Label
        self.status_lbl = ctk.CTkLabel(self, text="No file loaded.", text_color=COLORS["text_dim"])
        self.status_lbl.grid(row=2, column=0, padx=40, sticky="w")
        
        # Data Info Frame
        self.info_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_card"])
        self.info_frame.grid(row=3, column=0, pady=20, padx=40, sticky="ew")
        self.info_frame.grid_columnconfigure(1, weight=1)
        
        self.shape_lbl = ctk.CTkLabel(self.info_frame, text="Rows: 0 | Columns: 0", font=FONTS["body"])
        self.shape_lbl.grid(row=0, column=0, pady=10, padx=20, sticky="w")
        
        # Column Selection Frame
        self.col_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_card"])
        self.col_frame.grid(row=4, column=0, pady=10, padx=40, sticky="ew")
        self.col_frame.grid_columnconfigure(0, weight=1)
        self.col_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.col_frame, text="Target (Output) Column", font=FONTS["header"], text_color=COLORS["magenta"]).grid(row=0, column=0, pady=10, padx=20, sticky="w")
        self.target_combo = ctk.CTkComboBox(self.col_frame, values=[], command=self.update_inputs)
        self.target_combo.grid(row=1, column=0, pady=(0, 20), padx=20, sticky="ew")
        
        ctk.CTkLabel(self.col_frame, text="Features (Input) Columns", font=FONTS["header"], text_color=COLORS["cyan"]).grid(row=0, column=1, pady=10, padx=20, sticky="w")
        
        # Scrollable frame for checkboxes
        self.features_scroll = ctk.CTkScrollableFrame(self.col_frame, height=150, fg_color=COLORS["bg"])
        self.features_scroll.grid(row=1, column=1, pady=(0, 20), padx=20, sticky="ew")
        self.feature_vars = {}
        
        # Proceed Button
        self.proceed_btn = ctk.CTkButton(
            self, text="Proceed to Preprocessing →",
            font=FONTS["header"], height=50,
            fg_color=COLORS["primary_dark"],
            hover_color=COLORS["primary"],
            text_color="#000",
            state="disabled",
            command=self.go_to_preprocessing
        )
        self.proceed_btn.grid(row=5, column=0, pady=30, padx=40, sticky="ew")

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=(("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not filename:
            return
            
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filename)
            else:
                df = pd.read_excel(filename)
                
            set_state("df", df)
            set_state("data_loaded", True)
            set_state("preprocessed", False) # Reset
            
            self.status_lbl.configure(text=f"Loaded: {os.path.basename(filename)}", text_color=COLORS["green"])
            self.shape_lbl.configure(text=f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
            
            columns = df.select_dtypes(include=['number']).columns.tolist()
            if not columns:
                messagebox.showerror("Error", "No numeric columns found in dataset.")
                return
                
            # Setup target combo
            self.target_combo.configure(values=columns)
            self.target_combo.set(columns[-1])  # Default to last col
            
            # Setup feature checkboxes
            for widget in self.features_scroll.winfo_children():
                widget.destroy()
                
            self.feature_vars.clear()
            for i, col in enumerate(columns):
                var = ctk.StringVar(value="on")
                chk = ctk.CTkCheckBox(
                    self.features_scroll, text=col, variable=var, 
                    onvalue="on", offvalue="off",
                    command=self.check_ready
                )
                chk.grid(row=i, column=0, sticky="w", pady=2, padx=5)
                self.feature_vars[col] = var
                
            self.update_inputs(self.target_combo.get())
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{str(e)}")

    def update_inputs(self, selected_target):
        # Deselect target from features
        for col, var in self.feature_vars.items():
            if col == selected_target:
                var.set("off")
                # Optionally disable the checkbox entirely:
                # self.features_scroll.winfo_children()[list(self.feature_vars.keys()).index(col)].configure(state="disabled")
            else:
                var.set("on")
                
        self.check_ready()
        
    def check_ready(self):
        target = self.target_combo.get()
        features = [col for col, var in self.feature_vars.items() if var.get() == "on" and col != target]
        
        if target and features:
            self.proceed_btn.configure(state="normal")
            set_state("output_column", target)
            set_state("input_columns", features)
        else:
            self.proceed_btn.configure(state="disabled")

    def go_to_preprocessing(self):
        # Trigger navigation through parent app
        self.master.navigate_to("🔧  Preprocessing")
        
    def on_show(self):
        # Refresh if we already have data
        if get_state("data_loaded"):
            pass
