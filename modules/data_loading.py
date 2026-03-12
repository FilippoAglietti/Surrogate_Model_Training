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
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        self.header = ctk.CTkLabel(self, text="DATA LOADING 📂", font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(30, 20), sticky="w", padx=40)
        
        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        
        # Upload Button
        self.upload_btn = ctk.CTkButton(
            self.content_frame, text="Upload Dataset (.xlsx, .csv)", 
            font=FONTS["header"], height=50,
            command=self.browse_file
        )
        self.upload_btn.grid(row=0, column=0, pady=10, padx=40, sticky="ew")

        # Status Label
        self.status_lbl = ctk.CTkLabel(self.content_frame, text="No file loaded.", text_color=COLORS["text_dim"])
        self.status_lbl.grid(row=1, column=0, padx=40, sticky="w")
        
        # Data Info Frame
        self.info_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        self.info_frame.grid(row=2, column=0, pady=20, padx=40, sticky="ew")
        self.info_frame.grid_columnconfigure(0, weight=1)
        
        self.shape_lbl = ctk.CTkLabel(self.info_frame, text="Rows: 0 | Columns: 0", font=FONTS["body"])
        self.shape_lbl.grid(row=0, column=0, pady=10, padx=20, sticky="w")
        
        self.stats_box = ctk.CTkTextbox(self.info_frame, height=180, font=FONTS["code"], fg_color="#0D1117", text_color=COLORS["text"])
        self.stats_box.grid(row=1, column=0, pady=(0, 20), padx=20, sticky="ew")
        self.stats_box.configure(state="disabled")

        # Column Selection Frame
        self.col_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        self.col_frame.grid(row=3, column=0, pady=10, padx=40, sticky="ew")
        self.col_frame.grid_columnconfigure(0, weight=1)
        self.col_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.col_frame, text="Target (Output) Columns", font=FONTS["header"], text_color=COLORS["magenta"]).grid(row=0, column=0, pady=10, padx=20, sticky="w")
        self.targets_scroll = ctk.CTkScrollableFrame(self.col_frame, height=180, fg_color=COLORS["bg"])
        self.targets_scroll.grid(row=1, column=0, pady=(0, 20), padx=20, sticky="ew")
        self.target_vars = {}
        
        ctk.CTkLabel(self.col_frame, text="Features (Input) Columns", font=FONTS["header"], text_color=COLORS["cyan"]).grid(row=0, column=1, pady=10, padx=20, sticky="w")
        self.features_scroll = ctk.CTkScrollableFrame(self.col_frame, height=180, fg_color=COLORS["bg"])
        self.features_scroll.grid(row=1, column=1, pady=(0, 20), padx=20, sticky="ew")
        self.feature_vars = {}
        
        # Proceed Button
        self.proceed_btn = ctk.CTkButton(
            self.content_frame, text="Proceed to Preprocessing →",
            font=FONTS["header"], height=50,
            fg_color=COLORS["primary_dark"],
            hover_color=COLORS["primary"],
            text_color="#000",
            state="disabled",
            command=self.go_to_preprocessing
        )
        self.proceed_btn.grid(row=4, column=0, pady=30, padx=40, sticky="ew")

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
            
            # Print stats
            desc = df.describe().T
            desc_str = desc[['min', 'max', 'mean', 'std']].round(3).to_string()
            self.stats_box.configure(state="normal")
            self.stats_box.delete("0.0", "end")
            self.stats_box.insert("end", "--- DATASET STATISTICS ---\n" + desc_str)
            self.stats_box.configure(state="disabled")
            
            columns = df.select_dtypes(include=['number']).columns.tolist()
            if not columns:
                messagebox.showerror("Error", "No numeric columns found in dataset.")
                return
                
            # Setup features and targets checkboxes
            for widget in self.features_scroll.winfo_children():
                widget.destroy()
            for widget in self.targets_scroll.winfo_children():
                widget.destroy()
                
            self.feature_vars.clear()
            self.target_vars.clear()
            
            for i, col in enumerate(columns):
                # Target
                t_var = ctk.StringVar(value="off")
                if i == len(columns) - 1: # Default last column as target
                    t_var.set("on")
                t_chk = ctk.CTkCheckBox(self.targets_scroll, text=col, variable=t_var, onvalue="on", offvalue="off", command=self.check_ready)
                t_chk.grid(row=i, column=0, sticky="w", pady=2, padx=5)
                self.target_vars[col] = t_var
                
                # Feature
                f_var = ctk.StringVar(value="on")
                if i == len(columns) - 1: # Default last column off for features
                    f_var.set("off")
                f_chk = ctk.CTkCheckBox(self.features_scroll, text=col, variable=f_var, onvalue="on", offvalue="off", command=self.check_ready)
                f_chk.grid(row=i, column=0, sticky="w", pady=2, padx=5)
                self.feature_vars[col] = f_var
                
            self.check_ready()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{str(e)}")

    def check_ready(self):
        # We allow multiple targets and multiple features, but they must be disjoint sets!
        # Let's enforce disjoint sets: if chosen as target, disable as feature and vice-versa.
        
        targets = [c for c, v in self.target_vars.items() if v.get() == "on"]
        features = [c for c, v in self.feature_vars.items() if v.get() == "on"]
        
        overlap = set(targets).intersection(set(features))
        
        if len(targets) > 0 and len(features) > 0 and len(overlap) == 0:
            self.proceed_btn.configure(state="normal")
            # Store list of columns instead of a single string
            set_state("output_column", targets) 
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
