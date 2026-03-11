import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.theme import COLORS, FONTS
from utils.state import get_state, set_state


class PreprocessingFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        self.header = ctk.CTkLabel(self, text="PREPROCESSING 🔧", font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(30, 20), sticky="w", padx=30)
        
        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(1, weight=1)

        # Plot frame
        self.plot_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"], height=400)
        self.plot_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=20, padx=20)
        
        self.built_ui = False

    def on_show(self):
        if not get_state("data_loaded"):
            self._show_blocked("Load data and select columns first.\n← Go to 'Data Loading'")
            return
            
        if not self.built_ui:
            self._build_ui()
            self.built_ui = True
            
        # If preprocessed already, ensure plot is visible
        if get_state("preprocessed"):
            self.proceed_btn.configure(state="normal")
            self.status_lbl.configure(text="✓ Data preprocessed and split.", text_color=COLORS["green"])

    def _show_blocked(self, message):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.built_ui = False
        lbl = ctk.CTkLabel(self.content_frame, text=f"[ BLOCKED ]\n{message}", font=FONTS["header"], text_color=COLORS["red"])
        lbl.grid(row=0, column=0, pady=50, padx=20)

    def _build_ui(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
        # Plot frame needs to be recreated if we destroyed everything
        self.plot_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"], height=500)
        self.plot_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=20, padx=20)
        self.plot_frame.pack_propagate(False)
            
        # -- Settings Controls --
        settings_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        settings_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
        settings_frame.grid_columnconfigure(0, weight=1)
        settings_frame.grid_columnconfigure(1, weight=1)
        
        # Scaling
        ctk.CTkLabel(settings_frame, text="Normalization", font=FONTS["header"]).grid(row=0, column=0, pady=10, sticky="w", padx=20)
        self.scaling_var = ctk.StringVar(value="Min-Max (0, 1)")
        scalers = ["Min-Max (0, 1)", "Standard (Z-score)", "None"]
        self.scale_combo = ctk.CTkComboBox(settings_frame, values=scalers, variable=self.scaling_var)
        self.scale_combo.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        self.scale_tgt_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(settings_frame, text="Scale Target Variable", variable=self.scale_tgt_var).grid(row=2, column=0, padx=20, pady=(0, 20), sticky="w")
        
        # Split
        ctk.CTkLabel(settings_frame, text="Data Split", font=FONTS["header"]).grid(row=0, column=1, pady=10, sticky="w", padx=20)
        
        self.train_pct = ctk.DoubleVar(value=0.7)
        self.val_pct = ctk.DoubleVar(value=0.15)
        
        split_controls = ctk.CTkFrame(settings_frame, fg_color="transparent")
        split_controls.grid(row=1, column=1, rowspan=2, padx=20, sticky="nsew")
        
        ctk.CTkLabel(split_controls, text="Train %").grid(row=0, column=0, sticky="w")
        ctk.CTkSlider(split_controls, variable=self.train_pct, from_=0.1, to=0.9, number_of_steps=80, command=self._update_splits).grid(row=0, column=1, padx=10, sticky="ew")
        self.train_lbl = ctk.CTkLabel(split_controls, text="70%")
        self.train_lbl.grid(row=0, column=2, sticky="e")
        
        ctk.CTkLabel(split_controls, text="Val %").grid(row=1, column=0, sticky="w", pady=10)
        ctk.CTkSlider(split_controls, variable=self.val_pct, from_=0.05, to=0.5, number_of_steps=45, command=self._update_splits).grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        self.val_lbl = ctk.CTkLabel(split_controls, text="15%")
        self.val_lbl.grid(row=1, column=2, sticky="e", pady=10)
        
        self.test_lbl = ctk.CTkLabel(split_controls, text="Test % : 15%", text_color=COLORS["magenta"])
        self.test_lbl.grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, 10))
        
        # Actions
        actions_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        actions_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
        
        self.run_btn = ctk.CTkButton(actions_frame, text="⚡ RUN PREPROCESSING", height=40, command=self.run_preprocessing)
        self.run_btn.pack(side="left", padx=10)
        
        self.status_lbl = ctk.CTkLabel(actions_frame, text="Ready to run.", text_color=COLORS["text_dim"])
        self.status_lbl.pack(side="left", padx=20)
        
        self.proceed_btn = ctk.CTkButton(
            self.content_frame, text="Proceed to Model Builder →",
            font=FONTS["header"], height=50,
            fg_color=COLORS["primary_dark"],
            hover_color=COLORS["primary"],
            text_color="#000",
            state="disabled",
            command=self.go_to_model_builder
        )
        self.proceed_btn.grid(row=4, column=0, columnspan=2, pady=30, padx=20, sticky="ew")

    def _update_splits(self, _=None):
        t = self.train_pct.get()
        v = self.val_pct.get()
        if t + v >= 0.95:
            v = 0.95 - t
            self.val_pct.set(v)
            
        test = 1.0 - t - v
        self.train_lbl.configure(text=f"{t*100:.0f}%")
        self.val_lbl.configure(text=f"{v*100:.0f}%")
        self.test_lbl.configure(text=f"Test % : {test*100:.0f}%")

    def run_preprocessing(self):
        df = get_state("df")
        input_cols = get_state("input_columns")
        output_col = get_state("output_column")
        
        # Drop NaNs
        df = df.dropna(subset=input_cols + [output_col]).copy()
        
        X = df[input_cols].values
        y = df[[output_col]].values
        
        # Scaling
        scaler_X = None
        scaler_y = None
        scale_type = self.scaling_var.get()
        
        if scale_type == "Min-Max (0, 1)":
            scaler_X = MinMaxScaler()
            X = scaler_X.fit_transform(X)
            if self.scale_tgt_var.get():
                scaler_y = MinMaxScaler()
                y = scaler_y.fit_transform(y)
        elif scale_type == "Standard (Z-score)":
            scaler_X = StandardScaler()
            X = scaler_X.fit_transform(X)
            if self.scale_tgt_var.get():
                scaler_y = StandardScaler()
                y = scaler_y.fit_transform(y)
                
        # Split (Math)
        train_p = self.train_pct.get()
        val_p = self.val_pct.get()
        test_p = max(0.01, 1.0 - train_p - val_p)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1.0 - train_p), random_state=42
        )
        relative_val = val_p / (val_p + test_p)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1.0 - relative_val), random_state=42
        )
        
        # Save state for TF
        set_state("X_train", X_train)
        set_state("X_val", X_val)
        set_state("X_test", X_test)
        set_state("y_train", y_train)
        set_state("y_val", y_val)
        set_state("y_test", y_test)
        set_state("scaler_X", scaler_X)
        set_state("scaler_y", scaler_y)
        set_state("preprocessed", True)
        
        self.status_lbl.configure(text=f"✓ Split: {len(X_train)} Train | {len(X_val)} Val | {len(X_test)} Test", text_color=COLORS["green"])
        self.proceed_btn.configure(state="normal")
        
        # Draw plot
        self._draw_pair_plot(X_train, y_train.flatten(), input_cols, output_col)

    def _draw_pair_plot(self, X_tr, y_tr, input_cols, output_col):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        plot_df = pd.DataFrame(X_tr, columns=input_cols)
        plot_df[output_col] = y_tr
        
        # Use a max of 4 features + target so it doesn't hang tk/matplotlib
        cols_to_plot = input_cols[:4] + [output_col]
        plot_df = plot_df[cols_to_plot]
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(8, 6), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        
        # Seaborn pairplot
        g = sns.pairplot(
            plot_df, 
            diag_kind="hist",
            plot_kws={'alpha':0.6, 'color': COLORS['cyan'], 's':15},
            diag_kws={'color': COLORS['magenta']},
            corner=True
        )
        g.fig.set_facecolor(COLORS["bg_card"])
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(g.fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        # plt.close(g.fig)

    def go_to_model_builder(self):
        self.master.navigate_to("🏗  Model Builder")
