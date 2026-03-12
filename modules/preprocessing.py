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
from sklearn.decomposition import PCA

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

        # Plot frame (placeholder)
        self.plot_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"], height=400)
        self.plot_frame.grid(row=3, column=0, sticky="nsew", pady=20, padx=20)
        
        self.built_ui = False

    def on_show(self):
        if not get_state("data_loaded"):
            self._show_blocked("Load data and select columns first.\n← Go to 'Data Loading'")
            return
            
        if not self.built_ui:
            self._build_ui()
            self.built_ui = True
            
        # If preprocessed already, ensure buttons are in correct state
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
            
        # Plot frame
        self.plot_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"], height=550)
        self.plot_frame.grid(row=3, column=0, sticky="nsew", pady=20, padx=20)
        self.plot_frame.pack_propagate(False)
            
        # -- Settings Controls --
        settings_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        settings_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        settings_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Scaling & PCA Settings
        s_col = ctk.CTkFrame(settings_frame, fg_color="transparent")
        s_col.grid(row=0, column=0, sticky="nsew", padx=10)

        ctk.CTkLabel(s_col, text="Normalization", font=FONTS["header"]).grid(row=0, column=0, pady=(10, 5), sticky="w")
        self.scaling_var = ctk.StringVar(value="Min-Max (0, 1)")
        scalers = ["Min-Max (0, 1)", "Standard (Z-score)", "None"]
        ctk.CTkComboBox(s_col, values=scalers, variable=self.scaling_var).grid(row=1, column=0, pady=(0, 10), sticky="ew")
        
        self.scale_tgt_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(s_col, text="Scale Target Variables", variable=self.scale_tgt_var).grid(row=2, column=0, pady=(0, 10), sticky="w")
        
        ctk.CTkLabel(s_col, text="PCA (Inputs)", font=FONTS["header"]).grid(row=3, column=0, pady=(10, 5), sticky="w")
        self.pca_x_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(s_col, text="Enable Input PCA", variable=self.pca_x_var).grid(row=4, column=0, pady=(0, 5), sticky="w")
        
        self.pca_x_comp = ctk.CTkEntry(s_col, placeholder_text="Num Components")
        self.pca_x_comp.insert(0, "3")
        self.pca_x_comp.grid(row=5, column=0, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(s_col, text="PCA (Outputs)", font=FONTS["header"]).grid(row=6, column=0, pady=(10, 5), sticky="w")
        self.pca_y_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(s_col, text="Enable Output PCA", variable=self.pca_y_var).grid(row=7, column=0, pady=(0, 5), sticky="w")
        
        self.pca_y_comp = ctk.CTkEntry(s_col, placeholder_text="Num Components")
        self.pca_y_comp.insert(0, "2")
        self.pca_y_comp.grid(row=8, column=0, pady=(0, 10), sticky="ew")

        # Split Settings
        sp_col = ctk.CTkFrame(settings_frame, fg_color="transparent")
        sp_col.grid(row=0, column=1, sticky="nsew", padx=10)

        ctk.CTkLabel(sp_col, text="Data Split", font=FONTS["header"]).grid(row=0, column=0, pady=(10, 15), sticky="w")
        
        self.train_pct = ctk.DoubleVar(value=0.7)
        self.val_pct = ctk.DoubleVar(value=0.15)
        
        ctk.CTkLabel(sp_col, text="Train %").grid(row=1, column=0, sticky="w")
        ctk.CTkSlider(sp_col, variable=self.train_pct, from_=0.1, to=0.9, number_of_steps=80, command=self._update_splits).grid(row=2, column=0, sticky="ew")
        self.train_lbl = ctk.CTkLabel(sp_col, text="70%")
        self.train_lbl.grid(row=3, column=0, sticky="e")
        
        ctk.CTkLabel(sp_col, text="Val %").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ctk.CTkSlider(sp_col, variable=self.val_pct, from_=0.05, to=0.5, number_of_steps=45, command=self._update_splits).grid(row=5, column=0, sticky="ew")
        self.val_lbl = ctk.CTkLabel(sp_col, text="15%")
        self.val_lbl.grid(row=6, column=0, sticky="e")
        
        self.test_lbl = ctk.CTkLabel(sp_col, text="Test % : 15%", text_color=COLORS["magenta"], font=("Helvetica", 14, "bold"))
        self.test_lbl.grid(row=7, column=0, sticky="w", pady=(20, 10))
        
        # Plot Options
        pl_col = ctk.CTkFrame(settings_frame, fg_color="transparent")
        pl_col.grid(row=0, column=2, sticky="nsew", padx=10)

        ctk.CTkLabel(pl_col, text="Charts", font=FONTS["header"]).grid(row=0, column=0, pady=(10, 15), sticky="w")
        
        self.chart_type = ctk.StringVar(value="Correlation Heatmap")
        ctk.CTkRadioButton(pl_col, text="Correlation Heatmap", variable=self.chart_type, value="Correlation Heatmap").grid(row=1, column=0, pady=(0, 10), sticky="w")
        ctk.CTkRadioButton(pl_col, text="Pair Plot (Distributions)", variable=self.chart_type, value="Pair Plot").grid(row=2, column=0, pady=(0, 10), sticky="w")

        # Actions
        actions_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        actions_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        
        self.run_btn = ctk.CTkButton(actions_frame, text="⚡ RUN PREPROCESSING & PLOT", height=40, font=("Helvetica", 14, "bold"), command=self.run_preprocessing)
        self.run_btn.pack(side="left")
        
        self.status_lbl = ctk.CTkLabel(actions_frame, text="Ready to run.", font=("Helvetica", 14), text_color=COLORS["text_dim"])
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
        self.proceed_btn.grid(row=4, column=0, pady=30, padx=20, sticky="ew")

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
        output_cols = get_state("output_column") # This is now a list!
        
        # Drop NaNs
        full_cols = input_cols + output_cols
        df = df.dropna(subset=full_cols).copy()
        
        X = df[input_cols].values
        y = df[output_cols].values
        
        # 1. Scaling
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

        # 2. PCA
        pca_X = None
        pca_y = None
        try:
            if self.pca_x_var.get():
                cx = int(self.pca_x_comp.get())
                pca_X = PCA(n_components=cx)
                X = pca_X.fit_transform(X)
                
            if self.pca_y_var.get():
                cy = int(self.pca_y_comp.get())
                pca_y = PCA(n_components=cy)
                y = pca_y.fit_transform(y)
        except Exception as e:
            self.status_lbl.configure(text=f"PCA Error: {e}", text_color=COLORS["red"])
            return
                
        # 3. Split (Math)
        train_p = self.train_pct.get()
        val_p = self.val_pct.get()
        test_p = max(0.01, 1.0 - train_p - val_p)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1.0 - train_p), random_state=42
        )
        relative_val = val_p / (val_p + test_p)
        X_val, X_test, y_val, y_test, df_train, df_temp = train_test_split(
            X_temp, y_temp, df.drop(df.index[:len(X_train)]), test_size=(1.0 - relative_val), random_state=42
        )
        # We don't save the split DFs properly because we scaled X and Y.
        # But wait, we just need the preprocessed matrices.
        
        # Save state for TF
        set_state("X_train", X_train)
        set_state("X_val", X_val)
        set_state("X_test", X_test)
        
        set_state("y_train", y_train)
        set_state("y_val", y_val)
        set_state("y_test", y_test)
        
        set_state("scaler_X", scaler_X)
        set_state("scaler_y", scaler_y)
        set_state("pca_X", pca_X)
        set_state("pca_y", pca_y)
        
        set_state("preprocessed", True)
        
        self.status_lbl.configure(text=f"✓ Split: {len(X_train)} Tr | {len(X_val)} Val | {len(X_test)} Te", text_color=COLORS["green"])
        self.proceed_btn.configure(state="normal")
        
        # Draw plot (on Original un-PCA'd data for interpretability)
        self._draw_plot(df, input_cols, output_cols)

    def _draw_plot(self, df, input_cols, output_cols):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        plot_df = df[input_cols + output_cols]
        chart = self.chart_type.get()
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 6), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(COLORS["bg_card"])
        
        if chart == "Correlation Heatmap":
            corr = plot_df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, 
                        cbar_kws={'label': 'Pearson Correlation'})
            ax.set_title("Correlation Matrix", color="white")
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        elif chart == "Pair Plot":
            # Very heavy if > 5 variables
            plt.close(fig)
            cols = list(plot_df.columns[:6]) # Cap at 6 to prevent lockups
            g = sns.pairplot(
                plot_df[cols], 
                diag_kind="hist",
                plot_kws={'alpha':0.6, 'color': COLORS['cyan'], 's':15},
                diag_kws={'color': COLORS['magenta']},
                corner=True
            )
            g.fig.set_facecolor(COLORS["bg_card"])
            
            canvas = FigureCanvasTkAgg(g.fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

    def go_to_model_builder(self):
        self.master.navigate_to("🏗  Model Builder")
