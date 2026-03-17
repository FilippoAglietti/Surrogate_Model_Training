import customtkinter as ctk
import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from utils.theme import COLORS, FONTS
from utils.state import get_state, set_state
from utils.plot_utils import add_save_button


class PreprocessingFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkLabel(self, text="PREPROCESSING 🔧", font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(30, 20), sticky="w", padx=30)

        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)

        self.plot_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        self.plot_frame.grid(row=3, column=0, sticky="nsew", pady=20, padx=20)

        self.built_ui = False

    # ─── lifecycle ────────────────────────────────────────────────────────────

    def on_show(self):
        if not get_state("data_loaded"):
            self._show_blocked("Load data and select columns first.\n← Go to 'Data Loading'")
            return
        if not self.built_ui:
            self._build_ui()
            self.built_ui = True
            # Restore widget values from a loaded session if available
            cfg = get_state("preprocessing_config")
            if cfg:
                self._restore_widgets(cfg)
        if get_state("preprocessed"):
            self.proceed_btn.configure(state="normal")
            self.status_lbl.configure(text="✓ Data preprocessed and split.", text_color=COLORS["green"])

    # ── Session helpers ────────────────────────────────────────────────────────

    def get_session_config(self) -> dict:
        """Snapshot current widget values for session serialization."""
        return {
            "scaling_method":  self.scaling_var.get(),
            "scale_targets":   self.scale_tgt_var.get(),
            "pca_x_enabled":   self.pca_x_var.get(),
            "pca_x_components": self.pca_x_comp.get(),
            "pca_y_enabled":   self.pca_y_var.get(),
            "pca_y_components": self.pca_y_comp.get(),
            "train_pct":       self.train_pct.get(),
            "val_pct":         self.val_pct.get(),
            "chart_combined":  self.chart_combined.get(),
            "chart_box":       self.chart_box.get(),
            "chart_kde":       self.chart_kde.get(),
            "chart_parallel":  self.chart_parallel.get(),
            "chart_outlier":   self.chart_outlier.get(),
        }

    def restore_from_session(self, cfg: dict) -> None:
        """Build the UI if needed, then populate widgets from *cfg*."""
        if not self.built_ui:
            self._build_ui()
            self.built_ui = True
        self._restore_widgets(cfg)
        if get_state("preprocessed"):
            self.proceed_btn.configure(state="normal")
            X_train = get_state("X_train")
            X_val   = get_state("X_val")
            X_test  = get_state("X_test")
            n_tr = len(X_train) if X_train is not None else 0
            n_va = len(X_val)   if X_val   is not None else 0
            n_te = len(X_test)  if X_test  is not None else 0
            self.status_lbl.configure(
                text=f"✓ Restored: {n_tr} Train | {n_va} Val | {n_te} Test",
                text_color=COLORS["green"],
            )
            self._restore_plots()

    def _restore_plots(self) -> None:
        """Re-draw preprocessing plots from saved state data."""
        plot_df        = get_state("plot_df")
        plot_in_cols   = get_state("plot_input_cols")
        plot_out_cols  = get_state("plot_output_cols")
        if plot_df is not None and plot_in_cols and plot_out_cols:
            self._draw_all_plots(plot_df, plot_in_cols, plot_out_cols)

    def _restore_widgets(self, cfg: dict) -> None:
        """Apply a preprocessing_config dict to all widget vars."""
        if "scaling_method" in cfg:
            self.scaling_var.set(cfg["scaling_method"])
        if "scale_targets" in cfg:
            self.scale_tgt_var.set(cfg["scale_targets"])
        if "pca_x_enabled" in cfg:
            self.pca_x_var.set(cfg["pca_x_enabled"])
        if "pca_x_components" in cfg:
            self.pca_x_comp.delete(0, "end")
            self.pca_x_comp.insert(0, str(cfg["pca_x_components"]))
        if "pca_y_enabled" in cfg:
            self.pca_y_var.set(cfg["pca_y_enabled"])
        if "pca_y_components" in cfg:
            self.pca_y_comp.delete(0, "end")
            self.pca_y_comp.insert(0, str(cfg["pca_y_components"]))
        if "train_pct" in cfg:
            self.train_pct.set(cfg["train_pct"])
        if "val_pct" in cfg:
            self.val_pct.set(cfg["val_pct"])
        if "chart_combined" in cfg:
            self.chart_combined.set(cfg["chart_combined"])
        if "chart_box" in cfg:
            self.chart_box.set(cfg["chart_box"])
        if "chart_kde" in cfg:
            self.chart_kde.set(cfg["chart_kde"])
        if "chart_parallel" in cfg:
            self.chart_parallel.set(cfg["chart_parallel"])
        if "chart_outlier" in cfg:
            self.chart_outlier.set(cfg["chart_outlier"])
        self._update_splits()

    def _show_blocked(self, message):
        for w in self.content_frame.winfo_children():
            w.destroy()
        self.built_ui = False
        ctk.CTkLabel(self.content_frame, text=f"[ BLOCKED ]\n{message}",
                     font=FONTS["header"], text_color=COLORS["red"]).grid(row=0, column=0, pady=50, padx=20)

    # ─── UI build ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        for w in self.content_frame.winfo_children():
            w.destroy()

        # Refresh plot_frame
        self.plot_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        self.plot_frame.grid(row=3, column=0, sticky="nsew", pady=20, padx=20)

        # ── Settings panel ─────────────────────────────────────────────────
        settings = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        settings.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        settings.grid_columnconfigure((0, 1, 2), weight=1)

        # ─ Column 0: Scaling + PCA ─────────────────────────────────────────
        s_col = ctk.CTkFrame(settings, fg_color="transparent")
        s_col.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(s_col, text="Normalization", font=FONTS["header"]).grid(row=0, column=0, pady=(10, 5), sticky="w")
        self.scaling_var = ctk.StringVar(value="Min-Max (0, 1)")
        ctk.CTkComboBox(s_col, values=["Min-Max (0, 1)", "Standard (Z-score)", "None"],
                        variable=self.scaling_var).grid(row=1, column=0, pady=(0, 8), sticky="ew")

        self.scale_tgt_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(s_col, text="Scale Target Variables", variable=self.scale_tgt_var).grid(row=2, column=0, pady=(0, 12), sticky="w")

        # PCA Inputs
        ctk.CTkLabel(s_col, text="PCA (Inputs)", font=FONTS["header"]).grid(row=3, column=0, pady=(8, 5), sticky="w")
        self.pca_x_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(s_col, text="Enable Input PCA", variable=self.pca_x_var).grid(row=4, column=0, pady=(0, 4), sticky="w")

        pca_x_row = ctk.CTkFrame(s_col, fg_color="transparent")
        pca_x_row.grid(row=5, column=0, sticky="ew")
        self.pca_x_comp = ctk.CTkEntry(pca_x_row, placeholder_text="Num Components", width=120)
        self.pca_x_comp.insert(0, "3")
        self.pca_x_comp.pack(side="left")
        ctk.CTkButton(pca_x_row, text="Scree", width=70, height=28,
                      fg_color=COLORS["bg_card_hover"],
                      command=lambda: self._show_scree("X")).pack(side="left", padx=6)

        # PCA Outputs
        ctk.CTkLabel(s_col, text="PCA (Outputs)", font=FONTS["header"]).grid(row=6, column=0, pady=(12, 5), sticky="w")
        self.pca_y_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(s_col, text="Enable Output PCA", variable=self.pca_y_var).grid(row=7, column=0, pady=(0, 4), sticky="w")

        pca_y_row = ctk.CTkFrame(s_col, fg_color="transparent")
        pca_y_row.grid(row=8, column=0, sticky="ew")
        self.pca_y_comp = ctk.CTkEntry(pca_y_row, placeholder_text="Num Components", width=120)
        self.pca_y_comp.insert(0, "2")
        self.pca_y_comp.pack(side="left")
        ctk.CTkButton(pca_y_row, text="Scree", width=70, height=28,
                      fg_color=COLORS["bg_card_hover"],
                      command=lambda: self._show_scree("Y")).pack(side="left", padx=6)

        # ─ Column 1: Data Split ────────────────────────────────────────────
        sp_col = ctk.CTkFrame(settings, fg_color="transparent")
        sp_col.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(sp_col, text="Data Split", font=FONTS["header"]).grid(row=0, column=0, pady=(10, 15), sticky="w")
        self.train_pct = ctk.DoubleVar(value=0.7)
        self.val_pct   = ctk.DoubleVar(value=0.15)

        ctk.CTkLabel(sp_col, text="Train %").grid(row=1, column=0, sticky="w")
        ctk.CTkSlider(sp_col, variable=self.train_pct, from_=0.1, to=0.9, number_of_steps=80,
                      command=self._update_splits).grid(row=2, column=0, sticky="ew")
        self.train_lbl = ctk.CTkLabel(sp_col, text="70%")
        self.train_lbl.grid(row=3, column=0, sticky="e")

        ctk.CTkLabel(sp_col, text="Val %").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ctk.CTkSlider(sp_col, variable=self.val_pct, from_=0.05, to=0.5, number_of_steps=45,
                      command=self._update_splits).grid(row=5, column=0, sticky="ew")
        self.val_lbl = ctk.CTkLabel(sp_col, text="15%")
        self.val_lbl.grid(row=6, column=0, sticky="e")

        self.test_lbl = ctk.CTkLabel(sp_col, text="Test % : 15%",
                                     text_color=COLORS["magenta"], font=("Helvetica", 14, "bold"))
        self.test_lbl.grid(row=7, column=0, sticky="w", pady=(20, 10))

        # ─ Column 2: Chart Selection ───────────────────────────────────────
        pl_col = ctk.CTkFrame(settings, fg_color="transparent")
        pl_col.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(pl_col, text="Charts to Generate", font=FONTS["header"]).grid(row=0, column=0, pady=(10, 10), sticky="w")

        self.chart_combined   = ctk.BooleanVar(value=True)
        self.chart_box        = ctk.BooleanVar(value=False)
        self.chart_kde        = ctk.BooleanVar(value=False)
        self.chart_parallel   = ctk.BooleanVar(value=False)
        self.chart_outlier    = ctk.BooleanVar(value=False)

        ctk.CTkCheckBox(pl_col, text="Combined Matrix (Scatter + Correlation)",
                        variable=self.chart_combined).grid(row=1, column=0, pady=4, sticky="w")
        ctk.CTkCheckBox(pl_col, text="Box / Violin Plot",
                        variable=self.chart_box).grid(row=2, column=0, pady=4, sticky="w")
        ctk.CTkCheckBox(pl_col, text="KDE Distributions",
                        variable=self.chart_kde).grid(row=3, column=0, pady=4, sticky="w")
        ctk.CTkCheckBox(pl_col, text="Parallel Coordinates",
                        variable=self.chart_parallel,
                        command=self._toggle_parallel_picker).grid(row=4, column=0, pady=4, sticky="w")
        ctk.CTkCheckBox(pl_col, text="Outlier Detection (IQR)",
                        variable=self.chart_outlier).grid(row=5, column=0, pady=4, sticky="w")

        # Parallel coordinates column picker (shown when checkbox is checked)
        self._parallel_picker_frame = ctk.CTkFrame(pl_col, fg_color=COLORS["bg"])
        self._parallel_picker_frame.grid(row=6, column=0, sticky="ew", pady=(4, 0))
        self._parallel_picker_frame.grid_remove()
        self._parallel_col_vars = {}

        # ── Action bar ─────────────────────────────────────────────────────
        actions = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        actions.grid(row=1, column=0, sticky="ew", padx=20, pady=10)

        self.run_btn = ctk.CTkButton(actions, text="⚡ RUN PREPROCESSING & PLOT",
                                     height=40, font=("Helvetica", 14, "bold"),
                                     command=self.run_preprocessing)
        self.run_btn.pack(side="left")

        self.status_lbl = ctk.CTkLabel(actions, text="Ready to run.",
                                       font=("Helvetica", 14), text_color=COLORS["text_dim"])
        self.status_lbl.pack(side="left", padx=20)

        # ── Proceed button ─────────────────────────────────────────────────
        self.proceed_btn = ctk.CTkButton(
            self.content_frame, text="Proceed to Model Builder →",
            font=FONTS["header"], height=50,
            fg_color=COLORS["primary_dark"], hover_color=COLORS["primary"],
            text_color="#000", state="disabled",
            command=self.go_to_model_builder
        )
        self.proceed_btn.grid(row=4, column=0, pady=30, padx=20, sticky="ew")

    # ─── parallel coords picker ───────────────────────────────────────────────

    def _toggle_parallel_picker(self):
        if not self.chart_parallel.get():
            self._parallel_picker_frame.grid_remove()
            return

        # Rebuild picker from current columns
        for w in self._parallel_picker_frame.winfo_children():
            w.destroy()
        self._parallel_col_vars.clear()

        input_cols  = get_state("input_columns") or []
        output_cols = get_state("output_column")  or []
        all_cols = input_cols + output_cols

        ctk.CTkLabel(self._parallel_picker_frame, text="Select columns for Parallel Coordinates:",
                     font=("Helvetica", 11), text_color=COLORS["text_dim"]).pack(anchor="w", padx=6, pady=(4, 2))

        inner = ctk.CTkScrollableFrame(self._parallel_picker_frame, height=120, fg_color="transparent")
        inner.pack(fill="x", padx=4)

        for col in all_cols:
            var = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(inner, text=col, variable=var).pack(anchor="w", pady=1)
            self._parallel_col_vars[col] = var

        self._parallel_picker_frame.grid()

    # ─── split labels ─────────────────────────────────────────────────────────

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

    # ─── scree plot ───────────────────────────────────────────────────────────

    def _show_scree(self, axis):
        df          = get_state("df")
        input_cols  = get_state("input_columns") or []
        output_cols = get_state("output_column")  or []

        if axis == "X":
            cols = input_cols
            label = "Inputs"
        else:
            cols = output_cols
            label = "Outputs"

        if not cols:
            self.status_lbl.configure(text="No columns available for Scree Plot.", text_color=COLORS["red"])
            return

        data = df[cols].dropna().values

        # Quick standard scale for PCA
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

        max_comp = min(len(cols), len(data))
        pca_full = PCA(n_components=max_comp)
        pca_full.fit(data)

        evr       = pca_full.explained_variance_ratio_
        cumulative = np.cumsum(evr)
        x_ticks   = np.arange(1, max_comp + 1)

        # Build popup window
        win = tk.Toplevel()
        win.title(f"Scree Plot — {label}")
        win.configure(bg=COLORS["bg_card"])
        win.geometry("820x580")

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax.set_facecolor(COLORS["bg_card"])

        ax.bar(x_ticks, evr * 100, color=COLORS["cyan"], alpha=0.75, label="Individual variance")
        ax.plot(x_ticks, cumulative * 100, color=COLORS["orange"], marker="o",
                linewidth=2, label="Cumulative variance")

        # 90% and 95% reference lines
        for thresh, col in [(90, COLORS["green"]), (95, COLORS["magenta"])]:
            ax.axhline(thresh, color=col, linestyle="--", linewidth=1, alpha=0.7, label=f"{thresh}% line")

        ax.set_xlabel("Number of Components", color=COLORS["text"])
        ax.set_ylabel("Explained Variance (%)", color=COLORS["text"])
        ax.set_title(f"PCA Scree Plot — {label}", color="white", fontsize=12)
        ax.tick_params(colors=COLORS["text"])
        ax.set_xticks(x_ticks)
        for spine in ax.spines.values(): spine.set_color(COLORS["border"])
        ax.legend(fontsize=9)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        add_save_button(win, canvas, "scree_plot.png")

    # ─── core preprocessing ───────────────────────────────────────────────────

    def run_preprocessing(self):
        df          = get_state("df")
        input_cols  = get_state("input_columns")
        output_cols = get_state("output_column")

        full_cols = input_cols + output_cols
        df = df.dropna(subset=full_cols).copy()

        X = df[input_cols].values.astype(float)
        y = df[output_cols].values.astype(float)

        # 1. Scaling
        scaler_X = scaler_y = None
        scale_type = self.scaling_var.get()
        if scale_type == "Min-Max (0, 1)":
            scaler_X = MinMaxScaler(); X = scaler_X.fit_transform(X)
            if self.scale_tgt_var.get():
                scaler_y = MinMaxScaler(); y = scaler_y.fit_transform(y)
        elif scale_type == "Standard (Z-score)":
            scaler_X = StandardScaler(); X = scaler_X.fit_transform(X)
            if self.scale_tgt_var.get():
                scaler_y = StandardScaler(); y = scaler_y.fit_transform(y)

        # 2. PCA
        pca_X = pca_y = None
        try:
            if self.pca_x_var.get():
                cx = int(self.pca_x_comp.get())
                pca_X = PCA(n_components=cx); X = pca_X.fit_transform(X)
            if self.pca_y_var.get():
                cy = int(self.pca_y_comp.get())
                pca_y = PCA(n_components=cy); y = pca_y.fit_transform(y)
        except Exception as e:
            self.status_lbl.configure(text=f"PCA Error: {e}", text_color=COLORS["red"])
            return

        # 3. Split
        train_p = self.train_pct.get()
        val_p   = self.val_pct.get()
        test_p  = max(0.01, 1.0 - train_p - val_p)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1.0 - train_p), random_state=42)
        rel_val = val_p / (val_p + test_p)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1.0 - rel_val), random_state=42)

        set_state("X_train", X_train); set_state("X_val", X_val); set_state("X_test", X_test)
        set_state("y_train", y_train); set_state("y_val", y_val); set_state("y_test", y_test)
        set_state("scaler_X", scaler_X); set_state("scaler_y", scaler_y)
        set_state("pca_X", pca_X); set_state("pca_y", pca_y)
        set_state("preprocessed", True)
        set_state("preprocessing_config", self.get_session_config())
        set_state("session_unsaved", True)

        self.status_lbl.configure(
            text=f"✓ Split: {len(X_train)} Train | {len(X_val)} Val | {len(X_test)} Test",
            text_color=COLORS["green"])
        self.proceed_btn.configure(state="normal")

        # 4. Draw selected plots (PCA-space if PCA was applied, otherwise original)
        if self.pca_x_var.get() and pca_X is not None:
            pc_x_cols = [f"PC_X_{i+1}" for i in range(X.shape[1])]
            plot_input_df = pd.DataFrame(X, columns=pc_x_cols)
            plot_input_cols = pc_x_cols
        else:
            plot_input_df = df[input_cols].reset_index(drop=True)
            plot_input_cols = input_cols

        if self.pca_y_var.get() and pca_y is not None:
            pc_y_cols = [f"PC_Y_{i+1}" for i in range(y.shape[1])]
            plot_output_df = pd.DataFrame(y, columns=pc_y_cols)
            plot_output_cols = pc_y_cols
        else:
            plot_output_df = df[output_cols].reset_index(drop=True)
            plot_output_cols = output_cols

        plot_df = pd.concat([plot_input_df, plot_output_df], axis=1)
        set_state("plot_df", plot_df)
        set_state("plot_input_cols", plot_input_cols)
        set_state("plot_output_cols", plot_output_cols)
        self._draw_all_plots(plot_df, plot_input_cols, plot_output_cols)

    # ─── plot dispatcher ──────────────────────────────────────────────────────

    def _draw_all_plots(self, df, input_cols, output_cols):
        for w in self.plot_frame.winfo_children():
            w.destroy()

        tabs_to_make = []
        if self.chart_combined.get():  tabs_to_make.append("Combined Matrix")
        if self.chart_box.get():       tabs_to_make.append("Box / Violin")
        if self.chart_kde.get():       tabs_to_make.append("KDE Distributions")
        if self.chart_parallel.get():  tabs_to_make.append("Parallel Coordinates")
        if self.chart_outlier.get():   tabs_to_make.append("Outlier Detection")

        if not tabs_to_make:
            ctk.CTkLabel(self.plot_frame, text="No charts selected.", text_color=COLORS["text_dim"],
                         font=FONTS["header"]).pack(pady=30)
            return

        tab_h = 780 if len(tabs_to_make) > 1 else 700
        tv = ctk.CTkTabview(self.plot_frame, fg_color=COLORS["bg"], height=tab_h)
        tv.pack(fill="both", expand=True, padx=10, pady=10)

        for t in tabs_to_make:
            tv.add(t)

        all_cols = input_cols + output_cols

        if "Combined Matrix" in tabs_to_make:
            self._draw_combined_matrix(tv.tab("Combined Matrix"), df, all_cols)
        if "Box / Violin" in tabs_to_make:
            self._draw_box_violin(tv.tab("Box / Violin"), df, all_cols)
        if "KDE Distributions" in tabs_to_make:
            self._draw_kde(tv.tab("KDE Distributions"), df, all_cols)
        if "Parallel Coordinates" in tabs_to_make:
            sel = [c for c, v in self._parallel_col_vars.items() if v.get()] if self._parallel_col_vars else all_cols
            self._draw_parallel(tv.tab("Parallel Coordinates"), df, sel, output_cols)
        if "Outlier Detection" in tabs_to_make:
            self._draw_outlier_detection(tv.tab("Outlier Detection"), df, input_cols, output_cols)

    # ─── Combined Matrix (scatter below diag + hist diag + corr above diag) ───

    def _draw_combined_matrix(self, parent, df, cols):
        # Cap at 8 columns for readability
        cols = cols[:8]
        n = len(cols)
        plot_df = df[cols].dropna()
        corr = plot_df.corr()

        plt.style.use("dark_background")
        fig, axes = plt.subplots(n, n, figsize=(max(8, n * 2.2), max(8, n * 2.2)), dpi=90)
        fig.patch.set_facecolor(COLORS["bg_card"])
        if n == 1:
            axes = [[axes]]

        cmap_val = plt.cm.RdYlGn

        for i in range(n):
            for j in range(n):
                ax = axes[i][j]
                ax.set_facecolor(COLORS["bg_card"])
                ax.tick_params(colors=COLORS["text"], labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color(COLORS["border"])

                xi, xj = plot_df[cols[j]].values, plot_df[cols[i]].values

                if i == j:
                    # Diagonal: histogram + KDE line
                    ax.hist(xi, bins=20, color=COLORS["cyan"], alpha=0.6, density=True)
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(xi[np.isfinite(xi)])
                        xs = np.linspace(xi.min(), xi.max(), 100)
                        ax.plot(xs, kde(xs), color=COLORS["orange"], linewidth=1.5)
                    except Exception:
                        pass
                    ax.set_xlabel(cols[j], color=COLORS["text"], fontsize=8)

                elif i > j:
                    # Lower triangle: scatter
                    ax.scatter(xi, xj, s=4, alpha=0.5, color=COLORS["cyan"])
                    # Regression line
                    try:
                        m, b = np.polyfit(xi[np.isfinite(xi) & np.isfinite(xj)],
                                          xj[np.isfinite(xi) & np.isfinite(xj)], 1)
                        xs = np.array([xi.min(), xi.max()])
                        ax.plot(xs, m * xs + b, color=COLORS["orange"], linewidth=1, alpha=0.8)
                    except Exception:
                        pass

                else:
                    # Upper triangle: correlation value as colored rectangle
                    r = corr.loc[cols[i], cols[j]]
                    norm_r = (r + 1) / 2  # map [-1,1] -> [0,1]
                    bg_color = cmap_val(norm_r)
                    ax.set_facecolor(bg_color)
                    ax.text(0.5, 0.5, f"{r:.2f}", ha="center", va="center",
                            transform=ax.transAxes,
                            fontsize=max(8, 14 - n),
                            fontweight="bold",
                            color="black" if 0.3 < norm_r < 0.7 else "white")
                    ax.set_xticks([]); ax.set_yticks([])

                # Labels on edges only
                if j > 0: ax.set_yticks([])
                if i < n - 1: ax.set_xticks([])

        plt.tight_layout(pad=0.5)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        add_save_button(parent, canvas, "combined_matrix.png")
        plt.close(fig)

    # ─── Box / Violin ─────────────────────────────────────────────────────────

    def _draw_box_violin(self, parent, df, cols):
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            ctk.CTkLabel(parent, text="No numeric columns to plot.", text_color=COLORS["red"]).pack(pady=20)
            return

        plot_df = df[cols].copy()
        # Normalize each column to [0,1] so they're comparable on same scale
        normed = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min() + 1e-10)

        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(cols) * 0.9), 5), dpi=90)
        fig.patch.set_facecolor(COLORS["bg_card"])

        palette = [COLORS["cyan"], COLORS["magenta"], COLORS["orange"],
                   COLORS["green"], COLORS["yellow"]]
        col_palette = [palette[i % len(palette)] for i in range(len(cols))]

        for ax in [ax1, ax2]:
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text"], labelsize=9)
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])

        # Box plot
        bp = ax1.boxplot(normed.values, patch_artist=True, medianprops={"color": "white", "linewidth": 2})
        for patch, color in zip(bp["boxes"], col_palette):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        for whisker in bp["whiskers"]: whisker.set_color(COLORS["text_dim"])
        for cap in bp["caps"]: cap.set_color(COLORS["text_dim"])
        for flier in bp["fliers"]: flier.set(marker="o", color=COLORS["red"], alpha=0.5, markersize=4)
        ax1.set_xticks(range(1, len(cols) + 1))
        ax1.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax1.set_title("Box Plot (normalized)", color="white", fontsize=11)
        ax1.set_ylabel("Value (normalized)", color=COLORS["text"])

        # Violin plot
        vp = ax2.violinplot(normed.values, positions=range(1, len(cols) + 1),
                            showmedians=True, showextrema=True)
        for pc, color in zip(vp["bodies"], col_palette):
            pc.set_facecolor(color); pc.set_alpha(0.55)
        vp["cmedians"].set_color("white")
        vp["cmaxes"].set_color(COLORS["text_dim"])
        vp["cmins"].set_color(COLORS["text_dim"])
        vp["cbars"].set_color(COLORS["text_dim"])
        ax2.set_xticks(range(1, len(cols) + 1))
        ax2.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax2.set_title("Violin Plot (normalized)", color="white", fontsize=11)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        add_save_button(parent, canvas, "box_violin.png")
        plt.close(fig)

    # ─── KDE Distributions ────────────────────────────────────────────────────

    def _draw_kde(self, parent, df, cols):
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            ctk.CTkLabel(parent, text="No numeric columns.", text_color=COLORS["red"]).pack(pady=20)
            return

        n = len(cols)
        ncols_grid = min(4, n)
        nrows_grid = int(np.ceil(n / ncols_grid))

        plt.style.use("dark_background")
        fig, axes = plt.subplots(nrows_grid, ncols_grid,
                                 figsize=(ncols_grid * 3.2, nrows_grid * 2.8), dpi=90)
        fig.patch.set_facecolor(COLORS["bg_card"])
        axes_flat = np.array(axes).flatten()

        palette = [COLORS["cyan"], COLORS["magenta"], COLORS["orange"],
                   COLORS["green"], COLORS["yellow"]]

        for i, col in enumerate(cols):
            ax = axes_flat[i]
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text"], labelsize=8)
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])

            data = df[col].dropna().values
            color = palette[i % len(palette)]
            ax.hist(data, bins=25, color=color, alpha=0.4, density=True)
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                xs = np.linspace(data.min(), data.max(), 200)
                ax.plot(xs, kde(xs), color=color, linewidth=2)
            except Exception:
                pass

            mean_v = data.mean()
            ax.axvline(mean_v, color="white", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_title(col, color="white", fontsize=9)

        for i in range(n, len(axes_flat)):
            axes_flat[i].axis("off")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        add_save_button(parent, canvas, "kde_distributions.png")
        plt.close(fig)

    # ─── Parallel Coordinates ─────────────────────────────────────────────────

    def _draw_parallel(self, parent, df, sel_cols, output_cols):
        if len(sel_cols) < 2:
            ctk.CTkLabel(parent, text="Select at least 2 columns.", text_color=COLORS["red"]).pack(pady=20)
            return

        plot_df = df[sel_cols].dropna().copy()
        if len(plot_df) > 500:
            plot_df = plot_df.sample(500, random_state=42)

        # Normalize to [0, 1] per column for display
        normed = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min() + 1e-10)

        # Color by first output column (if available in selection)
        color_col = None
        for oc in output_cols:
            if oc in sel_cols:
                color_col = oc
                break

        if color_col:
            raw_vals = plot_df[color_col].values
            norm_color = (raw_vals - raw_vals.min()) / (raw_vals.max() - raw_vals.min() + 1e-10)
        else:
            norm_color = np.linspace(0, 1, len(normed))

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(max(8, len(sel_cols) * 1.5), 5), dpi=90)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax.set_facecolor(COLORS["bg_card"])

        cmap = plt.cm.plasma
        x_ticks = list(range(len(sel_cols)))

        for i, row in enumerate(normed.values):
            color = cmap(norm_color[i])
            ax.plot(x_ticks, row, color=color, alpha=0.35, linewidth=0.8)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(sel_cols, rotation=30, ha="right", fontsize=9, color=COLORS["text"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["min", "25%", "50%", "75%", "max"], color=COLORS["text_dim"], fontsize=8)
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values(): spine.set_color(COLORS["border"])

        title = f"Parallel Coordinates  (color = {color_col})" if color_col else "Parallel Coordinates"
        ax.set_title(title, color="white", fontsize=11)

        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(colors=COLORS["text"], labelsize=8)
        if color_col:
            cbar.set_label(color_col, color=COLORS["text"], fontsize=9)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        add_save_button(parent, canvas, "parallel_coordinates.png")
        plt.close(fig)

    # ─── Outlier Detection ────────────────────────────────────────────────────

    def _draw_outlier_detection(self, parent, df, input_cols, output_cols):
        num_cols = [c for c in input_cols + output_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            ctk.CTkLabel(parent, text="No numeric columns for outlier detection.",
                         text_color=COLORS["red"]).pack(pady=20)
            return

        # IQR outlier mask per column
        Q1 = df[num_cols].quantile(0.25)
        Q3 = df[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask = ((df[num_cols] < lower) | (df[num_cols] > upper))
        outlier_per_col = outlier_mask.sum()
        is_outlier_row  = outlier_mask.any(axis=1)
        n_outlier_rows  = int(is_outlier_row.sum())
        pct = n_outlier_rows / len(df) * 100

        # Summary
        top_frame = ctk.CTkFrame(parent, fg_color="transparent")
        top_frame.pack(fill="x", padx=10, pady=8)

        color = COLORS["red"] if pct > 10 else COLORS["orange"] if pct > 3 else COLORS["green"]
        ctk.CTkLabel(top_frame,
                     text=f"Outliers detected: {n_outlier_rows} / {len(df)} rows  ({pct:.1f}%)  — IQR × 1.5",
                     font=("Helvetica", 13, "bold"), text_color=color).pack(side="left", padx=10)

        # Remove button
        self._df_for_outlier = df
        self._outlier_mask   = is_outlier_row

        def remove_outliers():
            clean_df = self._df_for_outlier[~self._outlier_mask].reset_index(drop=True)
            set_state("df", clean_df)
            set_state("preprocessed", False)
            self.proceed_btn.configure(state="disabled")
            self.status_lbl.configure(
                text=f"Removed {n_outlier_rows} outlier rows. Re-run preprocessing.",
                text_color=COLORS["orange"])
            for w in parent.winfo_children():
                w.destroy()
            ctk.CTkLabel(parent,
                         text=f"✓ {n_outlier_rows} outlier rows removed.\nRe-run preprocessing to apply.",
                         font=FONTS["header"], text_color=COLORS["green"]).pack(pady=40)

        ctk.CTkButton(top_frame, text="Remove Outliers from Dataset",
                      fg_color=COLORS["red"], hover_color="#AA0000", height=32,
                      command=remove_outliers).pack(side="left", padx=10)

        # Bar chart of outliers per column
        mv_df = outlier_per_col.sort_values(ascending=True)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8, max(2.5, len(mv_df) * 0.45 + 0.6)), dpi=90)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax.set_facecolor(COLORS["bg_card"])

        bar_colors = [COLORS["red"] if v > 0.1 * len(df) else COLORS["orange"]
                      if v > 0.03 * len(df) else COLORS["cyan"] for v in mv_df]
        bars = ax.barh(mv_df.index, mv_df.values, color=bar_colors, alpha=0.85)

        for bar, val in zip(bars, mv_df.values):
            pct_c = val / len(df) * 100
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{int(val)}  ({pct_c:.1f}%)", va="center", ha="left",
                    color=COLORS["text"], fontsize=9)

        ax.set_xlabel("Outlier Count", color=COLORS["text"])
        ax.set_xlim(0, max(mv_df.max() * 1.3, 1))
        ax.tick_params(colors=COLORS["text"], labelsize=9)
        for spine in ax.spines.values(): spine.set_color(COLORS["border"])
        ax.set_title("Outliers per Column (IQR × 1.5)", color="white", fontsize=11)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=4)
        add_save_button(parent, canvas, "outlier_detection.png")
        plt.close(fig)

    # ─── navigation ───────────────────────────────────────────────────────────

    def go_to_model_builder(self):
        self.master.navigate_to("🏗  Model Builder")
