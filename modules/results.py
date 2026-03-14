import customtkinter as ctk
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
import zipfile
import pickle
import os

from utils.theme import COLORS, FONTS
from utils.state import get_state


class ResultsFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self.header = ctk.CTkLabel(self, text="RESULTS & ANALYSIS 📊", font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(30, 20), sticky="w", padx=30)
        
        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        
        self.built_ui = False

    def on_show(self):
        if not get_state("trained"):
            self._show_blocked("Train a model first.\n← Go to 'Model Builder'")
            return
            
        if not self.built_ui:
            self._build_ui()
            self.built_ui = True

    def _show_blocked(self, message):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.built_ui = False
        lbl = ctk.CTkLabel(self.content_frame, text=f"[ BLOCKED ]\n{message}", font=FONTS["header"], text_color=COLORS["red"])
        lbl.grid(row=0, column=0, pady=50, padx=20)

    def _build_ui(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
        model = get_state("model")
        X_test = get_state("X_test")
        y_test = get_state("y_test")
        scaler_y = get_state("scaler_y")
        pca_y = get_state("pca_y")
        input_cols = get_state("input_columns")
        output_cols = get_state("output_column") # List
        
        y_pred = model.predict(X_test, verbose=0)
        y_true = y_test

        # Reverse PCA then Scaler
        if pca_y is not None:
            y_pred = pca_y.inverse_transform(y_pred)
            y_true = pca_y.inverse_transform(y_true)
            
        if scaler_y is not None:
            y_pred_orig = scaler_y.inverse_transform(y_pred)
            y_true_orig = scaler_y.inverse_transform(y_true)
        else:
            y_pred_orig = y_pred
            y_true_orig = y_true

        # If it happens to be 1D, make it 2D
        if len(y_pred_orig.shape) == 1:
            y_pred_orig = y_pred_orig.reshape(-1, 1)
            y_true_orig = y_true_orig.reshape(-1, 1)

        self.y_p = y_pred_orig
        self.y_t = y_true_orig
        self.out_cols = output_cols
        
        # Calculate overall metrics
        mse = mean_squared_error(self.y_t, self.y_p)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_t, self.y_p)
        r2_avg = r2_score(self.y_t, self.y_p) # Default is uniform_average
        r2_raw = r2_score(self.y_t, self.y_p, multioutput='raw_values')
        
        # Safe MAPE
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((self.y_t - self.y_p) / (self.y_t + 1e-10))) * 100
        
        # UI Metrics
        metrics_f = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        metrics_f.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        for i in range(5): metrics_f.grid_columnconfigure(i, weight=1)
        
        metrics = [("Avg MSE", f"{mse:.4f}"), ("Avg RMSE", f"{rmse:.4f}"), ("Avg MAE", f"{mae:.4f}"), ("Avg R²", f"{r2_avg:.4f}"), ("Avg MAPE %", f"{mape:.2f}")]
        for i, (k, v) in enumerate(metrics):
            ctk.CTkLabel(metrics_f, text=k, font=FONTS["header"], text_color=COLORS["cyan"]).grid(row=0, column=i, pady=(10,0))
            ctk.CTkLabel(metrics_f, text=v, font=FONTS["title"]).grid(row=1, column=i, pady=(0,10))

        # Show Per-Target R2
        if len(output_cols) > 1:
            r2_txt = " | ".join([f"{c}: {r:.4f}" for c, r in zip(output_cols, r2_raw)])
            ctk.CTkLabel(self.content_frame, text=f"Target R²: {r2_txt}", font=("Helvetica", 14), text_color=COLORS["green"]).grid(row=1, column=0, pady=(0, 10))

        # Plots
        self.tabview = ctk.CTkTabview(self.content_frame, fg_color=COLORS["bg_card"], height=600)
        self.tabview.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
        self.tabview.add("Pred vs Actual")
        self.tabview.add("Test Index Series")
        self.tabview.add("Residuals")
        self.tabview.add("Q-Q Plot")
        self.tabview.add("Per-Target Metrics")
        self.tabview.add("Worst Predictions")
        self.tabview.add("SHAP Values")
        self.tabview.add("Export & Wrapper")

        self._setup_pred_actual(self.tabview.tab("Pred vs Actual"))
        self._setup_series(self.tabview.tab("Test Index Series"))
        self._setup_residuals(self.tabview.tab("Residuals"))
        self._setup_qq(self.tabview.tab("Q-Q Plot"))
        self._setup_per_target_metrics(self.tabview.tab("Per-Target Metrics"))
        self._setup_worst_predictions(self.tabview.tab("Worst Predictions"))
        self._setup_shap(self.tabview.tab("SHAP Values"), model)

        # Export
        self._setup_export(self.tabview.tab("Export & Wrapper"), model)

    def _setup_pred_actual(self, parent):
        num_targets = len(self.out_cols)
        
        plot_frame = ctk.CTkFrame(parent, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True)
        
        if num_targets <= 4:
            # Draw grid
            self._draw_pred_grid(plot_frame, self.out_cols, self.y_t, self.y_p)
        else:
            # Draw dropdown + single plot
            top = ctk.CTkFrame(parent, fg_color="transparent")
            top.pack(fill="x", pady=5)
            ctk.CTkLabel(top, text="Select Target:").pack(side="left", padx=10)
            
            sel_var = ctk.StringVar(value=self.out_cols[0])
            cb = ctk.CTkComboBox(top, values=self.out_cols, variable=sel_var, command=lambda x: self._draw_single_pred(plot_frame, x, self.out_cols.index(x)))
            cb.pack(side="left", padx=10)
            
            self._draw_single_pred(plot_frame, self.out_cols[0], 0)

    def _draw_pred_grid(self, parent, cols, y_t, y_p):
        for w in parent.winfo_children(): w.destroy()
        
        num = len(cols)
        rows = 1 if num <= 2 else 2
        cols_grid = 1 if num == 1 else 2
        
        fig, axes = plt.subplots(rows, cols_grid, figsize=(9, 6), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        
        if num == 1:
            axes_flat = [axes]
        else:
            axes_flat = axes.flatten()
            
        for i in range(len(axes_flat)):
            ax = axes_flat[i]
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text"])
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])
            
            if i < num:
                ax.scatter(y_t[:, i], y_p[:, i], alpha=0.6, color=COLORS["cyan"], s=10)
                min_v, max_v = min(y_t[:, i].min(), y_p[:, i].min()), max(y_t[:, i].max(), y_p[:, i].max())
                ax.plot([min_v, max_v], [min_v, max_v], color=COLORS["green"], linestyle="--")
                ax.set_title(cols[i], color="white", fontsize=10)
            else:
                ax.axis('off')
                
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)

    def _draw_single_pred(self, parent, col_name, idx):
        for w in parent.winfo_children(): w.destroy()
        
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax.set_facecolor(COLORS["bg_card"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values(): spine.set_color(COLORS["border"])
        
        y_ti = self.y_t[:, idx]
        y_pi = self.y_p[:, idx]
        
        ax.scatter(y_ti, y_pi, alpha=0.6, color=COLORS["cyan"], s=15)
        min_v, max_v = min(y_ti.min(), y_pi.min()), max(y_ti.max(), y_pi.max())
        ax.plot([min_v, max_v], [min_v, max_v], color=COLORS["green"], linestyle="--")
        ax.set_title(f"Predicted vs Actual ({col_name})", color="white")
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
        
    def _setup_series(self, parent):
        num_targets = len(self.out_cols)
        
        plot_frame = ctk.CTkFrame(parent, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True)
        
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", pady=5, before=plot_frame)
        ctk.CTkLabel(top, text="Select Target:").pack(side="left", padx=10)
        
        sel_var = ctk.StringVar(value=self.out_cols[0])
        cb = ctk.CTkComboBox(top, values=self.out_cols, variable=sel_var, command=lambda x: self._draw_series(plot_frame, x, self.out_cols.index(x)))
        cb.pack(side="left", padx=10)
        
        self._draw_series(plot_frame, self.out_cols[0], 0)

    def _draw_series(self, parent, col_name, idx):
        for w in parent.winfo_children(): w.destroy()
        
        fig, ax = plt.subplots(figsize=(9, 4), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax.set_facecolor(COLORS["bg_card"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values(): spine.set_color(COLORS["border"])
        
        y_ti = self.y_t[:, idx]
        y_pi = self.y_p[:, idx]
        x_ax = np.arange(len(y_ti))
        
        # Sort by actual to make the plot readable
        sort_idx = np.argsort(y_ti)
        
        ax.plot(x_ax, y_ti[sort_idx], color=COLORS["orange"], label="Actual (Sorted)", linewidth=2)
        ax.scatter(x_ax, y_pi[sort_idx], color=COLORS["cyan"], label="Predicted", s=10, alpha=0.7)
        ax.set_title(f"{col_name} - Pred vs Real over Test Samples", color="white")
        ax.legend()
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)

    def _setup_residuals(self, parent):
        plot_frame = ctk.CTkFrame(parent, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True)
        
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", pady=5, before=plot_frame)
        ctk.CTkLabel(top, text="Select Target:").pack(side="left", padx=10)
        
        sel_var = ctk.StringVar(value=self.out_cols[0])
        cb = ctk.CTkComboBox(top, values=self.out_cols, variable=sel_var, command=lambda x: self._draw_single_res(plot_frame, x, self.out_cols.index(x)))
        cb.pack(side="left", padx=10)
        
        self._draw_single_res(plot_frame, self.out_cols[0], 0)

    def _draw_single_res(self, parent, col_name, idx):
        for w in parent.winfo_children(): w.destroy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        
        y_ti = self.y_t[:, idx]
        y_pi = self.y_p[:, idx]
        res = y_ti - y_pi
        
        for ax in [ax1, ax2]:
            fig.patch.set_facecolor(COLORS["bg_card"])
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text"])
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])
            
        ax1.scatter(y_pi, res, color=COLORS["orange"], alpha=0.6, s=15)
        ax1.axhline(0, color=COLORS["green"], linestyle="--")
        ax1.set_title(f"Residuals vs Pred ({col_name})", color="white")
        
        ax2.hist(res, bins=40, color=COLORS["magenta"], alpha=0.7)
        ax2.set_title("Residual Distribution", color="white")
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)

    # ─── Q-Q Plot ─────────────────────────────────────────────────────────────

    def _setup_qq(self, parent):
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", pady=5)
        ctk.CTkLabel(top, text="Select Target:").pack(side="left", padx=10)
        sel_var = ctk.StringVar(value=self.out_cols[0])
        plot_frame = ctk.CTkFrame(parent, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True)
        ctk.CTkComboBox(top, values=self.out_cols, variable=sel_var,
                        command=lambda x: self._draw_qq(plot_frame, x, self.out_cols.index(x))).pack(side="left", padx=10)
        self._draw_qq(plot_frame, self.out_cols[0], 0)

    def _draw_qq(self, parent, col_name, idx):
        for w in parent.winfo_children(): w.destroy()
        res = self.y_t[:, idx] - self.y_p[:, idx]

        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])

        for ax in [ax1, ax2]:
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text"])
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])

        # Q-Q plot
        (osm, osr), (slope, intercept, r) = stats.probplot(res, dist="norm")
        ax1.scatter(osm, osr, color=COLORS["cyan"], s=10, alpha=0.7, label="Residuals")
        line_x = np.array([osm.min(), osm.max()])
        ax1.plot(line_x, slope * line_x + intercept, color=COLORS["orange"], linewidth=2, label="Normal line")
        ax1.set_xlabel("Theoretical Quantiles", color=COLORS["text"])
        ax1.set_ylabel("Sample Quantiles", color=COLORS["text"])
        ax1.set_title(f"Q-Q Plot — {col_name}  (R={r:.3f})", color="white")
        ax1.legend(fontsize=9)

        # Residual histogram with normal overlay
        ax2.hist(res, bins=35, color=COLORS["magenta"], alpha=0.6, density=True, label="Residuals")
        mu, sigma = res.mean(), res.std()
        xs = np.linspace(res.min(), res.max(), 200)
        ax2.plot(xs, stats.norm.pdf(xs, mu, sigma), color=COLORS["cyan"], linewidth=2, label="Normal fit")
        ax2.axvline(0, color=COLORS["green"], linestyle="--", linewidth=1)
        ax2.set_title("Residual Distribution vs Normal", color="white")
        ax2.legend(fontsize=9)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
        plt.close(fig)

    # ─── Per-Target Metrics ───────────────────────────────────────────────────

    def _setup_per_target_metrics(self, parent):
        num_targets = len(self.out_cols)
        rows = []
        for i, col in enumerate(self.out_cols):
            yt = self.y_t[:, i]
            yp = self.y_p[:, i]
            rmse = np.sqrt(mean_squared_error(yt, yp))
            mae  = mean_absolute_error(yt, yp)
            r2   = r2_score(yt, yp)
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((yt - yp) / (yt + 1e-10))) * 100
            rows.append((col, r2, rmse, mae, mape))

        # Table header
        col_names = ["Target", "R²", "RMSE", "MAE", "MAPE %"]

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Metrics.Treeview",
                        background="#1A1A2E", foreground=COLORS["text"],
                        fieldbackground="#1A1A2E", rowheight=26, font=("Helvetica", 11))
        style.configure("Metrics.Treeview.Heading",
                        background=COLORS["bg_card"], foreground=COLORS["cyan"],
                        font=("Helvetica", 11, "bold"), relief="flat")
        style.map("Metrics.Treeview", background=[("selected", COLORS["primary_dark"])])

        container = tk.Frame(parent, bg="#1A1A2E")
        container.pack(fill="x", padx=20, pady=20)

        tree = ttk.Treeview(container, columns=col_names, show="headings",
                            style="Metrics.Treeview", height=num_targets + 1)
        for cn in col_names:
            tree.heading(cn, text=cn)
            tree.column(cn, width=120, anchor="center")
        tree.column("Target", width=180, anchor="w")

        tree.tag_configure("good",   foreground=COLORS["green"])
        tree.tag_configure("medium", foreground=COLORS["orange"])
        tree.tag_configure("bad",    foreground=COLORS["red"])

        for col, r2, rmse, mae, mape in rows:
            tag = "good" if r2 > 0.9 else "medium" if r2 > 0.7 else "bad"
            tree.insert("", "end",
                        values=[col, f"{r2:.4f}", f"{rmse:.4f}", f"{mae:.4f}", f"{mape:.2f}"],
                        tags=(tag,))

        tree.pack(fill="x")

        # Bar chart of R² per target
        if num_targets > 1:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(max(6, num_targets * 1.2), 3), dpi=90)
            fig.patch.set_facecolor(COLORS["bg_card"])
            ax.set_facecolor(COLORS["bg_card"])
            r2_vals = [r[1] for r in rows]
            colors  = [COLORS["green"] if r > 0.9 else COLORS["orange"] if r > 0.7 else COLORS["red"] for r in r2_vals]
            ax.bar(self.out_cols, r2_vals, color=colors, alpha=0.8)
            ax.axhline(0.9, color=COLORS["green"],  linestyle="--", linewidth=1, alpha=0.6, label="R²=0.9")
            ax.axhline(0.7, color=COLORS["orange"], linestyle="--", linewidth=1, alpha=0.6, label="R²=0.7")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("R²", color=COLORS["text"])
            ax.tick_params(colors=COLORS["text"])
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])
            ax.set_title("R² per Target", color="white")
            ax.legend(fontsize=9)
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="x", padx=20, pady=10)
            plt.close(fig)

    # ─── Worst Predictions ────────────────────────────────────────────────────

    def _setup_worst_predictions(self, parent, top_n=20):
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", pady=5)
        ctk.CTkLabel(top, text="Select Target:").pack(side="left", padx=10)
        sel_var = ctk.StringVar(value=self.out_cols[0])
        table_frame = ctk.CTkFrame(parent, fg_color="transparent")
        table_frame.pack(fill="both", expand=True)
        ctk.CTkComboBox(top, values=self.out_cols, variable=sel_var,
                        command=lambda x: self._draw_worst(table_frame, x, self.out_cols.index(x), top_n)).pack(side="left", padx=10)
        self._draw_worst(table_frame, self.out_cols[0], 0, top_n)

    def _draw_worst(self, parent, col_name, idx, top_n):
        for w in parent.winfo_children(): w.destroy()
        yt = self.y_t[:, idx]
        yp = self.y_p[:, idx]
        abs_err = np.abs(yt - yp)
        worst_idx = np.argsort(abs_err)[::-1][:top_n]

        col_names = ["Sample #", "Actual", "Predicted", "Abs Error", "% Error"]

        style = ttk.Style()
        style.configure("Worst.Treeview",
                        background="#1A1A2E", foreground=COLORS["text"],
                        fieldbackground="#1A1A2E", rowheight=24, font=("Helvetica", 10))
        style.configure("Worst.Treeview.Heading",
                        background=COLORS["bg_card"], foreground=COLORS["orange"],
                        font=("Helvetica", 10, "bold"), relief="flat")
        style.map("Worst.Treeview", background=[("selected", COLORS["primary_dark"])])

        container = tk.Frame(parent, bg="#1A1A2E")
        container.pack(fill="both", expand=True, padx=10, pady=6)

        tree = ttk.Treeview(container, columns=col_names, show="headings",
                            style="Worst.Treeview", height=min(top_n, 15))
        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)

        for cn in col_names:
            tree.heading(cn, text=cn)
            tree.column(cn, width=110, anchor="center")
        tree.column("Sample #", width=80)

        tree.tag_configure("high_err", foreground=COLORS["red"])
        median_err = float(np.median(abs_err))

        for rank, si in enumerate(worst_idx):
            pct = abs(yt[si] - yp[si]) / (abs(yt[si]) + 1e-10) * 100
            tag = "high_err" if abs_err[si] > 2 * median_err else ""
            tree.insert("", "end",
                        values=[si, f"{yt[si]:.4f}", f"{yp[si]:.4f}",
                                f"{abs_err[si]:.4f}", f"{pct:.1f}%"],
                        tags=(tag,))

        # Scatter plot highlighting worst samples
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=90)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax.set_facecolor(COLORS["bg_card"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values(): spine.set_color(COLORS["border"])

        all_idx = np.arange(len(yt))
        good_mask = np.ones(len(yt), dtype=bool)
        good_mask[worst_idx] = False
        ax.scatter(yt[good_mask], yp[good_mask], color=COLORS["cyan"], s=8, alpha=0.4, label="OK")
        ax.scatter(yt[worst_idx], yp[worst_idx], color=COLORS["red"], s=20, alpha=0.9, label=f"Worst {top_n}", zorder=5)
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], color=COLORS["green"], linestyle="--", linewidth=1)
        ax.set_title(f"Pred vs Actual — {col_name} (worst highlighted)", color="white", fontsize=10)
        ax.legend(fontsize=9)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", padx=10, pady=6)
        plt.close(fig)

    # ─── SHAP Values ──────────────────────────────────────────────────────────

    def _setup_shap(self, parent, model):
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", pady=5)

        ctk.CTkLabel(top, text="Select Target:").pack(side="left", padx=10)
        sel_var = ctk.StringVar(value=self.out_cols[0])
        plot_frame = ctk.CTkFrame(parent, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True)

        def run_shap(col_name):
            self._draw_shap(plot_frame, model, col_name, self.out_cols.index(col_name))

        ctk.CTkComboBox(top, values=self.out_cols, variable=sel_var,
                        command=run_shap).pack(side="left", padx=10)
        ctk.CTkButton(top, text="Compute SHAP", height=30,
                      command=lambda: run_shap(sel_var.get())).pack(side="left", padx=10)
        ctk.CTkLabel(top, text="(may take a few seconds)",
                     text_color=COLORS["text_dim"], font=("Helvetica", 11)).pack(side="left")

        ctk.CTkLabel(plot_frame,
                     text="Click 'Compute SHAP' to generate feature importance.",
                     text_color=COLORS["text_dim"], font=FONTS["header"]).pack(pady=40)

    def _draw_shap(self, parent, model, col_name, output_idx):
        for w in parent.winfo_children(): w.destroy()
        ctk.CTkLabel(parent, text="Computing SHAP values...",
                     text_color=COLORS["cyan"], font=FONTS["header"]).pack(pady=20)
        parent.update()

        try:
            import shap
        except ImportError:
            for w in parent.winfo_children(): w.destroy()
            ctk.CTkLabel(parent,
                         text="SHAP library not installed.\nRun:  pip install shap",
                         text_color=COLORS["orange"], font=FONTS["header"]).pack(pady=40)
            return

        try:
            X_test    = get_state("X_test")
            input_cols = get_state("input_columns")
            pca_X     = get_state("pca_X")

            # Use a background sample (max 100 rows) for the explainer
            bg_size = min(100, len(X_test))
            bg = X_test[:bg_size]

            # Single-output wrapper so SHAP gets a 1-D prediction
            def predict_fn(x):
                raw = model.predict(x, verbose=0)
                return raw[:, output_idx] if raw.ndim > 1 else raw

            explainer = shap.KernelExplainer(predict_fn, bg)
            sample_size = min(50, len(X_test))
            shap_vals = explainer.shap_values(X_test[:sample_size], nsamples=80)

            # If PCA was applied we have latent features, not original column names
            if pca_X is not None:
                feat_names = [f"PC{i+1}" for i in range(X_test.shape[1])]
            else:
                feat_names = input_cols if input_cols else [f"F{i}" for i in range(X_test.shape[1])]

            shap_arr = np.array(shap_vals)
            mean_abs = np.abs(shap_arr).mean(axis=0)
            order    = np.argsort(mean_abs)[::-1]

            for w in parent.winfo_children(): w.destroy()

            plt.style.use("dark_background")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, len(feat_names) * 0.4 + 1)), dpi=90)
            fig.patch.set_facecolor(COLORS["bg_card"])

            for ax in [ax1, ax2]:
                ax.set_facecolor(COLORS["bg_card"])
                ax.tick_params(colors=COLORS["text"], labelsize=9)
                for spine in ax.spines.values(): spine.set_color(COLORS["border"])

            # Bar chart (mean |SHAP|)
            colors = [COLORS["cyan"] if i == order[0] else COLORS["magenta"]
                      if i in order[:3] else COLORS["text_dim"] for i in range(len(feat_names))]
            ax1.barh([feat_names[i] for i in order[::-1]],
                     mean_abs[order[::-1]], color=colors[::-1], alpha=0.85)
            ax1.set_xlabel("Mean |SHAP value|", color=COLORS["text"])
            ax1.set_title(f"Feature Importance — {col_name}", color="white", fontsize=11)

            # Beeswarm-style: SHAP value vs feature value (top 8)
            top_feats = order[:8]
            for rank, fi in enumerate(top_feats[::-1]):
                sv = shap_arr[:, fi]
                fv_norm = (X_test[:sample_size, fi] - X_test[:sample_size, fi].min()) / \
                          (X_test[:sample_size, fi].max() - X_test[:sample_size, fi].min() + 1e-8)
                y_jitter = rank + np.random.uniform(-0.25, 0.25, len(sv))
                sc = ax2.scatter(sv, y_jitter, c=fv_norm, cmap="coolwarm", s=12, alpha=0.7)

            ax2.set_yticks(range(len(top_feats)))
            ax2.set_yticklabels([feat_names[i] for i in top_feats[::-1]], fontsize=9)
            ax2.axvline(0, color=COLORS["text_dim"], linewidth=1, linestyle="--")
            ax2.set_xlabel("SHAP value", color=COLORS["text"])
            ax2.set_title("Top-8 SHAP Beeswarm", color="white", fontsize=11)
            cbar = fig.colorbar(sc, ax=ax2, fraction=0.03, pad=0.02)
            cbar.set_label("Feature value (norm)", color=COLORS["text"], fontsize=8)
            cbar.ax.tick_params(colors=COLORS["text"], labelsize=7)

            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
            plt.close(fig)

        except Exception as e:
            for w in parent.winfo_children(): w.destroy()
            ctk.CTkLabel(parent, text=f"SHAP error: {e}",
                         text_color=COLORS["red"], font=("Helvetica", 12)).pack(pady=20)

    def _setup_export(self, parent, model):
        # Create a DF for all predictions
        data = {}
        for i, col in enumerate(self.out_cols):
            data[f"Actual_{col}"] = self.y_t[:, i]
            data[f"Predicted_{col}"] = self.y_p[:, i]
            data[f"Residual_{col}"] = self.y_t[:, i] - self.y_p[:, i]
        self.df_res = pd.DataFrame(data)
        
        ctk.CTkLabel(parent, text="Save Data & Model for Inference", font=FONTS["header"]).pack(pady=(20, 10))
        
        b_frame = ctk.CTkFrame(parent, fg_color="transparent")
        b_frame.pack(pady=10)
        ctk.CTkButton(b_frame, text="📄 Download Test Results (CSV)", command=self._dl_csv).pack(side="left", padx=10)
        ctk.CTkButton(b_frame, text="📊 Download Test Results (Excel)", command=self._dl_excel).pack(side="left", padx=10)
        
        w_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg"])
        w_frame.pack(pady=30, padx=50, fill="x")
        ctk.CTkLabel(w_frame, text="Model Wrapper Bundle (Deployable)", font=FONTS["title"], text_color=COLORS["green"]).pack(pady=(20, 5))
        ctk.CTkLabel(w_frame, text="This wraps the Preprocessors (Scalers, PCA) alongside the Keras neural network.\nLoad this archive directly in the 'Inference' tab to make live predictions.", text_color=COLORS["text"]).pack(pady=(0, 20))
        
        ctk.CTkButton(w_frame, text="📦 EXPORT MODEL WRAPPER (.zip)", height=50, font=("Helvetica", 14, "bold"), fg_color=COLORS["primary"], hover_color=COLORS["primary_dark"], text_color="#000", command=lambda: self._dl_model(model)).pack(pady=(0, 20))

    def _dl_csv(self):
        fn = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if fn:
            self.df_res.to_csv(fn, index=False)
            messagebox.showinfo("Success", f"Saved to {fn}")

    def _dl_excel(self):
        fn = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
        if fn:
            self.df_res.to_excel(fn, index=False)
            messagebox.showinfo("Success", f"Saved to {fn}")

    def _dl_model(self, model):
        fn = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("Model Wrapper ZIP", "*.zip")], initialfile="model_wrapper.zip")
        if fn:
            try:
                # Save keras model to temp
                model.save("temp_model.keras")
                
                # Get max and mins for slider Base points
                df_orig = get_state("df")
                input_cols = get_state("input_columns")
                
                metadata = {
                    "scaler_X": get_state("scaler_X"),
                    "scaler_y": get_state("scaler_y"),
                    "pca_X": get_state("pca_X"),
                    "pca_y": get_state("pca_y"),
                    "input_columns": input_cols,
                    "output_columns": get_state("output_column"), 
                    "train_min": df_orig[input_cols].min(numeric_only=True).to_dict(),
                    "train_max": df_orig[input_cols].max(numeric_only=True).to_dict(),
                    "train_mean": df_orig[input_cols].mean(numeric_only=True).to_dict()
                }
                
                with open("temp_meta.pkl", "wb") as f:
                    pickle.dump(metadata, f)
                    
                with zipfile.ZipFile(fn, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write("temp_model.keras", arcname="model.keras")
                    zf.write("temp_meta.pkl", arcname="metadata.pkl")
                    
                if os.path.exists("temp_model.keras"): os.remove("temp_model.keras")
                if os.path.exists("temp_meta.pkl"): os.remove("temp_meta.pkl")
                    
                messagebox.showinfo("Success", f"Deployable wrapper bundle saved to:\n{fn}\n\nYou can load this directly in the Inference tab!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save wrapper:\n{e}")
