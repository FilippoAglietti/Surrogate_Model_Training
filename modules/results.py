import customtkinter as ctk
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
import json
import os
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
import zipfile
import pickle
import tempfile

from utils.theme import COLORS, FONTS
from utils.state import get_state
from utils.plot_utils import add_save_button


class ResultsFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.header = ctk.CTkLabel(self, text="RESULTS & ANALYSIS 📊",
                                   font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(30, 20), sticky="w", padx=30)
        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.built_ui = False

    def on_show(self):
        if not get_state("trained"):
            self._show_blocked("Train a model first.\n← Go to 'Model Builder'")
            return
        if not self.built_ui or get_state("results_stale"):
            self._build_ui()
            self.built_ui = True
            from utils.state import set_state
            set_state("results_stale", False)

    def _show_blocked(self, message):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.built_ui = False
        lbl = ctk.CTkLabel(self.content_frame, text=f"[ BLOCKED ]\n{message}",
                           font=FONTS["header"], text_color=COLORS["red"])
        lbl.grid(row=0, column=0, pady=50, padx=20)

    def _build_ui(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        surrogate = get_state("surrogate_model")
        if surrogate is None:
            # Fallback: wrap legacy Keras model
            model = get_state("model")
            if model is None:
                self._show_blocked("No trained model found.")
                return
            from models.nn_model import NeuralNetworkSurrogate
            surrogate = NeuralNetworkSurrogate(model)

        X_test = get_state("X_test")
        y_test = get_state("y_test")
        scaler_y = get_state("scaler_y")
        pca_y = get_state("pca_y")
        output_cols = get_state("output_column")

        y_pred = surrogate.predict(X_test)
        y_true = y_test

        if pca_y is not None:
            y_pred = pca_y.inverse_transform(y_pred)
            y_true = pca_y.inverse_transform(y_true)
        if scaler_y is not None:
            y_pred_orig = scaler_y.inverse_transform(y_pred)
            y_true_orig = scaler_y.inverse_transform(y_true)
        else:
            y_pred_orig = y_pred
            y_true_orig = y_true

        if len(y_pred_orig.shape) == 1:
            y_pred_orig = y_pred_orig.reshape(-1, 1)
            y_true_orig = y_true_orig.reshape(-1, 1)

        self.y_p = y_pred_orig
        self.y_t = y_true_orig
        self.out_cols = output_cols
        self.surrogate = surrogate

        mse = mean_squared_error(self.y_t, self.y_p)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_t, self.y_p)
        r2_avg = r2_score(self.y_t, self.y_p)
        r2_raw = r2_score(self.y_t, self.y_p, multioutput='raw_values')
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((self.y_t - self.y_p) / (self.y_t + 1e-10))) * 100

        # Algo badge
        ctk.CTkLabel(self.content_frame,
                     text=f"Algorithm: {surrogate.algo_name}",
                     font=("Helvetica", 13, "bold"),
                     text_color=COLORS["magenta"]).grid(row=0, column=0, pady=(5, 0), sticky="w", padx=25)

        # Metrics bar
        metrics_f = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        metrics_f.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        for i in range(5): metrics_f.grid_columnconfigure(i, weight=1)
        for i, (k, v) in enumerate([("Avg MSE", f"{mse:.4f}"), ("Avg RMSE", f"{rmse:.4f}"),
                                     ("Avg MAE", f"{mae:.4f}"), ("Avg R²", f"{r2_avg:.4f}"),
                                     ("Avg MAPE %", f"{mape:.2f}")]):
            ctk.CTkLabel(metrics_f, text=k, font=FONTS["header"], text_color=COLORS["cyan"]).grid(
                row=0, column=i, pady=(10, 0))
            ctk.CTkLabel(metrics_f, text=v, font=FONTS["title"]).grid(row=1, column=i, pady=(0, 10))

        if len(output_cols) > 1:
            r2_txt = " | ".join([f"{c}: {r:.4f}" for c, r in zip(output_cols, r2_raw)])
            ctk.CTkLabel(self.content_frame, text=f"Target R²: {r2_txt}",
                         font=("Helvetica", 14), text_color=COLORS["green"]).grid(
                row=2, column=0, pady=(0, 10))

        self.tabview = ctk.CTkTabview(self.content_frame, fg_color=COLORS["bg_card"], height=600)
        self.tabview.grid(row=3, column=0, sticky="nsew", padx=20, pady=10)
        for tab_name in ["Pred vs Actual", "Test Index Series", "Residuals", "Q-Q Plot",
                         "Per-Target Metrics", "Worst Predictions", "SHAP Values", "Export & Wrapper"]:
            self.tabview.add(tab_name)

        self._setup_pred_actual(self.tabview.tab("Pred vs Actual"))
        self._setup_series(self.tabview.tab("Test Index Series"))
        self._setup_residuals(self.tabview.tab("Residuals"))
        self._setup_qq(self.tabview.tab("Q-Q Plot"))
        self._setup_per_target_metrics(self.tabview.tab("Per-Target Metrics"))
        self._setup_worst_predictions(self.tabview.tab("Worst Predictions"))
        self._setup_shap(self.tabview.tab("SHAP Values"))
        self._setup_export(self.tabview.tab("Export & Wrapper"))

    # ─── Pred vs Actual ───────────────────────────────────────────────────────

    def _setup_pred_actual(self, parent):
        plot_frame = ctk.CTkFrame(parent, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True)
        if len(self.out_cols) <= 4:
            self._draw_pred_grid(plot_frame, self.out_cols, self.y_t, self.y_p)
        else:
            top = ctk.CTkFrame(parent, fg_color="transparent")
            top.pack(fill="x", pady=5)
            ctk.CTkLabel(top, text="Select Target:").pack(side="left", padx=10)
            sel_var = ctk.StringVar(value=self.out_cols[0])
            ctk.CTkComboBox(top, values=self.out_cols, variable=sel_var,
                            command=lambda x: self._draw_single_pred(plot_frame, x, self.out_cols.index(x))
                            ).pack(side="left", padx=10)
            self._draw_single_pred(plot_frame, self.out_cols[0], 0)

    def _draw_pred_grid(self, parent, cols, y_t, y_p):
        for w in parent.winfo_children(): w.destroy()
        num = len(cols)
        rows = 1 if num <= 2 else 2
        cols_grid = 1 if num == 1 else 2
        fig, axes = plt.subplots(rows, cols_grid, figsize=(9, 6), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        axes_flat = [axes] if num == 1 else axes.flatten()
        for i in range(len(axes_flat)):
            ax = axes_flat[i]
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text"])
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])
            if i < num:
                ax.scatter(y_t[:, i], y_p[:, i], alpha=0.6, color=COLORS["cyan"], s=10)
                mn, mx = min(y_t[:, i].min(), y_p[:, i].min()), max(y_t[:, i].max(), y_p[:, i].max())
                ax.plot([mn, mx], [mn, mx], color=COLORS["green"], linestyle="--")
                ax.set_title(cols[i], color="white", fontsize=10)
            else:
                ax.axis('off')
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
        add_save_button(parent, canvas, "pred_vs_actual.png")

    def _draw_single_pred(self, parent, col_name, idx):
        for w in parent.winfo_children(): w.destroy()
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax.set_facecolor(COLORS["bg_card"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values(): spine.set_color(COLORS["border"])
        y_ti, y_pi = self.y_t[:, idx], self.y_p[:, idx]
        ax.scatter(y_ti, y_pi, alpha=0.6, color=COLORS["cyan"], s=15)
        mn, mx = min(y_ti.min(), y_pi.min()), max(y_ti.max(), y_pi.max())
        ax.plot([mn, mx], [mn, mx], color=COLORS["green"], linestyle="--")
        ax.set_title(f"Predicted vs Actual ({col_name})", color="white")
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
        add_save_button(parent, canvas, "pred_vs_actual_single.png")

    # ─── Test Index Series ────────────────────────────────────────────────────

    def _setup_series(self, parent):
        plot_frame = ctk.CTkFrame(parent, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True)
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", pady=5, before=plot_frame)
        ctk.CTkLabel(top, text="Select Target:").pack(side="left", padx=10)
        sel_var = ctk.StringVar(value=self.out_cols[0])
        ctk.CTkComboBox(top, values=self.out_cols, variable=sel_var,
                        command=lambda x: self._draw_series(plot_frame, x, self.out_cols.index(x))
                        ).pack(side="left", padx=10)
        self._draw_series(plot_frame, self.out_cols[0], 0)

    def _draw_series(self, parent, col_name, idx):
        for w in parent.winfo_children(): w.destroy()
        fig, ax = plt.subplots(figsize=(9, 4), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax.set_facecolor(COLORS["bg_card"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values(): spine.set_color(COLORS["border"])
        y_ti, y_pi = self.y_t[:, idx], self.y_p[:, idx]
        x_ax = np.arange(len(y_ti))
        sort_idx = np.argsort(y_ti)
        ax.plot(x_ax, y_ti[sort_idx], color=COLORS["orange"], label="Actual (Sorted)", linewidth=2)
        ax.scatter(x_ax, y_pi[sort_idx], color=COLORS["cyan"], label="Predicted", s=10, alpha=0.7)
        ax.set_title(f"{col_name} - Pred vs Real over Test Samples", color="white")
        ax.legend()
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
        add_save_button(parent, canvas, "test_series.png")

    # ─── Residuals ────────────────────────────────────────────────────────────

    def _setup_residuals(self, parent):
        plot_frame = ctk.CTkFrame(parent, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True)
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", pady=5, before=plot_frame)
        ctk.CTkLabel(top, text="Select Target:").pack(side="left", padx=10)
        sel_var = ctk.StringVar(value=self.out_cols[0])
        ctk.CTkComboBox(top, values=self.out_cols, variable=sel_var,
                        command=lambda x: self._draw_single_res(plot_frame, x, self.out_cols.index(x))
                        ).pack(side="left", padx=10)
        self._draw_single_res(plot_frame, self.out_cols[0], 0)

    def _draw_single_res(self, parent, col_name, idx):
        for w in parent.winfo_children(): w.destroy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        y_ti, y_pi = self.y_t[:, idx], self.y_p[:, idx]
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
        add_save_button(parent, canvas, "residuals.png")

    # ─── Q-Q Plot ─────────────────────────────────────────────────────────────

    def _setup_qq(self, parent):
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", pady=5)
        ctk.CTkLabel(top, text="Select Target:").pack(side="left", padx=10)
        sel_var = ctk.StringVar(value=self.out_cols[0])
        plot_frame = ctk.CTkFrame(parent, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True)
        ctk.CTkComboBox(top, values=self.out_cols, variable=sel_var,
                        command=lambda x: self._draw_qq(plot_frame, x, self.out_cols.index(x))
                        ).pack(side="left", padx=10)
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
        (osm, osr), (slope, intercept, r) = stats.probplot(res, dist="norm")
        ax1.scatter(osm, osr, color=COLORS["cyan"], s=10, alpha=0.7)
        line_x = np.array([osm.min(), osm.max()])
        ax1.plot(line_x, slope * line_x + intercept, color=COLORS["orange"], linewidth=2)
        ax1.set_title(f"Q-Q Plot — {col_name}  (R={r:.3f})", color="white")
        ax2.hist(res, bins=35, color=COLORS["magenta"], alpha=0.6, density=True)
        mu, sigma = res.mean(), res.std()
        xs = np.linspace(res.min(), res.max(), 200)
        ax2.plot(xs, stats.norm.pdf(xs, mu, sigma), color=COLORS["cyan"], linewidth=2)
        ax2.axvline(0, color=COLORS["green"], linestyle="--", linewidth=1)
        ax2.set_title("Residual Distribution vs Normal", color="white")
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
        add_save_button(parent, canvas, "qq_plot.png")
        plt.close(fig)

    # ─── Per-Target Metrics ───────────────────────────────────────────────────

    def _setup_per_target_metrics(self, parent):
        num_targets = len(self.out_cols)
        rows = []
        for i, col in enumerate(self.out_cols):
            yt, yp = self.y_t[:, i], self.y_p[:, i]
            rmse = np.sqrt(mean_squared_error(yt, yp))
            mae = mean_absolute_error(yt, yp)
            r2 = r2_score(yt, yp)
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((yt - yp) / (yt + 1e-10))) * 100
            rows.append((col, r2, rmse, mae, mape))

        col_names = ["Target", "R²", "RMSE", "MAE", "MAPE %"]
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Metrics.Treeview", background="#1A1A2E", foreground=COLORS["text"],
                        fieldbackground="#1A1A2E", rowheight=26, font=("Helvetica", 11))
        style.configure("Metrics.Treeview.Heading", background=COLORS["bg_card"],
                        foreground=COLORS["cyan"], font=("Helvetica", 11, "bold"), relief="flat")
        style.map("Metrics.Treeview", background=[("selected", COLORS["primary_dark"])])

        container = tk.Frame(parent, bg="#1A1A2E")
        container.pack(fill="x", padx=20, pady=20)
        tree = ttk.Treeview(container, columns=col_names, show="headings",
                            style="Metrics.Treeview", height=num_targets + 1)
        for cn in col_names:
            tree.heading(cn, text=cn)
            tree.column(cn, width=120, anchor="center")
        tree.column("Target", width=180, anchor="w")
        tree.tag_configure("good", foreground=COLORS["green"])
        tree.tag_configure("medium", foreground=COLORS["orange"])
        tree.tag_configure("bad", foreground=COLORS["red"])
        for col, r2, rmse, mae, mape in rows:
            tag = "good" if r2 > 0.9 else "medium" if r2 > 0.7 else "bad"
            tree.insert("", "end", values=[col, f"{r2:.4f}", f"{rmse:.4f}", f"{mae:.4f}", f"{mape:.2f}"],
                        tags=(tag,))
        tree.pack(fill="x")

        if num_targets > 1:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(max(6, num_targets * 1.2), 3), dpi=90)
            fig.patch.set_facecolor(COLORS["bg_card"])
            ax.set_facecolor(COLORS["bg_card"])
            r2_vals = [r[1] for r in rows]
            colors = [COLORS["green"] if r > 0.9 else COLORS["orange"] if r > 0.7 else COLORS["red"]
                      for r in r2_vals]
            ax.bar(self.out_cols, r2_vals, color=colors, alpha=0.8)
            ax.axhline(0.9, color=COLORS["green"], linestyle="--", linewidth=1, alpha=0.6)
            ax.axhline(0.7, color=COLORS["orange"], linestyle="--", linewidth=1, alpha=0.6)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("R²", color=COLORS["text"])
            ax.tick_params(colors=COLORS["text"])
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])
            ax.set_title("R² per Target", color="white")
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="x", padx=20, pady=10)
            add_save_button(parent, canvas, "r2_per_target.png")
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
                        command=lambda x: self._draw_worst(table_frame, x, self.out_cols.index(x), top_n)
                        ).pack(side="left", padx=10)
        self._draw_worst(table_frame, self.out_cols[0], 0, top_n)

    def _draw_worst(self, parent, col_name, idx, top_n):
        for w in parent.winfo_children(): w.destroy()
        yt, yp = self.y_t[:, idx], self.y_p[:, idx]
        abs_err = np.abs(yt - yp)
        worst_idx = np.argsort(abs_err)[::-1][:top_n]
        col_names = ["Sample #", "Actual", "Predicted", "Abs Error", "% Error"]
        style = ttk.Style()
        style.configure("Worst.Treeview", background="#1A1A2E", foreground=COLORS["text"],
                        fieldbackground="#1A1A2E", rowheight=24, font=("Helvetica", 10))
        style.configure("Worst.Treeview.Heading", background=COLORS["bg_card"],
                        foreground=COLORS["orange"], font=("Helvetica", 10, "bold"), relief="flat")
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
        for si in worst_idx:
            pct = abs(yt[si] - yp[si]) / (abs(yt[si]) + 1e-10) * 100
            tag = "high_err" if abs_err[si] > 2 * median_err else ""
            tree.insert("", "end",
                        values=[si, f"{yt[si]:.4f}", f"{yp[si]:.4f}", f"{abs_err[si]:.4f}", f"{pct:.1f}%"],
                        tags=(tag,))
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=90)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax.set_facecolor(COLORS["bg_card"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values(): spine.set_color(COLORS["border"])
        good_mask = np.ones(len(yt), dtype=bool)
        good_mask[worst_idx] = False
        ax.scatter(yt[good_mask], yp[good_mask], color=COLORS["cyan"], s=8, alpha=0.4, label="OK")
        ax.scatter(yt[worst_idx], yp[worst_idx], color=COLORS["red"], s=20, alpha=0.9,
                   label=f"Worst {top_n}", zorder=5)
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], color=COLORS["green"], linestyle="--", linewidth=1)
        ax.set_title(f"Pred vs Actual — {col_name} (worst highlighted)", color="white", fontsize=10)
        ax.legend(fontsize=9)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", padx=10, pady=6)
        add_save_button(parent, canvas, "worst_predictions.png")
        plt.close(fig)

    # ─── SHAP Values ──────────────────────────────────────────────────────────

    def _setup_shap(self, parent):
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", pady=5)
        ctk.CTkLabel(top, text="Select Target:").pack(side="left", padx=10)
        sel_var = ctk.StringVar(value=self.out_cols[0])
        plot_frame = ctk.CTkFrame(parent, fg_color="transparent")
        plot_frame.pack(fill="both", expand=True)

        def run_shap(col_name):
            self._draw_shap(plot_frame, col_name, self.out_cols.index(col_name))

        ctk.CTkComboBox(top, values=self.out_cols, variable=sel_var,
                        command=run_shap).pack(side="left", padx=10)
        ctk.CTkButton(top, text="Compute SHAP", height=30,
                      command=lambda: run_shap(sel_var.get())).pack(side="left", padx=10)
        ctk.CTkLabel(top, text="(may take a few seconds)",
                     text_color=COLORS["text_dim"], font=("Helvetica", 11)).pack(side="left")
        ctk.CTkLabel(plot_frame, text="Click 'Compute SHAP' to generate feature importance.",
                     text_color=COLORS["text_dim"], font=FONTS["header"]).pack(pady=40)

    def _draw_shap(self, parent, col_name, output_idx):
        for w in parent.winfo_children(): w.destroy()
        ctk.CTkLabel(parent, text="Computing SHAP values...",
                     text_color=COLORS["cyan"], font=FONTS["header"]).pack(pady=20)
        parent.update()

        try:
            import shap
        except ImportError:
            for w in parent.winfo_children(): w.destroy()
            ctk.CTkLabel(parent, text="SHAP library not installed.\nRun:  pip install shap",
                         text_color=COLORS["orange"], font=FONTS["header"]).pack(pady=40)
            return

        try:
            X_test = get_state("X_test")
            input_cols = get_state("input_columns")
            pca_X = get_state("pca_X")
            surrogate = self.surrogate
            algo = surrogate.algo_name
            sklearn_est = surrogate.get_sklearn_estimator()

            bg_size = min(100, len(X_test))
            sample_size = min(50, len(X_test))
            bg = X_test[:bg_size]

            # ── SHAP dispatch ──
            if algo == "XGBoost" and sklearn_est is not None:
                if hasattr(sklearn_est, 'estimators_'):
                    # MultiOutputRegressor — get estimator for this output
                    est = sklearn_est.estimators_[output_idx]
                else:
                    est = sklearn_est
                explainer = shap.TreeExplainer(est)
                sv = explainer.shap_values(X_test[:sample_size])
                shap_arr = np.array(sv)
                if shap_arr.ndim == 3:
                    shap_arr = shap_arr[:, :, 0]

            elif algo == "Random Forest" and sklearn_est is not None:
                explainer = shap.TreeExplainer(sklearn_est)
                sv = explainer.shap_values(X_test[:sample_size])
                if isinstance(sv, list):
                    shap_arr = np.array(sv[output_idx] if output_idx < len(sv) else sv[0])
                elif np.array(sv).ndim == 3:
                    shap_arr = np.array(sv)[:, :, output_idx]
                else:
                    shap_arr = np.array(sv)

            else:
                # KernelExplainer for NN / GPR
                def predict_fn(x):
                    raw = surrogate.predict(x)
                    return raw[:, output_idx] if raw.ndim > 1 else raw
                explainer = shap.KernelExplainer(predict_fn, bg)
                shap_arr = np.array(explainer.shap_values(X_test[:sample_size], nsamples=80))

            if pca_X is not None:
                feat_names = [f"PC{i+1}" for i in range(X_test.shape[1])]
            else:
                feat_names = input_cols if input_cols else [f"F{i}" for i in range(X_test.shape[1])]

            mean_abs = np.abs(shap_arr).mean(axis=0)
            order = np.argsort(mean_abs)[::-1]

            for w in parent.winfo_children(): w.destroy()

            plt.style.use("dark_background")
            fig, (ax1, ax2) = plt.subplots(1, 2,
                                           figsize=(12, max(4, len(feat_names) * 0.4 + 1)), dpi=90)
            fig.patch.set_facecolor(COLORS["bg_card"])
            for ax in [ax1, ax2]:
                ax.set_facecolor(COLORS["bg_card"])
                ax.tick_params(colors=COLORS["text"], labelsize=9)
                for spine in ax.spines.values(): spine.set_color(COLORS["border"])

            colors = [COLORS["cyan"] if i == order[0] else COLORS["magenta"]
                      if i in order[:3] else COLORS["text_dim"] for i in range(len(feat_names))]
            ax1.barh([feat_names[i] for i in order[::-1]], mean_abs[order[::-1]],
                     color=colors[::-1], alpha=0.85)
            ax1.set_xlabel("Mean |SHAP value|", color=COLORS["text"])
            ax1.set_title(f"Feature Importance — {col_name}  [{algo}]", color="white", fontsize=11)

            top_feats = order[:8]
            for rank, fi in enumerate(top_feats[::-1]):
                sv_col = shap_arr[:, fi]
                fv_norm = (X_test[:sample_size, fi] - X_test[:sample_size, fi].min()) / \
                          (X_test[:sample_size, fi].max() - X_test[:sample_size, fi].min() + 1e-8)
                y_jitter = rank + np.random.uniform(-0.25, 0.25, len(sv_col))
                ax2.scatter(sv_col, y_jitter, c=fv_norm, cmap="coolwarm", s=12, alpha=0.7)
            ax2.set_yticks(range(len(top_feats)))
            ax2.set_yticklabels([feat_names[i] for i in top_feats[::-1]], fontsize=9)
            ax2.axvline(0, color=COLORS["text_dim"], linewidth=1, linestyle="--")
            ax2.set_xlabel("SHAP value", color=COLORS["text"])
            ax2.set_title("Top-8 SHAP Beeswarm", color="white", fontsize=11)

            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
            add_save_button(parent, canvas, "shap_values.png")
            plt.close(fig)

        except Exception as e:
            for w in parent.winfo_children(): w.destroy()
            ctk.CTkLabel(parent, text=f"SHAP error: {e}",
                         text_color=COLORS["red"], font=("Helvetica", 12)).pack(pady=20)

    # ─── Export & Wrapper ─────────────────────────────────────────────────────

    def _setup_export(self, parent):
        data = {}
        for i, col in enumerate(self.out_cols):
            data[f"Actual_{col}"] = self.y_t[:, i]
            data[f"Predicted_{col}"] = self.y_p[:, i]
            data[f"Residual_{col}"] = self.y_t[:, i] - self.y_p[:, i]
        self.df_res = pd.DataFrame(data)

        ctk.CTkLabel(parent, text="Save Data & Model for Inference",
                     font=FONTS["header"]).pack(pady=(20, 10))
        b_frame = ctk.CTkFrame(parent, fg_color="transparent")
        b_frame.pack(pady=10)
        ctk.CTkButton(b_frame, text="📄 Download Test Results (CSV)", command=self._dl_csv).pack(side="left", padx=10)
        ctk.CTkButton(b_frame, text="📊 Download Test Results (Excel)", command=self._dl_excel).pack(side="left", padx=10)

        w_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg"])
        w_frame.pack(pady=30, padx=50, fill="x")
        ctk.CTkLabel(w_frame, text="Model Wrapper Bundle (Deployable)",
                     font=FONTS["title"], text_color=COLORS["green"]).pack(pady=(20, 5))
        algo = self.surrogate.algo_name
        ctk.CTkLabel(w_frame,
                     text=f"Algorithm: {algo}\n"
                          "Wraps Preprocessors (Scalers, PCA) + trained model.\n"
                          "Load this archive in the 'Inference' tab to make live predictions.",
                     text_color=COLORS["text"]).pack(pady=(0, 20))
        ctk.CTkButton(w_frame, text="📦 EXPORT MODEL WRAPPER (.zip)", height=50,
                      font=("Helvetica", 14, "bold"), fg_color=COLORS["primary"],
                      hover_color=COLORS["primary_dark"], text_color="#000",
                      command=self._dl_model).pack(pady=(0, 20))

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

    def _dl_model(self):
        surrogate = get_state("surrogate_model")
        if surrogate is None:
            messagebox.showerror("Error", "No trained model found.")
            return

        fn = filedialog.asksaveasfilename(
            defaultextension=".zip", filetypes=[("Model Wrapper ZIP", "*.zip")],
            initialfile="model_wrapper.zip")
        if not fn:
            return

        try:
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
                "train_mean": df_orig[input_cols].mean(numeric_only=True).to_dict(),
            }
            manifest = {"algorithm": surrogate.algo_name, "version": 2}

            with tempfile.TemporaryDirectory() as td:
                model_dir = os.path.join(td, "model")
                surrogate.save(model_dir)

                with open(os.path.join(td, "metadata.pkl"), "wb") as f:
                    pickle.dump(metadata, f)
                with open(os.path.join(td, "manifest.json"), "w") as f:
                    json.dump(manifest, f)

                with zipfile.ZipFile(fn, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(os.path.join(td, "manifest.json"), arcname="manifest.json")
                    zf.write(os.path.join(td, "metadata.pkl"), arcname="metadata.pkl")
                    for root, dirs, files in os.walk(model_dir):
                        for file in files:
                            filepath = os.path.join(root, file)
                            arcname = os.path.relpath(filepath, td)
                            zf.write(filepath, arcname=arcname)

            messagebox.showinfo("Success",
                                f"Deployable wrapper saved to:\n{fn}\n\n"
                                f"Algorithm: {surrogate.algo_name}\n"
                                "Load this in the Inference tab!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save wrapper:\n{e}")
