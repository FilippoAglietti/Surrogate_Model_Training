import customtkinter as ctk
import numpy as np
import pandas as pd
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.theme import COLORS, FONTS
from utils.state import get_state


class ResultsFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        self.header = ctk.CTkLabel(self, text="RESULTS & ANALYSIS 📊", font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(30, 20), sticky="w", padx=30)
        
        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        
        self.built_ui = False

    def on_show(self):
        if not get_state("trained"):
            self._show_blocked("Train a model first.\n← Go to 'Training Dashboard'")
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
        input_cols = get_state("input_columns")
        output_col = get_state("output_column")
        
        y_pred = model.predict(X_test, verbose=0)
        y_true = y_test

        if scaler_y is not None:
            y_pred_orig = scaler_y.inverse_transform(y_pred)
            y_true_orig = scaler_y.inverse_transform(y_true)
        else:
            y_pred_orig = y_pred
            y_true_orig = y_true

        y_p = y_pred_orig.flatten()
        y_t = y_true_orig.flatten()
        
        # Metrics
        mse = mean_squared_error(y_t, y_p)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        mape = np.mean(np.abs((y_t - y_p) / (y_t + 1e-10))) * 100
        
        metrics_f = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        metrics_f.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        for i in range(5): metrics_f.grid_columnconfigure(i, weight=1)
        
        metrics = [("MSE", f"{mse:.6f}"), ("RMSE", f"{rmse:.6f}"), ("MAE", f"{mae:.6f}"), ("R²", f"{r2:.4f}"), ("MAPE %", f"{mape:.2f}")]
        for i, (k, v) in enumerate(metrics):
            ctk.CTkLabel(metrics_f, text=k, font=FONTS["header"], text_color=COLORS["cyan"]).grid(row=0, column=i, pady=(10,0))
            ctk.CTkLabel(metrics_f, text=v, font=FONTS["title"]).grid(row=1, column=i, pady=(0,10))

        # Plots
        self.tabview = ctk.CTkTabview(self.content_frame, fg_color=COLORS["bg_card"])
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        self.tabview.add("Pred vs Actual")
        self.tabview.add("Residuals")
        self.tabview.add("Feature Importance")
        self.tabview.add("Export")
        
        # Draw Pred vs Actual
        self._plot_pred_actual(self.tabview.tab("Pred vs Actual"), y_t, y_p, output_col)
        
        # Draw Residuals
        self._plot_residuals(self.tabview.tab("Residuals"), y_t, y_p)
        
        # Draw Feature Importance
        self._plot_fi(self.tabview.tab("Feature Importance"), model, X_test, y_t, y_p, scaler_y, input_cols)
        
        # Draw Export
        self._setup_export(self.tabview.tab("Export"), y_t, y_p, model)

    def _plot_pred_actual(self, parent, y_t, y_p, out_col):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax.set_facecolor(COLORS["bg_card"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values(): spine.set_color(COLORS["border"])
        
        ax.scatter(y_t, y_p, alpha=0.6, color=COLORS["cyan"], s=15)
        
        min_v, max_v = min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())
        ax.plot([min_v, max_v], [min_v, max_v], color=COLORS["green"], linestyle="--")
        
        ax.set_title(f"Predicted vs Actual ({out_col})", color="white")
        ax.set_xlabel("Actual", color="white")
        ax.set_ylabel("Predicted", color="white")
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)

    def _plot_residuals(self, parent, y_t, y_p):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        res = y_t - y_p
        
        for ax in [ax1, ax2]:
            fig.patch.set_facecolor(COLORS["bg_card"])
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text"])
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])
            
        ax1.scatter(y_p, res, color=COLORS["orange"], alpha=0.6, s=15)
        ax1.axhline(0, color=COLORS["green"], linestyle="--")
        ax1.set_title("Residuals vs Predicted", color="white")
        ax1.set_xlabel("Predicted", color="white")
        ax1.set_ylabel("Residual", color="white")
        
        ax2.hist(res, bins=40, color=COLORS["magenta"], alpha=0.7)
        ax2.set_title("Residual Distribution", color="white")
        ax2.set_xlabel("Residual", color="white")
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)

    def _plot_fi(self, parent, model, X_test, y_t, y_p, scaler_y, input_cols):
        btn = ctk.CTkButton(parent, text="⚡ Compute Permutation Importance", command=lambda: self._compute_fi(parent, btn, model, X_test, y_t, y_p, scaler_y, input_cols))
        btn.pack(pady=20)
        
    def _compute_fi(self, parent, btn, model, X_test, y_t, y_p, scaler_y, input_cols):
        btn.configure(state="disabled", text="Computing...")
        parent.update()
        
        baseline_mse = mean_squared_error(y_t, y_p)
        importances = []
        
        for i in range(len(input_cols)):
            X_perm = np.copy(X_test)
            np.random.shuffle(X_perm[:, i])
            y_perm_pred = model.predict(X_perm, verbose=0)
            if scaler_y is not None:
                y_perm_pred = scaler_y.inverse_transform(y_perm_pred)
            y_perm_pred = y_perm_pred.flatten()
            importances.append(mean_squared_error(y_t, y_perm_pred) - baseline_mse)
            
        btn.destroy()
        
        # Sort
        idxs = np.argsort(importances)
        sorted_cols = [input_cols[i] for i in idxs]
        sorted_imps = [importances[i] for i in idxs]
        
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax.set_facecolor(COLORS["bg_card"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values(): spine.set_color(COLORS["border"])
        
        ax.barh(sorted_cols, sorted_imps, color=COLORS["cyan"])
        ax.set_title("Permutation Feature Importance (ΔMSE)", color="white")
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)

    def _setup_export(self, parent, y_t, y_p, model):
        # Result preview
        self.df_res = pd.DataFrame({"Actual": y_t, "Predicted": y_p, "Residual": y_t - y_p, "Abs Error": np.abs(y_t - y_p)})
        
        # Export Buttons
        b_frame = ctk.CTkFrame(parent, fg_color="transparent")
        b_frame.pack(pady=30)
        
        ctk.CTkButton(b_frame, text="📄 Download CSV", command=self._dl_csv).pack(side="left", padx=10)
        ctk.CTkButton(b_frame, text="📊 Download Excel", command=self._dl_excel).pack(side="left", padx=10)
        ctk.CTkButton(b_frame, text="🧠 Download Model (.h5)", command=lambda: self._dl_model(model)).pack(side="left", padx=10)

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
        fn = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("H5 Model", "*.h5")])
        if fn:
            model.save(fn)
            messagebox.showinfo("Success", f"Model saved to {fn}")
