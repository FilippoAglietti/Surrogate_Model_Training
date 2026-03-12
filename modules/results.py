import customtkinter as ctk
import numpy as np
import pandas as pd
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
        self.tabview.add("Export & Wrapper")
        
        # We need a frame holding the dropdown if num targets > 4
        self._setup_pred_actual(self.tabview.tab("Pred vs Actual"))
        self._setup_series(self.tabview.tab("Test Index Series"))
        self._setup_residuals(self.tabview.tab("Residuals"))
        
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
