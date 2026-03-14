import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from itertools import combinations
import zipfile
import pickle
import os
import tempfile
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score

from utils.theme import COLORS, FONTS
from utils.state import set_state
from utils.plot_utils import add_save_button

class InferenceFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        self.header = ctk.CTkLabel(self, text="INFERENCE & PREDICTION 🔮", font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(30, 10), sticky="w", padx=30)
        
        # Top bar: Load Wrapper
        load_frame = ctk.CTkFrame(self, fg_color="transparent")
        load_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        
        self.load_btn = ctk.CTkButton(load_frame, text="📂 LOAD MODEL WRAPPER (.zip)", font=FONTS["header"], height=40, command=self.load_wrapper)
        self.load_btn.pack(side="left", padx=10)
        
        self.model_status_lbl = ctk.CTkLabel(load_frame, text="No model loaded.", text_color=COLORS["text_dim"])
        self.model_status_lbl.pack(side="left", padx=20)
        
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=2, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        self.tabview = ctk.CTkTabview(self.content_frame, fg_color=COLORS["bg_card"], state="disabled")
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.tabview.add("Batch Prediction (Excel)")
        self.tabview.add("1D Sensitivity")
        self.tabview.add("2D Sensitivity")
        
        self.model = None
        self.meta = {}
        
        # Debounce timer for sliders
        self._update_job = None
        self._slider_vars = {}
        self._plots_built = False

    def load_wrapper(self):
        fn = filedialog.askopenfilename(title="Select Model Wrapper", filetypes=[("ZIP Files", "*.zip")])
        if not fn: return
        
        try:
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(fn, 'r') as zf:
                    zf.extractall(td)
                
                keras_path = os.path.join(td, "model.keras")
                meta_path = os.path.join(td, "metadata.pkl")
                
                if not os.path.exists(keras_path) or not os.path.exists(meta_path):
                    raise Exception("Invalid wrapper: Missing model.keras or metadata.pkl")
                    
                self.model = load_model(keras_path)
                with open(meta_path, "rb") as f:
                    self.meta = pickle.load(f)
                    
            self.model_status_lbl.configure(text=f"✓ Loaded: {os.path.basename(fn)}\nInputs: {len(self.meta['input_columns'])} | Targets: {len(self.meta['output_columns'])}", text_color=COLORS["green"])
            self.tabview.configure(state="normal")
            
            self._setup_batch_tab()
            self._setup_sensitivity_tab()
            self._setup_2d_sensitivity_tab()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load wrapper:\n{e}")

    def _predict_raw(self, X_raw_df):
        X = X_raw_df.values
        
        # Apply Input scaling
        scaler_x = self.meta.get("scaler_X")
        if scaler_x is not None:
            X = scaler_x.transform(X)
            
        # Apply Input PCA
        pca_x = self.meta.get("pca_X")
        if pca_x is not None:
            X = pca_x.transform(X)
            
        # Predict
        y_pred = self.model.predict(X, verbose=0)
        
        # Apply Reverse Output PCA
        pca_y = self.meta.get("pca_y")
        if pca_y is not None:
            y_pred = pca_y.inverse_transform(y_pred)
            
        # Apply Reverse Output Scaling
        scaler_y = self.meta.get("scaler_y")
        if scaler_y is not None:
            y_pred = scaler_y.inverse_transform(y_pred)
            
        return y_pred

    # ========================== BATCH PREDICTION ==========================
    
    def _setup_batch_tab(self):
        tab = self.tabview.tab("Batch Prediction (Excel)")
        for w in tab.winfo_children(): w.destroy()
        
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)
        
        cmd_frame = ctk.CTkFrame(tab, fg_color="transparent")
        cmd_frame.grid(row=0, column=0, pady=10, sticky="ew")
        
        self.bp_btn = ctk.CTkButton(cmd_frame, text="Upload Excel & Predict", command=self._run_batch_predict)
        self.bp_btn.pack(side="left", padx=10)
        
        self.bp_dl = ctk.CTkButton(cmd_frame, text="Download Results", command=self._dl_batch, state="disabled", fg_color=COLORS["green"], hover_color="#2E7D32")
        self.bp_dl.pack(side="left", padx=10)
        
        self.bp_res_lbl = ctk.CTkLabel(cmd_frame, text="", text_color=COLORS["cyan"])
        self.bp_res_lbl.pack(side="left", padx=20)
        
        self.bp_plot_frame = ctk.CTkScrollableFrame(tab, fg_color=COLORS["bg"])
        self.bp_plot_frame.grid(row=2, column=0, sticky="nsew", pady=10)
        
        self.bp_df_res = None

    def _run_batch_predict(self):
        fn = filedialog.askopenfilename(title="Select Batch Data", filetypes=[("Excel/CSV", "*.xlsx *.csv")])
        if not fn: return
        
        try:
            df = pd.read_csv(fn) if fn.endswith(".csv") else pd.read_excel(fn)
            req_cols = self.meta["input_columns"]
            missing = [c for c in req_cols if c not in df.columns]
            if missing:
                messagebox.showerror("Error", f"Dataset is missing required inputs:\n{missing}")
                return
                
            X_raw = df[req_cols]
            y_pred = self._predict_raw(X_raw)
            
            # Store results
            res_df = df.copy()
            out_cols = self.meta["output_columns"]
            
            has_actuals = all(c in df.columns for c in out_cols)
            
            for i, c in enumerate(out_cols):
                res_df[f"Predicted_{c}"] = y_pred[:, i]
                if has_actuals:
                    res_df[f"Residual_{c}"] = res_df[c] - y_pred[:, i]
                    
            self.bp_df_res = res_df
            self.bp_dl.configure(state="normal")
            
            if has_actuals:
                y_act = df[out_cols].values
                r2 = r2_score(y_act, y_pred)
                self.bp_res_lbl.configure(text=f"Prediction Complete. Average R² against actuals: {r2:.4f}")
                self._plot_batch_results(y_act, y_pred, out_cols)
            else:
                self.bp_res_lbl.configure(text="Prediction Complete. No actuals found in file to compare against.")
                for w in self.bp_plot_frame.winfo_children(): w.destroy()
                
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _plot_batch_results(self, y_a, y_p, cols):
        for w in self.bp_plot_frame.winfo_children(): w.destroy()
        
        # Pred vs Actual Grid
        fig, axes = plt.subplots(1, len(cols), figsize=(max(5*len(cols), 8), 4), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        if len(cols) == 1: axes = [axes]
        
        for i, ax in enumerate(axes):
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text"])
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])
            
            ax.scatter(y_a[:, i], y_p[:, i], color=COLORS["cyan"], alpha=0.6, s=10)
            ax.set_title(f"Pred vs Actual ({cols[i]})", color="white")
            
        plt.tight_layout()
        cv1 = FigureCanvasTkAgg(fig, master=self.bp_plot_frame)
        cv1.draw()
        cv1.get_tk_widget().pack(fill="x", pady=20)
        add_save_button(self.bp_plot_frame, cv1, "batch_pred_vs_actual.png")

        # Pred vs Test Num Grid
        fig2, axes2 = plt.subplots(1, len(cols), figsize=(max(5*len(cols), 8), 4), dpi=100)
        fig2.patch.set_facecolor(COLORS["bg_card"])
        if len(cols) == 1: axes2 = [axes2]
        
        x_ax = np.arange(len(y_a))
        for i, ax in enumerate(axes2):
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text"])
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])
            
            ax.plot(x_ax, y_a[:, i], color=COLORS["orange"], label="Actual", alpha=0.8)
            ax.plot(x_ax, y_p[:, i], color=COLORS["cyan"], label="Predicted", alpha=0.8, linestyle="--")
            ax.set_title(f"Series ({cols[i]})", color="white")
            ax.legend(prop={'size': 8})
            
        plt.tight_layout()
        cv2 = FigureCanvasTkAgg(fig2, master=self.bp_plot_frame)
        cv2.draw()
        cv2.get_tk_widget().pack(fill="x", pady=20)
        add_save_button(self.bp_plot_frame, cv2, "batch_series.png")

    def _dl_batch(self):
        if self.bp_df_res is None: return
        fn = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")])
        if fn:
            if fn.endswith(".csv"): self.bp_df_res.to_csv(fn, index=False)
            else: self.bp_df_res.to_excel(fn, index=False)
            messagebox.showinfo("Success", f"Saved to {fn}")

    # ========================== SENSITIVITY ==========================
    
    def _setup_sensitivity_tab(self):
        tab = self.tabview.tab("1D Sensitivity")
        for w in tab.winfo_children(): w.destroy()
        
        tab.grid_columnconfigure(0, weight=0) # Sliders
        tab.grid_columnconfigure(1, weight=1) # Plots
        tab.grid_rowconfigure(0, weight=1)
        
        # Sliders Column
        sf = ctk.CTkScrollableFrame(tab, width=300, fg_color=COLORS["bg"])
        sf.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        ctk.CTkLabel(sf, text="Base Point (Inputs)", font=FONTS["header"]).pack(pady=10)
        
        self._slider_vars.clear()
        
        inputs = self.meta["input_columns"]
        t_min = self.meta["train_min"]
        t_max = self.meta["train_max"]
        t_mean = self.meta["train_mean"]
        
        for i, col in enumerate(inputs):
            f = ctk.CTkFrame(sf, fg_color="transparent")
            f.pack(fill="x", pady=5)
            
            lbl_val = ctk.CTkLabel(f, text=f"{t_mean[col]:.2f}", font=("Helvetica", 12, "bold"), text_color=COLORS["cyan"])
            lbl_val.pack(side="right")
            
            lbl_name = ctk.CTkLabel(f, text=col)
            lbl_name.pack(side="left")
            
            var = ctk.DoubleVar(value=t_mean[col])
            self._slider_vars[col] = (var, lbl_val)
            
            slider = ctk.CTkSlider(sf, variable=var, from_=t_min[col], to=t_max[col], command=lambda val, c=col: self._on_slider_change(c, val))
            slider.pack(fill="x", pady=(0, 10))
            
        # Plots Column
        self.sens_plot_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.sens_plot_frame.grid(row=0, column=1, sticky="nsew")
        self.sens_plot_frame.grid_columnconfigure(0, weight=1)
        self.sens_plot_frame.grid_rowconfigure(0, weight=1)
        
        # Limit plots to 2 targets to avoid overcrowding
        out_cols = self.meta["output_columns"][:2]
        if len(self.meta["output_columns"]) > 2:
            ctk.CTkLabel(self.sens_plot_frame, text=f"Note: Only plotting first 2 targets: {out_cols}", text_color=COLORS["orange"]).pack(pady=5)
            
        self.sens_canvas = None
        self._draw_sensitivity_plots()

    def _on_slider_change(self, col, val):
        var, lbl = self._slider_vars[col]
        lbl.configure(text=f"{float(val):.2f}")
        
        # Debounce to prevent UI lag
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        self._update_job = self.after(300, self._draw_sensitivity_plots)

    def _draw_sensitivity_plots(self):
        for w in self.sens_plot_frame.winfo_children(): 
            if isinstance(w, ctk.CTkLabel): continue # Keep the note
            w.destroy()
            
        inputs = self.meta["input_columns"]
        out_cols = self.meta["output_columns"][:2]
        n_inputs = len(inputs)
        
        # Build grid
        cols_grid = min(3, n_inputs)
        rows_grid = int(np.ceil(n_inputs / cols_grid))
        
        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(4 * cols_grid, 3 * rows_grid), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_card"])
        if n_inputs == 1: axes = [axes]
        else: axes = axes.flatten()
        
        # Collect base point
        base_df = pd.DataFrame([{c: self._slider_vars[c][0].get() for c in inputs}])
        base_pred = self._predict_raw(base_df)[0]  # shape (n_outputs,)

        for i, inp_col in enumerate(inputs):
            ax = axes[i]
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text"], labelsize=8)
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])

            # Generate 50 points for this input feature
            scan = np.linspace(self.meta["train_min"][inp_col], self.meta["train_max"][inp_col], 50)

            # Construct scanning dataframe
            scan_df = pd.DataFrame(np.repeat(base_df.values, 50, axis=0), columns=inputs)
            scan_df[inp_col] = scan

            # Predict
            y_scan = self._predict_raw(scan_df)

            # Plot each target line + base point marker
            colors = [COLORS["cyan"], COLORS["magenta"]]
            base_x = base_df[inp_col].values[0]
            for t_idx, t_col in enumerate(out_cols):
                ax.plot(scan, y_scan[:, t_idx], color=colors[t_idx % len(colors)], label=t_col, linewidth=2)
                ax.scatter([base_x], [base_pred[t_idx]],
                           color="white", s=60, zorder=5, marker="x", linewidths=2)

            ax.set_title(inp_col, color="white", fontsize=10)
            if i == 0: ax.legend(fontsize=8)
            
        # Hide extra axes
        for i in range(n_inputs, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        self.sens_canvas = FigureCanvasTkAgg(fig, master=self.sens_plot_frame)
        self.sens_canvas.draw()
        self.sens_canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
        add_save_button(self.sens_plot_frame, self.sens_canvas, "1d_sensitivity.png")

    # ========================== 2D SENSITIVITY CONTOUR ==========================

    def _setup_2d_sensitivity_tab(self):
        tab = self.tabview.tab("2D Sensitivity")
        for w in tab.winfo_children(): w.destroy()

        inputs     = self.meta["input_columns"]
        out_cols   = self.meta["output_columns"]
        t_min      = self.meta["train_min"]
        t_max      = self.meta["train_max"]
        t_mean     = self.meta["train_mean"]

        tab.grid_columnconfigure(0, weight=0)   # Controls
        tab.grid_columnconfigure(1, weight=1)   # Plot area
        tab.grid_rowconfigure(0, weight=1)

        # ── Left: Controls ────────────────────────────────────────────────
        ctrl = ctk.CTkScrollableFrame(tab, width=360, fg_color=COLORS["bg"])
        ctrl.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # Output selector
        ctk.CTkLabel(ctrl, text="Output to plot", font=FONTS["header"],
                     text_color=COLORS["magenta"]).pack(anchor="w", padx=10, pady=(10, 4))
        self._2d_out_var = ctk.StringVar(value=out_cols[0])
        ctk.CTkComboBox(ctrl, values=out_cols, variable=self._2d_out_var, width=320).pack(padx=10, pady=(0, 12))

        # Active inputs (max 5)
        ctk.CTkLabel(ctrl, text="Active Inputs  (max 5 — will be contour axes)",
                     font=FONTS["header"], text_color=COLORS["cyan"],
                     wraplength=320).pack(anchor="w", padx=10, pady=(6, 4))

        self._2d_active_vars   = {}   # col → BooleanVar
        self._2d_slider_vars   = {}   # col → (DoubleVar, label_widget)
        self._2d_fixed_vars    = {}   # col → StringVar (entry text)
        self._2d_row_frames    = {}   # col → frame holding slider OR entry

        MAX_ACTIVE = 5

        def toggle_input(col):
            # Enforce maximum
            n_active = sum(1 for v in self._2d_active_vars.values() if v.get())
            if n_active > MAX_ACTIVE:
                self._2d_active_vars[col].set(False)
                self._2d_active_count_lbl.configure(
                    text=f"Active: {MAX_ACTIVE}/{MAX_ACTIVE}  (max reached)",
                    text_color=COLORS["red"])
                return
            self._2d_active_count_lbl.configure(
                text=f"Active: {n_active}/{MAX_ACTIVE}",
                text_color=COLORS["cyan"] if n_active <= MAX_ACTIVE else COLORS["red"])
            _refresh_input_row(col)

        def _refresh_input_row(col):
            frame = self._2d_row_frames[col]
            for w in frame.winfo_children(): w.destroy()
            if self._2d_active_vars[col].get():
                # Show slider + value label
                var, _ = self._2d_slider_vars[col]
                lbl = ctk.CTkLabel(frame, text=f"{var.get():.3g}",
                                   font=("Helvetica", 11, "bold"), text_color=COLORS["cyan"], width=55)
                lbl.pack(side="right")
                self._2d_slider_vars[col] = (var, lbl)

                def on_slide(val, c=col):
                    sv, lb = self._2d_slider_vars[c]
                    lb.configure(text=f"{float(val):.3g}")

                sl = ctk.CTkSlider(frame, variable=var,
                                   from_=t_min[col], to=t_max[col],
                                   command=on_slide)
                sl.pack(fill="x", expand=True, padx=(0, 4))
            else:
                # Show fixed value entry
                sv = self._2d_fixed_vars[col]
                ctk.CTkLabel(frame, text="Fixed:", font=("Helvetica", 10),
                             text_color=COLORS["text_dim"]).pack(side="left", padx=4)
                ctk.CTkEntry(frame, textvariable=sv, width=90).pack(side="left")

        for col in inputs:
            active_var = ctk.BooleanVar(value=False)
            self._2d_active_vars[col] = active_var

            slider_var = ctk.DoubleVar(value=t_mean[col])
            self._2d_slider_vars[col] = (slider_var, None)

            fixed_var = ctk.StringVar(value=f"{t_mean[col]:.4g}")
            self._2d_fixed_vars[col] = fixed_var

            row = ctk.CTkFrame(ctrl, fg_color="transparent")
            row.pack(fill="x", padx=6, pady=3)

            chk = ctk.CTkCheckBox(row, text=col, variable=active_var,
                                  command=lambda c=col: toggle_input(c), width=140)
            chk.pack(side="left")

            val_frame = ctk.CTkFrame(row, fg_color="transparent")
            val_frame.pack(side="left", fill="x", expand=True, padx=(6, 0))
            self._2d_row_frames[col] = val_frame
            _refresh_input_row(col)

        self._2d_active_count_lbl = ctk.CTkLabel(ctrl, text="Active: 0/5",
                                                  font=("Helvetica", 11),
                                                  text_color=COLORS["cyan"])
        self._2d_active_count_lbl.pack(anchor="w", padx=10, pady=6)

        # Resolution
        ctk.CTkLabel(ctrl, text="Grid Resolution (N×N):", font=("Helvetica", 11)).pack(anchor="w", padx=10)
        self._2d_res_var = ctk.IntVar(value=30)
        ctk.CTkSlider(ctrl, variable=self._2d_res_var, from_=10, to=60, number_of_steps=10).pack(fill="x", padx=10)

        # Generate button
        ctk.CTkButton(ctrl, text="⚡ Generate Contour Grid", height=40,
                      font=("Helvetica", 13, "bold"),
                      command=self._generate_2d_contours).pack(fill="x", padx=10, pady=16)

        self._2d_status_lbl = ctk.CTkLabel(ctrl, text="", font=("Helvetica", 11),
                                           text_color=COLORS["text_dim"], wraplength=320)
        self._2d_status_lbl.pack(anchor="w", padx=10)

        # ── Right: Plot area ──────────────────────────────────────────────
        self._2d_plot_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self._2d_plot_frame.grid(row=0, column=1, sticky="nsew")
        self._2d_plot_frame.grid_columnconfigure(0, weight=1)
        self._2d_plot_frame.grid_rowconfigure(0, weight=1)

        ctk.CTkLabel(self._2d_plot_frame,
                     text="Select active inputs and click 'Generate'.",
                     text_color=COLORS["text_dim"], font=FONTS["header"]).pack(pady=60)

    def _generate_2d_contours(self):
        inputs   = self.meta["input_columns"]
        out_cols = self.meta["output_columns"]
        t_min    = self.meta["train_min"]
        t_max    = self.meta["train_max"]

        active = [c for c in inputs if self._2d_active_vars[c].get()]
        if len(active) < 2:
            self._2d_status_lbl.configure(text="Select at least 2 active inputs.", text_color=COLORS["red"])
            return

        out_col = self._2d_out_var.get()
        out_idx = out_cols.index(out_col)

        # Fixed values for inactive inputs
        fixed_vals = {}
        for col in inputs:
            if col not in active:
                try:
                    fixed_vals[col] = float(self._2d_fixed_vars[col].get())
                except ValueError:
                    fixed_vals[col] = float(self.meta["train_mean"][col])

        # Base slider values for active inputs (used when not on axis)
        active_base = {col: self._2d_slider_vars[col][0].get() for col in active}

        pairs = list(combinations(active, 2))
        n_pairs = len(pairs)
        n_pts   = self._2d_res_var.get()

        self._2d_status_lbl.configure(
            text=f"Computing {n_pairs} × {n_pts}² predictions…", text_color=COLORS["cyan"])
        self._2d_plot_frame.update()

        for w in self._2d_plot_frame.winfo_children(): w.destroy()

        ncols_g = min(n_pairs, 3)
        nrows_g = int(np.ceil(n_pairs / ncols_g))

        plt.style.use("dark_background")
        fig, axes = plt.subplots(nrows_g, ncols_g,
                                 figsize=(ncols_g * 4.5, nrows_g * 4.0), dpi=90)
        fig.patch.set_facecolor(COLORS["bg_card"])
        axes_flat = np.array(axes).flatten() if n_pairs > 1 else [axes]

        try:
            for pi, (col_x, col_y) in enumerate(pairs):
                ax = axes_flat[pi]
                ax.set_facecolor(COLORS["bg_card"])
                ax.tick_params(colors=COLORS["text"], labelsize=8)
                for spine in ax.spines.values(): spine.set_color(COLORS["border"])

                xs = np.linspace(t_min[col_x], t_max[col_x], n_pts)
                ys = np.linspace(t_min[col_y], t_max[col_y], n_pts)
                XX, YY = np.meshgrid(xs, ys)

                n_total = n_pts * n_pts
                rows_dict = {}
                # Fixed inputs
                for col, val in fixed_vals.items():
                    rows_dict[col] = np.full(n_total, val)
                # Active inputs at their base value (not on this pair's axes)
                for col in active:
                    if col not in (col_x, col_y):
                        rows_dict[col] = np.full(n_total, active_base[col])
                # Axis inputs vary on the grid
                rows_dict[col_x] = XX.ravel()
                rows_dict[col_y] = YY.ravel()

                grid_df = pd.DataFrame({c: rows_dict[c] for c in inputs})
                y_pred  = self._predict_raw(grid_df)
                ZZ = y_pred[:, out_idx].reshape(n_pts, n_pts)

                cf = ax.contourf(XX, YY, ZZ, levels=20, cmap="plasma")
                ax.contour(XX, YY, ZZ, levels=8, colors="white", alpha=0.2, linewidths=0.5)

                # Base point marker
                ax.scatter([active_base[col_x]], [active_base[col_y]],
                           color="white", s=60, zorder=5, marker="x", linewidths=2,
                           label="Base point")

                cbar = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.03)
                cbar.ax.tick_params(colors=COLORS["text"], labelsize=7)
                cbar.set_label(out_col, color=COLORS["text"], fontsize=8)

                ax.set_xlabel(col_x, color=COLORS["text"], fontsize=9)
                ax.set_ylabel(col_y, color=COLORS["text"], fontsize=9)
                ax.set_title(f"{col_x} × {col_y}", color="white", fontsize=10)

            # Hide unused axes
            for i in range(n_pairs, len(axes_flat)):
                axes_flat[i].axis("off")

            plt.tight_layout(pad=1.0)
            canvas = FigureCanvasTkAgg(fig, master=self._2d_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            add_save_button(self._2d_plot_frame, canvas, "2d_sensitivity.png")
            plt.close(fig)

            self._2d_status_lbl.configure(
                text=f"✓ {n_pairs} contour plot(s) — output: {out_col}",
                text_color=COLORS["green"])

        except Exception as e:
            for w in self._2d_plot_frame.winfo_children(): w.destroy()
            ctk.CTkLabel(self._2d_plot_frame, text=f"Error: {e}",
                         text_color=COLORS["red"]).pack(pady=20)
            self._2d_status_lbl.configure(text=f"Error: {e}", text_color=COLORS["red"])

    def on_show(self):
        pass
