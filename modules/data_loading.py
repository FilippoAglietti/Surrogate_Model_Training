import customtkinter as ctk
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from utils.theme import COLORS, FONTS
from utils.state import get_state, set_state


class DataLoadingFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

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

        # Data Info Frame (stats)
        self.info_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        self.info_frame.grid(row=2, column=0, pady=(10, 5), padx=40, sticky="ew")
        self.info_frame.grid_columnconfigure(0, weight=1)

        self.shape_lbl = ctk.CTkLabel(self.info_frame, text="Rows: 0 | Columns: 0", font=FONTS["body"])
        self.shape_lbl.grid(row=0, column=0, pady=10, padx=20, sticky="w")

        self.stats_box = ctk.CTkTextbox(self.info_frame, height=160, font=FONTS["code"], fg_color="#0D1117", text_color=COLORS["text"])
        self.stats_box.grid(row=1, column=0, pady=(0, 15), padx=20, sticky="ew")
        self.stats_box.configure(state="disabled")

        # Data Explorer (Preview + Missing Values) — populated after load
        self.explorer_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        self.explorer_frame.grid(row=3, column=0, pady=5, padx=40, sticky="ew")
        self.explorer_frame.grid_columnconfigure(0, weight=1)
        self._explorer_built = False

        # Column Selection Frame
        self.col_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        self.col_frame.grid(row=4, column=0, pady=10, padx=40, sticky="ew")
        self.col_frame.grid_columnconfigure(0, weight=1)
        self.col_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.col_frame, text="Target (Output) Columns", font=FONTS["header"], text_color=COLORS["magenta"]).grid(row=0, column=0, pady=10, padx=20, sticky="w")
        self.targets_scroll = ctk.CTkScrollableFrame(self.col_frame, height=200, fg_color=COLORS["bg"])
        self.targets_scroll.grid(row=1, column=0, pady=(0, 10), padx=20, sticky="ew")
        self.target_vars = {}

        ctk.CTkLabel(self.col_frame, text="Features (Input) Columns", font=FONTS["header"], text_color=COLORS["cyan"]).grid(row=0, column=1, pady=10, padx=20, sticky="w")
        self.features_scroll = ctk.CTkScrollableFrame(self.col_frame, height=200, fg_color=COLORS["bg"])
        self.features_scroll.grid(row=1, column=1, pady=(0, 10), padx=20, sticky="ew")
        self.feature_vars = {}

        # Warning label for type conflicts / overlap
        self.type_warn_lbl = ctk.CTkLabel(
            self.col_frame, text="", font=("Helvetica", 12),
            text_color=COLORS["orange"], wraplength=800, justify="left"
        )
        self.type_warn_lbl.grid(row=2, column=0, columnspan=2, pady=(0, 10), padx=20, sticky="w")

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
        self.proceed_btn.grid(row=5, column=0, pady=30, padx=40, sticky="ew")

        self._col_dtypes = {}

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _get_col_type(self, series):
        if pd.api.types.is_numeric_dtype(series):
            return "NUM"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "DATE"
        return "CAT"

    # ─── file loading ─────────────────────────────────────────────────────────

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=(("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not filename:
            return

        try:
            df = pd.read_csv(filename) if filename.endswith('.csv') else pd.read_excel(filename)

            set_state("df", df)
            set_state("data_loaded", True)
            set_state("preprocessed", False)
            set_state("session_unsaved", True)

            self.status_lbl.configure(text=f"Loaded: {os.path.basename(filename)}", text_color=COLORS["green"])
            self.shape_lbl.configure(text=f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

            # Stats textbox
            desc = df.describe().T
            desc_str = desc[['min', 'max', 'mean', 'std']].round(3).to_string()
            self.stats_box.configure(state="normal")
            self.stats_box.delete("0.0", "end")
            self.stats_box.insert("end", "--- DATASET STATISTICS ---\n" + desc_str)
            self.stats_box.configure(state="disabled")

            # Data explorer (preview + missing values)
            self._build_data_explorer(df)

            # Column type map (ALL columns, not just numeric)
            self._col_dtypes = {col: self._get_col_type(df[col]) for col in df.columns}

            # Rebuild column checkboxes
            for w in self.features_scroll.winfo_children(): w.destroy()
            for w in self.targets_scroll.winfo_children(): w.destroy()
            self.feature_vars.clear()
            self.target_vars.clear()

            for i, col in enumerate(df.columns):
                col_type = self._col_dtypes[col]
                if col_type == "NUM":
                    badge_color = COLORS["cyan"]
                elif col_type == "DATE":
                    badge_color = COLORS["orange"]
                else:
                    badge_color = COLORS["red"]

                # --- Target row ---
                t_row = ctk.CTkFrame(self.targets_scroll, fg_color="transparent")
                t_row.grid(row=i, column=0, sticky="ew", pady=2)
                t_var = ctk.StringVar(value="on" if i == len(df.columns) - 1 else "off")
                ctk.CTkCheckBox(t_row, text=col, variable=t_var, onvalue="on", offvalue="off",
                                command=self.check_ready).pack(side="left", padx=5)
                ctk.CTkLabel(t_row, text=col_type, font=("Helvetica", 10, "bold"),
                             text_color=badge_color, width=38).pack(side="left")
                self.target_vars[col] = t_var

                # --- Feature row ---
                f_row = ctk.CTkFrame(self.features_scroll, fg_color="transparent")
                f_row.grid(row=i, column=0, sticky="ew", pady=2)
                f_var = ctk.StringVar(value="off" if i == len(df.columns) - 1 else "on")
                ctk.CTkCheckBox(f_row, text=col, variable=f_var, onvalue="on", offvalue="off",
                                command=self.check_ready).pack(side="left", padx=5)
                ctk.CTkLabel(f_row, text=col_type, font=("Helvetica", 10, "bold"),
                             text_color=badge_color, width=38).pack(side="left")
                self.feature_vars[col] = f_var

            self.check_ready()

        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{str(e)}")

    # ─── data explorer ────────────────────────────────────────────────────────

    def _build_data_explorer(self, df):
        for w in self.explorer_frame.winfo_children():
            w.destroy()

        tabview = ctk.CTkTabview(self.explorer_frame, fg_color=COLORS["bg"], height=320)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)
        tabview.add("Preview (first 50 rows)")
        tabview.add("Missing Values")

        self._build_treeview(tabview.tab("Preview (first 50 rows)"), df.head(50))
        self._build_missing_values(tabview.tab("Missing Values"), df)

    def _build_treeview(self, parent, df):
        # Style dark Treeview
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.Treeview",
                        background="#1A1A2E", foreground=COLORS["text"],
                        fieldbackground="#1A1A2E", rowheight=24,
                        font=("Helvetica", 11))
        style.configure("Dark.Treeview.Heading",
                        background=COLORS["bg_card"], foreground=COLORS["cyan"],
                        font=("Helvetica", 11, "bold"), relief="flat")
        style.map("Dark.Treeview", background=[("selected", COLORS["primary_dark"])])

        container = tk.Frame(parent, bg="#1A1A2E")
        container.pack(fill="both", expand=True)

        cols = list(df.columns)
        tree = ttk.Treeview(container, columns=cols, show="headings", style="Dark.Treeview")

        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)

        col_width = max(80, min(160, 1000 // max(len(cols), 1)))
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=col_width, minwidth=60, anchor="center")

        tree.tag_configure("odd",  background="#16213E")
        tree.tag_configure("even", background="#1A1A2E")

        for idx, (_, row) in enumerate(df.iterrows()):
            tag = "odd" if idx % 2 == 0 else "even"
            values = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
            tree.insert("", "end", values=values, tags=(tag,))

    def _build_missing_values(self, parent, df):
        missing = df.isnull().sum()
        pct = (missing / len(df) * 100).round(2)
        total_missing = int(missing.sum())

        if total_missing == 0:
            ctk.CTkLabel(parent, text="✓ No missing values found in the dataset.",
                         font=FONTS["header"], text_color=COLORS["green"]).pack(pady=40)
            return

        mv_df = pd.DataFrame({"Missing": missing, "Pct": pct})
        mv_df = mv_df[mv_df["Missing"] > 0].sort_values("Pct", ascending=True)

        ctk.CTkLabel(parent,
                     text=f"Total missing: {total_missing} cells across {len(mv_df)} columns",
                     font=("Helvetica", 12), text_color=COLORS["orange"]).pack(pady=(8, 0))

        plt.style.use("dark_background")
        fig_h = max(2.5, len(mv_df) * 0.45 + 0.8)
        fig, ax = plt.subplots(figsize=(8, fig_h), dpi=90)
        fig.patch.set_facecolor(COLORS["bg_card"])
        ax.set_facecolor(COLORS["bg_card"])

        bar_colors = [COLORS["red"] if p > 20 else COLORS["orange"] if p > 5 else COLORS["cyan"]
                      for p in mv_df["Pct"]]
        bars = ax.barh(mv_df.index, mv_df["Pct"], color=bar_colors, alpha=0.85)

        for bar, pv, mv in zip(bars, mv_df["Pct"], mv_df["Missing"]):
            ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                    f"{pv:.1f}%  ({int(mv)} rows)", va="center", ha="left",
                    color=COLORS["text"], fontsize=9)

        ax.set_xlabel("Missing %", color=COLORS["text"])
        ax.set_xlim(0, min(100, mv_df["Pct"].max() * 1.35))
        ax.tick_params(colors=COLORS["text"], labelsize=9)
        for spine in ax.spines.values(): spine.set_color(COLORS["border"])
        ax.set_title("Missing Values by Column", color="white", fontsize=11)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=5)
        plt.close(fig)

    # ─── column validation ────────────────────────────────────────────────────

    def check_ready(self):
        targets  = [c for c, v in self.target_vars.items()  if v.get() == "on"]
        features = [c for c, v in self.feature_vars.items() if v.get() == "on"]
        overlap  = set(targets) & set(features)

        non_num_t = [c for c in targets  if self._col_dtypes.get(c, "NUM") != "NUM"]
        non_num_f = [c for c in features if self._col_dtypes.get(c, "NUM") != "NUM"]
        non_num   = list(set(non_num_t + non_num_f))

        warnings = []
        if overlap:
            warnings.append(f"Overlap in selections: {list(overlap)}")
        if non_num:
            warnings.append(f"Non-numeric columns selected (CAT/DATE): {non_num} — remove them to proceed")

        self.type_warn_lbl.configure(text=("⚠  " + "  |  ".join(warnings)) if warnings else "")

        ok = len(targets) > 0 and len(features) > 0 and not overlap and not non_num
        self.proceed_btn.configure(state="normal" if ok else "disabled")
        if ok:
            set_state("output_column", targets)
            set_state("input_columns", features)

    # ─── navigation ───────────────────────────────────────────────────────────

    def go_to_preprocessing(self):
        self.master.navigate_to("🔧  Preprocessing")

    def restore_from_session(self, config: dict) -> None:
        """Restore the Data Loading UI from AppState after a session load."""
        df = get_state("df")
        if df is None:
            return

        self.status_lbl.configure(
            text="Loaded (from session)",
            text_color=COLORS["green"],
        )
        self.shape_lbl.configure(text=f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        # Stats textbox
        desc = df.describe().T
        desc_str = desc[["min", "max", "mean", "std"]].round(3).to_string()
        self.stats_box.configure(state="normal")
        self.stats_box.delete("0.0", "end")
        self.stats_box.insert("end", "--- DATASET STATISTICS ---\n" + desc_str)
        self.stats_box.configure(state="disabled")

        self._build_data_explorer(df)
        self._col_dtypes = {col: self._get_col_type(df[col]) for col in df.columns}

        input_cols  = config.get("input_columns",  [])
        output_cols = config.get("output_column",  [])

        for w in self.features_scroll.winfo_children(): w.destroy()
        for w in self.targets_scroll.winfo_children():  w.destroy()
        self.feature_vars.clear()
        self.target_vars.clear()

        for i, col in enumerate(df.columns):
            col_type = self._col_dtypes[col]
            badge_color = (
                COLORS["cyan"]    if col_type == "NUM"  else
                COLORS["orange"]  if col_type == "DATE" else
                COLORS["red"]
            )

            t_row = ctk.CTkFrame(self.targets_scroll, fg_color="transparent")
            t_row.grid(row=i, column=0, sticky="ew", pady=2)
            t_var = ctk.StringVar(value="on" if col in output_cols else "off")
            ctk.CTkCheckBox(t_row, text=col, variable=t_var, onvalue="on", offvalue="off",
                            command=self.check_ready).pack(side="left", padx=5)
            ctk.CTkLabel(t_row, text=col_type, font=("Helvetica", 10, "bold"),
                         text_color=badge_color, width=38).pack(side="left")
            self.target_vars[col] = t_var

            f_row = ctk.CTkFrame(self.features_scroll, fg_color="transparent")
            f_row.grid(row=i, column=0, sticky="ew", pady=2)
            f_var = ctk.StringVar(value="on" if col in input_cols else "off")
            ctk.CTkCheckBox(f_row, text=col, variable=f_var, onvalue="on", offvalue="off",
                            command=self.check_ready).pack(side="left", padx=5)
            ctk.CTkLabel(f_row, text=col_type, font=("Helvetica", 10, "bold"),
                         text_color=badge_color, width=38).pack(side="left")
            self.feature_vars[col] = f_var

        self.check_ready()

    def on_show(self):
        # If data was restored from a session but widgets are empty, restore them
        if get_state("data_loaded") and get_state("df") is not None and not self.feature_vars:
            self.restore_from_session({
                "input_columns": get_state("input_columns", []),
                "output_column": get_state("output_column", []),
            })
