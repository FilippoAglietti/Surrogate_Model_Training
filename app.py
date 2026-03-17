import os
import customtkinter as ctk
from tkinter import filedialog, messagebox

from utils.theme import setup_theme, COLORS, FONTS, ASCII_BANNER
from utils.state import init_all_defaults, get_state, set_state

# Import all our module frames
from modules.data_loading import DataLoadingFrame
from modules.preprocessing import PreprocessingFrame
from modules.model_builder import ModelBuilderFrame
from modules.hyperopt import HyperoptFrame
from modules.results import ResultsFrame
from modules.inference import InferenceFrame

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Initialize app settings
        self.title("Surrogate Builder v4.0")
        self.geometry("1400x900")
        self.minsize(1000, 700)

        # Setup Theme & State
        setup_theme()
        init_all_defaults()

        # Configure grid for main window (1 row, 2 columns)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ── Sidebar ───────────────────────────────────────────────────────────
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=COLORS["bg_card"])
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)  # spacer between nav and session card

        # Logo
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="⚡\nSURROGATE\nBUILDER",
            font=FONTS["header"],
            text_color=COLORS["orange"],
            justify="center"
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 30))

        # Navigation Buttons
        self.nav_buttons = {}
        self.frames = {}

        self._pages = [
            ("📁  Data Loading", DataLoadingFrame),
            ("🔧  Preprocessing", PreprocessingFrame),
            ("🏗  Model Builder", ModelBuilderFrame),
            ("🔍  Hyperopt", HyperoptFrame),
            ("📊  Results", ResultsFrame),
            ("🔮  Inference", InferenceFrame),
        ]

        for i, (name, FrameClass) in enumerate(self._pages):
            btn = ctk.CTkButton(
                self.sidebar_frame,
                text=name,
                font=FONTS["body"],
                fg_color="transparent",
                text_color=COLORS["text"],
                hover_color=COLORS["border"],
                anchor="w",
                command=lambda n=name: self.navigate_to(n)
            )
            btn.grid(row=i + 1, column=0, padx=20, pady=5, sticky="ew")
            self.nav_buttons[name] = btn

            frame = FrameClass(self, corner_radius=0, fg_color=COLORS["bg"])
            self.frames[name] = frame

        # ── Session card (row 9, below spacer at row 8) ───────────────────────
        self._build_session_ui()

        # Status label
        self.status_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="v4.0 · CustomTkinter",
            font=("Consolas", 10),
            text_color=COLORS["text_dim"]
        )
        self.status_label.grid(row=10, column=0, padx=20, pady=(0, 14))

        # Start on default page and begin polling session UI
        default_page = get_state("nav_page")
        self.navigate_to(default_page)
        self._refresh_session_ui()
        
    # ── Navigation ────────────────────────────────────────────────────────────

    def navigate_to(self, page_name):
        """Switch the visible frame and update button styling."""
        set_state("nav_page", page_name)

        for name, btn in self.nav_buttons.items():
            if name == page_name:
                btn.configure(fg_color=COLORS["border"], text_color=COLORS["cyan"])
            else:
                btn.configure(fg_color="transparent", text_color=COLORS["text"])

        for frame in self.frames.values():
            frame.grid_forget()

        active_frame = self.frames[page_name]
        active_frame.grid(row=0, column=1, sticky="nsew")

        if hasattr(active_frame, "on_show"):
            active_frame.on_show()

    # ── Session UI ────────────────────────────────────────────────────────────

    def _build_session_ui(self):
        card = ctk.CTkFrame(
            self.sidebar_frame,
            fg_color=COLORS["bg_card_hover"],
            corner_radius=8,
        )
        card.grid(row=9, column=0, padx=12, pady=(0, 8), sticky="ew")
        card.grid_columnconfigure(0, weight=1)

        # Name row
        name_row = ctk.CTkFrame(card, fg_color="transparent")
        name_row.grid(row=0, column=0, padx=8, pady=(8, 2), sticky="ew")
        name_row.grid_columnconfigure(0, weight=1)

        self.session_name_lbl = ctk.CTkLabel(
            name_row, text="Untitled Session",
            font=("Consolas", 11, "bold"), text_color=COLORS["cyan"],
            anchor="w"
        )
        self.session_name_lbl.grid(row=0, column=0, sticky="ew")

        self.session_dot = ctk.CTkLabel(
            name_row, text="",
            font=("Consolas", 13), text_color=COLORS["orange"], width=14
        )
        self.session_dot.grid(row=0, column=1, sticky="e")

        # Button row
        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.grid(row=1, column=0, padx=6, pady=(2, 8), sticky="ew")

        for text, cmd in (
            ("New",  self._new_session),
            ("Open", self._open_session),
            ("Save", self._save_session),
        ):
            ctk.CTkButton(
                btn_row, text=text, height=26, width=62,
                font=("Consolas", 10),
                fg_color=COLORS["primary_dark"] if text == "Save" else COLORS["border"],
                hover_color=COLORS["primary"] if text == "Save" else COLORS["bg_card_hover"],
                text_color="#000" if text == "Save" else COLORS["text"],
                command=cmd,
            ).pack(side="left", padx=2)

    def _refresh_session_ui(self):
        """Update session name label and unsaved indicator; reschedules itself."""
        name  = get_state("session_name") or "Untitled Session"
        dirty = get_state("session_unsaved", False)
        # Truncate long names
        display = (name[:20] + "…") if len(name) > 20 else name
        self.session_name_lbl.configure(text=display)
        self.session_dot.configure(text="●" if dirty else "")
        self.after(500, self._refresh_session_ui)

    # ── Session actions ───────────────────────────────────────────────────────

    def _new_session(self):
        if get_state("session_unsaved") or get_state("data_loaded"):
            if not messagebox.askyesno(
                "New Session",
                "Discard current session and start fresh?",
            ):
                return
        from utils.session import reset_session
        reset_session()
        self._reset_all_frame_flags()
        self.frames["📁  Data Loading"].reset_ui()
        self.navigate_to("📁  Data Loading")

    def _open_session(self):
        if get_state("session_unsaved"):
            if not messagebox.askyesno(
                "Unsaved Changes",
                "You have unsaved changes. Open a different session anyway?",
            ):
                return
        path = filedialog.askopenfilename(
            title="Open Session",
            filetypes=[("Surrogate Model Project", "*.smproj"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            from utils.session import load_session
            session_data = load_session(path)
            self._reset_all_frame_flags()
            self._restore_ui_from_session(session_data)
        except RuntimeError as e:
            messagebox.showerror("Load Failed", str(e))

    def _save_session(self):
        # Snapshot widget values into state before saving
        pp = self.frames.get("🔧  Preprocessing")
        if pp and getattr(pp, "built_ui", False):
            set_state("preprocessing_config", pp.get_session_config())
        mb = self.frames.get("🏗  Model Builder")
        if mb and getattr(mb, "built_ui", False):
            set_state("model_builder_config", mb.get_session_config())

        if self.frames.get("🏗  Model Builder") and getattr(
            self.frames["🏗  Model Builder"], "is_running", False
        ):
            messagebox.showwarning(
                "Training in Progress",
                "Please wait for training to finish before saving.",
            )
            return

        path = get_state("session_path")
        if not path:
            name_hint = get_state("session_name") or "session"
            path = filedialog.asksaveasfilename(
                title="Save Session",
                initialfile=name_hint,
                defaultextension=".smproj",
                filetypes=[("Surrogate Model Project", "*.smproj"), ("All files", "*.*")],
            )
            if not path:
                return
            # Use filename (without ext) as session name
            set_state("session_name", os.path.splitext(os.path.basename(path))[0])

        try:
            from utils.session import save_session
            save_session(path)
        except RuntimeError as e:
            messagebox.showerror("Save Failed", str(e))

    # ── Session restore helpers ───────────────────────────────────────────────

    def _reset_all_frame_flags(self):
        """Reset module frame built_ui flags so they rebuild on next on_show.

        Only frames that use the lazy built_ui pattern have their content_frame
        children destroyed.  DataLoadingFrame builds its widgets in __init__ so
        we must NOT destroy them here — use DataLoadingFrame.reset_ui() instead.
        """
        for frame in self.frames.values():
            if hasattr(frame, "_explorer_built"):
                frame._explorer_built = False
            if hasattr(frame, "built_ui"):
                frame.built_ui = False
                # Destroy only lazy-built content (created inside _build_ui)
                if hasattr(frame, "content_frame"):
                    for w in frame.content_frame.winfo_children():
                        w.destroy()

    def _restore_ui_from_session(self, session_data: dict):
        """Orchestrate UI restoration after load_session() has populated AppState."""
        app_state = session_data.get("app_state", {})

        # Restore Data Loading UI
        dl = self.frames["📁  Data Loading"]
        dl.restore_from_session({
            "input_columns":     app_state.get("input_columns", []),
            "output_column":     app_state.get("output_column", []),
        })

        # Restore Preprocessing UI (build then populate widgets)
        pp = self.frames["🔧  Preprocessing"]
        pp_cfg = session_data.get("preprocessing_config")
        if pp_cfg and get_state("data_loaded"):
            pp.restore_from_session(pp_cfg)

        # Model Builder widget values restored lazily in on_show() from model_builder_config state key.

        # Hyperopt apply_btn is re-enabled in hyperopt.on_show() when best_params is in state.

        # Results tab is already marked stale by load_session

        # Navigate to furthest tab that was reached
        if app_state.get("trained"):
            self.navigate_to("📊  Results")
        elif app_state.get("preprocessed"):
            self.navigate_to("🏗  Model Builder")
        elif app_state.get("data_loaded"):
            self.navigate_to("🔧  Preprocessing")
        else:
            self.navigate_to("📁  Data Loading")


if __name__ == "__main__":
    app = App()
    app.mainloop()
