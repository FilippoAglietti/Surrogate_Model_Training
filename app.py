import customtkinter as ctk

from utils.theme import setup_theme, COLORS, FONTS, ASCII_BANNER
from utils.state import init_all_defaults, get_state, set_state

# Import all our module frames
from modules.data_loading import DataLoadingFrame
from modules.preprocessing import PreprocessingFrame
from modules.model_builder import ModelBuilderFrame
from modules.hyperopt import HyperoptFrame
from modules.training import TrainingFrame
from modules.results import ResultsFrame


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize app settings
        self.title("Surrogate Builder v2.0")
        self.geometry("1400x900")
        self.minsize(1000, 700)
        
        # Setup Theme & State
        setup_theme()
        init_all_defaults()
        
        # Configure grid for main window (1 row, 2 columns)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=COLORS["bg_card"])
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)  # spacer
        
        # Sidebar Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="⚡\nSURROGATE\nBUILDER", 
            font=FONTS["header"],
            text_color=COLORS["orange"],
            justify="center"
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(40, 40))
        
        # Navigation Buttons
        self.nav_buttons = {}
        self.frames = {}
        
        pages = [
            ("📁  Data Loading", DataLoadingFrame),
            ("🔧  Preprocessing", PreprocessingFrame),
            ("🏗  Model Builder", ModelBuilderFrame),
            ("🔍  Hyperopt", HyperoptFrame),
            ("🚀  Training", TrainingFrame),
            ("📊  Results", ResultsFrame)
        ]
        
        for i, (name, FrameClass) in enumerate(pages):
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
            btn.grid(row=i+1, column=0, padx=20, pady=5, sticky="ew")
            self.nav_buttons[name] = btn
            
            # Instantiate frame (hidden initially)
            frame = FrameClass(self, corner_radius=0, fg_color=COLORS["bg"])
            self.frames[name] = frame

        # Status area at bottom of sidebar
        self.status_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="v2.0 · CustomTkinter\nBuilt by Antigravity",
            font=("Consolas", 10),
            text_color=COLORS["text_dim"]
        )
        self.status_label.grid(row=9, column=0, padx=20, pady=20)

        # Start on default page
        default_page = get_state("nav_page")
        self.navigate_to(default_page)
        
    def navigate_to(self, page_name):
        """Switches the visible frame and updates button styling."""
        set_state("nav_page", page_name)
        
        # Update button colors
        for name, btn in self.nav_buttons.items():
            if name == page_name:
                btn.configure(fg_color=COLORS["border"], text_color=COLORS["cyan"])
            else:
                btn.configure(fg_color="transparent", text_color=COLORS["text"])
                
        # Hide all frames
        for frame in self.frames.values():
            frame.grid_forget()
            
        # Show selected frame
        active_frame = self.frames[page_name]
        active_frame.grid(row=0, column=1, sticky="nsew")
        
        # Tell frame it was opened so it can refresh data if needed
        if hasattr(active_frame, "on_show"):
            active_frame.on_show()


if __name__ == "__main__":
    app = App()
    app.mainloop()
