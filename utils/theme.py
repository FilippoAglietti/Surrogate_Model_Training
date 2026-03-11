"""
Theme definitions for CustomTkinter.
Defines colors, fonts, and the ASCII banner.
"""
import customtkinter as ctk

# ── Color Palette (Dark Neon) ─────────────────────────
COLORS = {
    "bg": "#0B0E14",
    "bg_card": "#121822",
    "bg_card_hover": "#1A2230",
    "text": "#E2E8F0",
    "text_dim": "#64748B",
    
    # Neon Accents
    "primary": "#00FF41",      # Matrix Green
    "primary_dark": "#00CC33",
    "green": "#00CC33",        # Alias
    "cyan": "#00E5FF",         # Electric Blue
    "magenta": "#FF00FF",      # Cyberpunk Pink
    "orange": "#FF6E40",       # Coral/Orange
    "yellow": "#FFD600",
    "red": "#FF3B30",
    
    # UI Elements
    "border": "#1E2A3B",
    "input_bg": "#0D1117"
}

# ── Fonts ────────────────────────────────────────────────
FONTS = {
    "title": ("Courier New", 28, "bold"),
    "header": ("Courier New", 18, "bold"),
    "body": ("Consolas", 12),
    "code": ("Consolas", 11)
}

# ── ASCII Art ────────────────────────────────────────────
ASCII_BANNER = """
   _____ _    _ _____  _____   ____   _____       _______ ______ 
  / ____| |  | |  __ \|  __ \ / __ \ / ____|   /\|__   __|  ____|
 | (___ | |  | | |__) | |__) | |  | | |  __   /  \  | |  | |__   
  \___ \| |  | |  _  /|  _  /| |  | | | |_ | / /\ \ | |  |  __|  
  ____) | |__| | | \ \| | \ \| |__| | |__| |/ ____ \| |  | |____ 
 |_____/ \____/|_|  \_\_|  \_\\____/ \_____/_/    \_\_|  |______|
                             ⚡ B U I L D E R ⚡
"""

def setup_theme():
    """Configures the global CustomTkinter theme."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")  # Built-in green theme as a fallback
