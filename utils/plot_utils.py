"""
Shared plot utilities.
"""
from tkinter import filedialog
import customtkinter as ctk
from utils.theme import COLORS


def add_save_button(parent, canvas, default_name="plot.png"):
    """Attach a compact 'Save Image' button below a FigureCanvasTkAgg canvas.

    Parameters
    ----------
    parent      : tkinter / CTk widget that the canvas was packed into.
    canvas      : FigureCanvasTkAgg instance.
    default_name: suggested filename shown in the save dialog.
    """
    def _save():
        fn = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Image",      "*.png"),
                ("PDF Document",   "*.pdf"),
                ("SVG Vector",     "*.svg"),
            ],
            initialfile=default_name,
        )
        if fn:
            canvas.figure.savefig(
                fn, dpi=150, bbox_inches="tight",
                facecolor=canvas.figure.get_facecolor(),
            )

    btn = ctk.CTkButton(
        parent,
        text="💾 Save",
        width=90, height=24,
        font=("Helvetica", 10),
        fg_color=COLORS["bg_card_hover"],
        hover_color="#3A3A5E",
        text_color=COLORS["text"],
        command=_save,
    )
    btn.pack(anchor="e", padx=10, pady=(0, 4))
    return btn
