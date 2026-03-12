import customtkinter as ctk
from modules.model_builder import ModelBuilderFrame
from utils.theme import setup_theme
from utils.state import init_all_defaults, set_state

app = ctk.CTk()
app.geometry("1400x900")
setup_theme()
init_all_defaults()
set_state("data_loaded", True)
set_state("preprocessed", True)
import pandas as pd
import numpy as np
set_state("X_train", np.zeros((10, 5)))
set_state("y_train", np.zeros((10, 1)))

frame = ModelBuilderFrame(app)
frame.pack(fill="both", expand=True)
frame.on_show()
app.mainloop()
