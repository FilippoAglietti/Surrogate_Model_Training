import customtkinter as ctk

from utils.theme import COLORS, FONTS
from utils.state import get_state, set_state

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LeakyReLU, ELU

def get_keras_activation(name: str):
    if name == "LeakyReLU": return LeakyReLU()
    elif name == "ELU": return ELU()
    mapping = {"ReLU": "relu", "SELU": "selu", "Tanh": "tanh", "Sigmoid": "sigmoid", "GELU": "gelu", "SiLU (Swish)": "swish"}
    return mapping.get(name, "relu")

def get_keras_loss(name: str):
    losses = {
        "MeanSquaredError": tf.keras.losses.MeanSquaredError(),
        "MeanAbsoluteError": tf.keras.losses.MeanAbsoluteError(),
        "Huber": tf.keras.losses.Huber(),
        "LogCosh": tf.keras.losses.LogCosh(),
    }
    return losses.get(name, tf.keras.losses.MeanSquaredError())

def get_keras_optimizer(name: str, lr: float):
    optimizers = {
        "Adam": tf.keras.optimizers.Adam(learning_rate=lr),
        "AdamW": tf.keras.optimizers.AdamW(learning_rate=lr) if hasattr(tf.keras.optimizers, 'AdamW') else tf.keras.optimizers.Adam(learning_rate=lr),
        "SGD": tf.keras.optimizers.SGD(learning_rate=lr),
        "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=lr),
    }
    return optimizers.get(name, tf.keras.optimizers.Adam(learning_rate=lr))

def build_surrogate_model(input_dim: int, output_dim: int, layers_config: list) -> Sequential:
    model = Sequential()
    first_layer_added = False
    
    for layer in layers_config:
        if not first_layer_added:
            model.add(Dense(layer["units"], input_shape=(input_dim,)))
            first_layer_added = True
        else:
            model.add(Dense(layer["units"]))
            
        act = get_keras_activation(layer["activation"])
        if isinstance(act, str):
            model.add(Activation(act))
        else:
            model.add(act)
            
        if layer["dropout"] > 0:
            model.add(Dropout(layer["dropout"]))
            
    model.add(Dense(output_dim))
    return model


class ModelBuilderFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        # Header
        self.header = ctk.CTkLabel(self, text="MODEL BUILDER 🏗", font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(30, 20), sticky="w", padx=30)

        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)

        self.layer_frames = []
        self.layers_config = get_state("layers_config")
        
        self.built_ui = False
        
        self.activation_options = ["ReLU", "LeakyReLU", "ELU", "SELU", "Tanh", "Sigmoid", "GELU", "SiLU (Swish)"]
        self.loss_options = ["MeanSquaredError", "MeanAbsoluteError", "Huber", "LogCosh"]
        self.optim_options = ["Adam", "AdamW", "SGD", "RMSprop"]

    def on_show(self):
        if not get_state("preprocessed"):
            self._show_blocked("Preprocess data first.\n← Go to 'Preprocessing'")
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
            
        # Top buttons
        btn_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        btn_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20), padx=20)
        
        ctk.CTkButton(btn_frame, text="＋ Add Layer", command=self._add_layer).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="－ Remove Layer", command=self._remove_layer).pack(side="left", padx=10)
        
        # Architecture Frame
        self.arch_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.arch_frame.grid(row=1, column=0, sticky="ew", padx=20)
        self.arch_frame.grid_columnconfigure(0, weight=1)
        
        self._render_layers()
        
        # Compile Config
        compile_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        compile_frame.grid(row=2, column=0, sticky="ew", pady=20, padx=20)
        compile_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        ctk.CTkLabel(compile_frame, text="Loss Function").grid(row=0, column=0, pady=10)
        self.loss_var = ctk.StringVar(value="MeanSquaredError")
        ctk.CTkComboBox(compile_frame, values=self.loss_options, variable=self.loss_var).grid(row=1, column=0, padx=20, pady=(0, 20))
        
        ctk.CTkLabel(compile_frame, text="Optimizer").grid(row=0, column=1)
        self.optim_var = ctk.StringVar(value="Adam")
        ctk.CTkComboBox(compile_frame, values=self.optim_options, variable=self.optim_var).grid(row=1, column=1, padx=20, pady=(0, 20))
        
        ctk.CTkLabel(compile_frame, text="Learning Rate").grid(row=0, column=2)
        self.lr_entry = ctk.CTkEntry(compile_frame)
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=1, column=2, padx=20, pady=(0, 20))
        
        # Epochs / Batch
        ctk.CTkLabel(compile_frame, text="Batch Size").grid(row=2, column=0)
        self.bs_entry = ctk.CTkEntry(compile_frame)
        self.bs_entry.insert(0, "64")
        self.bs_entry.grid(row=3, column=0, padx=20, pady=(0, 20))
        
        ctk.CTkLabel(compile_frame, text="Epochs").grid(row=2, column=1)
        self.ep_entry = ctk.CTkEntry(compile_frame)
        self.ep_entry.insert(0, "200")
        self.ep_entry.grid(row=3, column=1, padx=20, pady=(0, 20))

        # Build Actions
        action_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        action_frame.grid(row=3, column=0, pady=20)
        
        self.build_btn = ctk.CTkButton(
            action_frame, text="⚡ BUILD MODEL", height=40,
            command=self._build_model
        )
        self.build_btn.grid(row=0, column=0, padx=10)
        
        self.status_lbl = ctk.CTkLabel(action_frame, text="", text_color=COLORS["green"])
        self.status_lbl.grid(row=0, column=1, padx=20)

    def _render_layers(self):
        for widget in self.arch_frame.winfo_children():
            widget.destroy()
            
        self.layer_vars = []
        
        for i, l_cfg in enumerate(self.layers_config):
            f = ctk.CTkFrame(self.arch_frame, fg_color=COLORS["bg_card"], border_width=1, border_color=COLORS["border"])
            f.pack(fill="x", pady=5)
            f.grid_columnconfigure((0, 1, 2), weight=1)
            
            ctk.CTkLabel(f, text=f"Layer {i+1}", font=FONTS["header"], text_color=COLORS["magenta"]).grid(row=0, column=0, padx=10, pady=5, sticky="w")
            
            # Variables to track
            u_var = ctk.StringVar(value=str(l_cfg["units"]))
            a_var = ctk.StringVar(value=l_cfg["activation"])
            d_var = ctk.DoubleVar(value=l_cfg["dropout"])
            
            self.layer_vars.append({"units": u_var, "activation": a_var, "dropout": d_var})
            
            # Units
            uf = ctk.CTkFrame(f, fg_color="transparent")
            uf.grid(row=1, column=0, padx=10, pady=(0,10), sticky="ew")
            ctk.CTkLabel(uf, text="Neurons: ").pack(side="left")
            ctk.CTkEntry(uf, textvariable=u_var, width=80).pack(side="left")
            
            # Activation
            af = ctk.CTkFrame(f, fg_color="transparent")
            af.grid(row=1, column=1, padx=10, pady=(0,10), sticky="ew")
            ctk.CTkLabel(af, text="Act: ").pack(side="left")
            ctk.CTkComboBox(af, values=self.activation_options, variable=a_var, width=120).pack(side="left")
            
            # Dropout
            df = ctk.CTkFrame(f, fg_color="transparent")
            df.grid(row=1, column=2, padx=10, pady=(0,10), sticky="ew")
            dlbl = ctk.CTkLabel(df, text=f"Drop: {d_var.get():.2f}")
            dlbl.pack(side="left", padx=5)
            
            def update_drop_lbl(val, lbl=dlbl):
                lbl.configure(text=f"Drop: {float(val):.2f}")
                
            ctk.CTkSlider(df, variable=d_var, from_=0.0, to=0.8, number_of_steps=16, command=update_drop_lbl).pack(side="left", expand=True, fill="x")

    def _sync_layers_config(self):
        new_config = []
        for lv in self.layer_vars:
            try:
                u = int(lv["units"].get())
            except:
                u = 64
            new_config.append({
                "units": u,
                "activation": lv["activation"].get(),
                "dropout": float(lv["dropout"].get())
            })
        self.layers_config = new_config
        set_state("layers_config", new_config)

    def _add_layer(self):
        self._sync_layers_config()
        self.layers_config.append({"units": 64, "activation": "ReLU", "dropout": 0.0})
        self._render_layers()

    def _remove_layer(self):
        self._sync_layers_config()
        if len(self.layers_config) > 1:
            self.layers_config.pop()
            self._render_layers()

    def _build_model(self):
        self._sync_layers_config()
        
        try:
            lr = float(self.lr_entry.get())
            bs = int(self.bs_entry.get())
            ep = int(self.ep_entry.get())
        except ValueError:
            self.status_lbl.configure(text="Invalid numeric values for LR/Batch/Epochs", text_color=COLORS["red"])
            return

        input_dim = get_state("X_train").shape[1]
        
        model_config = {
            "layers": self.layers_config,
            "loss": self.loss_var.get(),
            "optimizer": self.optim_var.get(),
            "lr": lr,
            "batch_size": bs,
            "epochs": ep,
            "input_dim": input_dim,
        }

        # Tell downstream we have a config ready
        set_state("model_ready", True)
        set_state("model_config", model_config)
        set_state("trained", False)
        
        # Temp build to count params
        temp_model = build_surrogate_model(input_dim, 1, self.layers_config)
        set_state("model_params_count", temp_model.count_params())
        
        self.status_lbl.configure(text=f"✓ Built! {temp_model.count_params()} total parameters.", text_color=COLORS["green"])
