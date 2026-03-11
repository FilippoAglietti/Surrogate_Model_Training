import customtkinter as ctk

from utils.theme import COLORS, FONTS
from utils.state import get_state, set_state

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LeakyReLU, ELU
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import numpy as np
import threading
import queue
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --------------- Helper functions ---------------

def get_keras_activation(name: str):
    if name == "LeakyReLU": return LeakyReLU()
    elif name == "ELU": return ELU()
    mapping = {"ReLU": "relu", "SELU": "selu", "Tanh": "tanh", "Sigmoid": "sigmoid", "GELU": "gelu", "SiLU (Swish)": "swish", "Linear": "linear"}
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

def _get_regularizer(l1_val: float, l2_val: float):
    if l1_val > 0 and l2_val > 0:
        return l1_l2(l1=l1_val, l2=l2_val)
    elif l1_val > 0:
        return l1(l1_val)
    elif l2_val > 0:
        return l2(l2_val)
    return None

# --------------- Keras Callback ---------------

class TkinterUpdateCallback(Callback):
    """Custom Keras callback to send metrics to Tkinter via Queue."""
    def __init__(self, q, epochs, X_val, y_val, frame_ref):
        super().__init__()
        self.q = q
        self.epochs = epochs
        self.X_val = X_val
        self.y_val = y_val
        self.frame_ref = frame_ref
        self.train_losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs=None):
        if self.frame_ref.stop_training_flag:
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss', 0.0)
        train_loss = logs.get('loss', 0.0)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        val_pred = self.model.predict(self.X_val, verbose=0).flatten()
        y_val_flat = self.y_val.flatten()
        ss_res = np.sum((y_val_flat - val_pred) ** 2)
        ss_tot = np.sum((y_val_flat - np.mean(y_val_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        self.q.put({
            "type": "epoch", "epoch": epoch + 1,
            "train_loss": train_loss, "val_loss": val_loss, "r2": r2,
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses)
        })

# --------------- Model builder ---------------

def build_surrogate_model(input_dim, output_dim, num_layers, neurons, act_hidden, act_out, dropout_rate, l1_val, l2_val):
    """Build a Sequential Keras model from compact config."""
    model = Sequential()
    reg = _get_regularizer(l1_val, l2_val)

    for i in range(num_layers):
        kwargs = {"kernel_regularizer": reg} if reg else {}
        if i == 0:
            model.add(Dense(neurons, input_shape=(input_dim,), **kwargs))
        else:
            model.add(Dense(neurons, **kwargs))

        act = get_keras_activation(act_hidden)
        if isinstance(act, str):
            model.add(Activation(act))
        else:
            model.add(act)

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(output_dim))
    out_act = get_keras_activation(act_out)
    if isinstance(out_act, str):
        model.add(Activation(out_act))
    else:
        model.add(out_act)

    return model

# --------------- Frame ---------------

class ModelBuilderFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        self.header = ctk.CTkLabel(self, text="MODEL BUILDER & TRAINING 🏗", font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(20, 10), sticky="w", padx=30)

        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)

        self.built_ui = False

        self.activation_options = ["ReLU", "LeakyReLU", "ELU", "SELU", "Tanh", "Sigmoid", "GELU", "SiLU (Swish)", "Linear"]
        self.loss_options = ["MeanSquaredError", "MeanAbsoluteError", "Huber", "LogCosh"]
        self.optim_options = ["Adam", "AdamW", "SGD", "RMSprop"]

        self.is_running = False
        self.stop_training_flag = False
        self.q = queue.Queue()

    def on_show(self):
        if not get_state("preprocessed"):
            self._show_blocked("Preprocess data first.\n← Go to 'Preprocessing'")
            return
        if not self.built_ui:
            self._build_ui()
            self.built_ui = True
            
        applied = get_state("applied_hpo_params")
        if applied:
            self._apply_hpo_params(applied)
            set_state("applied_hpo_params", None)

    def _apply_hpo_params(self, p):
        self.num_layers_entry.delete(0, "end")
        self.num_layers_entry.insert(0, str(p.get("n_layers", 1)))
        self.neurons_entry.delete(0, "end")
        self.neurons_entry.insert(0, str(p.get("neurons", 64)))
        
        self.act_var.set(p.get("activation", "ReLU"))
        self.out_act_var.set(p.get("out_act", "Linear"))
        
        self.dropout_entry.delete(0, "end")
        self.dropout_entry.insert(0, str(p.get("dropout", 0.0)))
        self.l1_entry.delete(0, "end")
        self.l1_entry.insert(0, str(p.get("l1", 0.0)))
        self.l2_entry.delete(0, "end")
        self.l2_entry.insert(0, str(p.get("l2", 0.0)))
        
        self.loss_var.set(p.get("loss", "MeanSquaredError"))
        self.optim_var.set(p.get("optimizer", "Adam"))
        
        self.lr_entry.delete(0, "end")
        self.lr_entry.insert(0, str(p.get("lr", 0.001)))
        self.bs_entry.delete(0, "end")
        self.bs_entry.insert(0, str(p.get("batch_size", 64)))
        self.ep_entry.delete(0, "end")
        self.ep_entry.insert(0, str(p.get("epochs", 100)))
        
        self.es_pat_e.delete(0, "end")
        self.es_pat_e.insert(0, str(p.get("es_pat", 20)))
        self.es_del_e.delete(0, "end")
        self.es_del_e.insert(0, str(p.get("es_del", 0.00001)))
        
        self.lr_factor_e.delete(0, "end")
        self.lr_factor_e.insert(0, str(p.get("rlr_factor", 0.5)))
        self.lr_pat_e.delete(0, "end")
        self.lr_pat_e.insert(0, str(p.get("rlr_pat", 10)))
        
        self.train_lbl.configure(text="✨ Applied Best HPO Params!", text_color=COLORS["cyan"])

    def _show_blocked(self, message):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.built_ui = False
        lbl = ctk.CTkLabel(self.content_frame, text=f"[ BLOCKED ]\n{message}", font=FONTS["header"], text_color=COLORS["red"])
        lbl.grid(row=0, column=0, pady=50, padx=20)

    # ===================== UI =====================

    def _build_ui(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # ── Architecture Card ──
        arch_card = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        arch_card.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 5))
        arch_card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(arch_card, text="ARCHITECTURE", font=FONTS["header"], text_color=COLORS["magenta"]).grid(row=0, column=0, columnspan=4, padx=20, pady=(15, 10), sticky="w")

        # Row 1
        ctk.CTkLabel(arch_card, text="Num hidden layers").grid(row=1, column=0, padx=20, sticky="w")
        self.num_layers_entry = ctk.CTkEntry(arch_card)
        self.num_layers_entry.insert(0, "3")
        self.num_layers_entry.grid(row=2, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(arch_card, text="Neurons per hidden layer").grid(row=1, column=1, padx=10, sticky="w")
        self.neurons_entry = ctk.CTkEntry(arch_card)
        self.neurons_entry.insert(0, "64")
        self.neurons_entry.grid(row=2, column=1, padx=10, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(arch_card, text="Hidden Activation").grid(row=1, column=2, padx=10, sticky="w")
        self.act_var = ctk.StringVar(value="ReLU")
        ctk.CTkComboBox(arch_card, values=self.activation_options, variable=self.act_var).grid(row=2, column=2, padx=10, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(arch_card, text="Output Activation").grid(row=1, column=3, padx=20, sticky="w")
        self.out_act_var = ctk.StringVar(value="Linear")
        ctk.CTkComboBox(arch_card, values=self.activation_options, variable=self.out_act_var).grid(row=2, column=3, padx=20, pady=(0, 15), sticky="ew")

        # Row 2
        ctk.CTkLabel(arch_card, text="Dropout").grid(row=3, column=0, padx=20, sticky="w")
        self.dropout_entry = ctk.CTkEntry(arch_card)
        self.dropout_entry.insert(0, "0.0")
        self.dropout_entry.grid(row=4, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(arch_card, text="L1 Regularization").grid(row=3, column=1, padx=10, sticky="w")
        self.l1_entry = ctk.CTkEntry(arch_card)
        self.l1_entry.insert(0, "0.0")
        self.l1_entry.grid(row=4, column=1, padx=10, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(arch_card, text="L2 Regularization").grid(row=3, column=2, padx=10, sticky="w")
        self.l2_entry = ctk.CTkEntry(arch_card)
        self.l2_entry.insert(0, "0.0")
        self.l2_entry.grid(row=4, column=2, padx=10, pady=(0, 15), sticky="ew")

        # ── Compile & Callbacks Card ──
        compile_card = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        compile_card.grid(row=1, column=0, sticky="ew", padx=20, pady=5)
        compile_card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(compile_card, text="COMPILE & HYPERPARAMS", font=FONTS["header"], text_color=COLORS["magenta"]).grid(row=0, column=0, columnspan=4, padx=20, pady=(15, 10), sticky="w")

        # Compile Row 1
        ctk.CTkLabel(compile_card, text="Loss Function").grid(row=1, column=0, padx=20, sticky="w")
        self.loss_var = ctk.StringVar(value="MeanSquaredError")
        ctk.CTkComboBox(compile_card, values=self.loss_options, variable=self.loss_var).grid(row=2, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(compile_card, text="Optimizer").grid(row=1, column=1, padx=10, sticky="w")
        self.optim_var = ctk.StringVar(value="Adam")
        ctk.CTkComboBox(compile_card, values=self.optim_options, variable=self.optim_var).grid(row=2, column=1, padx=10, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(compile_card, text="Learning Rate").grid(row=1, column=2, padx=10, sticky="w")
        self.lr_entry = ctk.CTkEntry(compile_card)
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=2, column=2, padx=10, pady=(0, 15), sticky="ew")

        # Compile Row 2
        ctk.CTkLabel(compile_card, text="Batch Size").grid(row=3, column=0, padx=20, sticky="w")
        self.bs_entry = ctk.CTkEntry(compile_card)
        self.bs_entry.insert(0, "64")
        self.bs_entry.grid(row=4, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(compile_card, text="Epochs").grid(row=3, column=1, padx=10, sticky="w")
        self.ep_entry = ctk.CTkEntry(compile_card)
        self.ep_entry.insert(0, "200")
        self.ep_entry.grid(row=4, column=1, padx=10, pady=(0, 15), sticky="ew")

        # Separator inside Compile Card
        sep = ctk.CTkFrame(compile_card, height=2, fg_color=COLORS["border"])
        sep.grid(row=5, column=0, columnspan=4, sticky="ew", padx=20, pady=10)
        
        ctk.CTkLabel(compile_card, text="CALLBACKS", font=FONTS["header"], text_color=COLORS["cyan"]).grid(row=6, column=0, columnspan=4, padx=20, pady=(5, 10), sticky="w")

        # Callbacks Row 1
        self.use_es = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(compile_card, text="Early Stopping", variable=self.use_es).grid(row=7, column=0, padx=20, pady=10, sticky="w")

        ctk.CTkLabel(compile_card, text="ES Patience").grid(row=7, column=1, padx=10, sticky="w")
        self.es_pat_e = ctk.CTkEntry(compile_card)
        self.es_pat_e.insert(0, "20")
        self.es_pat_e.grid(row=7, column=1, padx=(100, 10), pady=10, sticky="w")

        ctk.CTkLabel(compile_card, text="ES Min Δ").grid(row=7, column=2, padx=10, sticky="w")
        self.es_del_e = ctk.CTkEntry(compile_card)
        self.es_del_e.insert(0, "0.00001")
        self.es_del_e.grid(row=7, column=2, padx=(80, 10), pady=10, sticky="w")

        # Callbacks Row 2
        self.use_reduce_lr = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(compile_card, text="Reduce LR", variable=self.use_reduce_lr).grid(row=8, column=0, padx=20, pady=(0, 20), sticky="w")

        ctk.CTkLabel(compile_card, text="LR Factor").grid(row=8, column=1, padx=10, sticky="w")
        self.lr_factor_e = ctk.CTkEntry(compile_card)
        self.lr_factor_e.insert(0, "0.5")
        self.lr_factor_e.grid(row=8, column=1, padx=(100, 10), pady=(0, 20), sticky="w")

        ctk.CTkLabel(compile_card, text="LR Patience").grid(row=8, column=2, padx=10, sticky="w")
        self.lr_pat_e = ctk.CTkEntry(compile_card)
        self.lr_pat_e.insert(0, "10")
        self.lr_pat_e.grid(row=8, column=2, padx=(100, 10), pady=(0, 20), sticky="w")

        ctk.CTkLabel(compile_card, text="Min LR").grid(row=8, column=3, padx=10, sticky="w")
        self.lr_min_e = ctk.CTkEntry(compile_card)
        self.lr_min_e.insert(0, "1e-6")
        self.lr_min_e.grid(row=8, column=3, padx=(70, 20), pady=(0, 20), sticky="w")

        # ── Button Row ──
        ctrl_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        ctrl_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=20)

        self.build_btn = ctk.CTkButton(ctrl_frame, text="⚡ BUILD", height=40, font=("Helvetica", 14, "bold"), fg_color=COLORS["cyan"], hover_color="#00ACC1", text_color="#000000", command=self._build_model)
        self.build_btn.pack(side="left")

        self.train_btn = ctk.CTkButton(ctrl_frame, text="🚀 START TRAINING", height=40, font=("Helvetica", 14, "bold"), state="disabled", command=self.start_training)
        self.train_btn.pack(side="left", padx=15)

        self.stop_btn = ctk.CTkButton(
            ctrl_frame, text="🛑 STOP", height=40, width=80, font=("Helvetica", 14, "bold"),
            fg_color=COLORS["red"], hover_color="#CC0000",
            state="disabled", command=self.stop_training
        )
        self.stop_btn.pack(side="left")

        self.train_lbl = ctk.CTkLabel(ctrl_frame, text="Model not built.", font=FONTS["body"], text_color=COLORS["text_dim"])
        self.train_lbl.pack(side="left", padx=25)

        self.pb = ctk.CTkProgressBar(ctrl_frame, width=250)
        self.pb.set(0)
        self.pb.pack(side="right")

        # ── Matplotlib Plot ──
        self.plot_frame = ctk.CTkFrame(self.content_frame, fg_color="#000", height=280)
        self.plot_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=5)
        self.plot_frame.grid_propagate(False)

        # ── Log Box ──
        self.log_box = ctk.CTkTextbox(self.content_frame, height=180, font=FONTS["code"], fg_color="#0D1117", text_color=COLORS["text"])
        self.log_box.grid(row=4, column=0, sticky="ew", padx=20, pady=(10, 30))
        self.log_box.configure(state="disabled")

    # ===================== Build =====================

    def _build_model(self):
        try:
            num_layers = int(self.num_layers_entry.get())
            neurons = int(self.neurons_entry.get())
            if num_layers < 1 or neurons < 1:
                raise ValueError("Must have at least 1 layer with 1 neuron.")
            
            lr = float(self.lr_entry.get())
            bs = int(self.bs_entry.get())
            ep = int(self.ep_entry.get())
            dropout = float(self.dropout_entry.get())
            l1_val = float(self.l1_entry.get())
            l2_val = float(self.l2_entry.get())
            
        except ValueError as e:
            self.train_lbl.configure(text=f"Error: {e}", text_color=COLORS["red"])
            return

        input_dim = get_state("X_train").shape[1]
        act_hidden = self.act_var.get()
        act_out = self.out_act_var.get()

        # Store layers_config for downstream compatibility (so results.py doesn't crash if it tries to read the dict)
        layers_config = [{"units": neurons, "activation": act_hidden, "dropout": dropout} for _ in range(num_layers)]
        set_state("layers_config", layers_config)

        model_config = {
            "num_layers": num_layers,
            "neurons": neurons,
            "act_hidden": act_hidden,
            "act_out": act_out,
            "dropout": dropout,
            "l1": l1_val,
            "l2": l2_val,
            "layers": layers_config,
            "loss": self.loss_var.get(),
            "optimizer": self.optim_var.get(),
            "lr": lr,
            "batch_size": bs,
            "epochs": ep,
            "input_dim": input_dim,
        }

        set_state("model_ready", True)
        set_state("model_config", model_config)
        set_state("trained", False)

        temp_model = build_surrogate_model(input_dim, 1, num_layers, neurons, act_hidden, act_out, dropout, l1_val, l2_val)
        set_state("model_params_count", temp_model.count_params())

        self.train_lbl.configure(text=f"✓ Built: {num_layers}x{neurons} [{temp_model.count_params()} params]", text_color=COLORS["green"])
        self.train_btn.configure(state="normal")

    # ===================== Training =====================

    def _log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.yview("end")
        self.log_box.configure(state="disabled")

    def stop_training(self):
        if self.is_running:
            self.stop_training_flag = True
            self.stop_btn.configure(state="disabled", text="Stopping...")

    def start_training(self):
        if self.is_running:
            return
        self.is_running = True
        self.stop_training_flag = False
        self.train_btn.configure(state="disabled", text="TRAINING...")
        self.stop_btn.configure(state="normal", text="🛑 STOP")
        self.build_btn.configure(state="disabled")
        self.log_box.configure(state="normal")
        self.log_box.delete("0.0", "end")
        self.log_box.configure(state="disabled")

        cfg = get_state("model_config")
        X_train, y_train = get_state("X_train"), get_state("y_train")
        X_val, y_val = get_state("X_val"), get_state("y_val")

        es_pat = int(self.es_pat_e.get())
        es_del = float(self.es_del_e.get())
        use_es = self.use_es.get()

        rlr_factor = float(self.lr_factor_e.get())
        rlr_pat = int(self.lr_pat_e.get())
        rlr_min = float(self.lr_min_e.get())
        use_rlr = self.use_reduce_lr.get()

        t = threading.Thread(
            target=self._run_training_thread,
            args=(cfg, X_train, y_train, X_val, y_val, use_es, es_pat, es_del, use_rlr, rlr_factor, rlr_pat, rlr_min)
        )
        t.start()

        self._init_plot()
        self.after(100, self.process_queue)

    def _init_plot(self):
        for w in self.plot_frame.winfo_children():
            w.destroy()
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 3), dpi=100)
        self.fig.patch.set_facecolor(COLORS["bg_card"])
        self.ax.set_facecolor(COLORS["bg_card"])
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.line_train, = self.ax.plot([], [], color=COLORS['cyan'], label="Train Loss")
        self.line_val, = self.ax.plot([], [], color=COLORS['orange'], label="Val Loss")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _update_plot(self, t_losses, v_losses):
        x = range(1, len(t_losses) + 1)
        self.line_train.set_data(x, t_losses)
        self.line_val.set_data(x, v_losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def process_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if isinstance(msg, dict) and msg["type"] == "epoch":
                    ep, tl, vl, r2 = msg["epoch"], msg["train_loss"], msg["val_loss"], msg["r2"]
                    self.pb.set(ep / get_state("model_config")["epochs"])
                    self.train_lbl.configure(text=f"Epoch {ep} | Train: {tl:.6f} | Val: {vl:.6f} | R²: {r2:.4f}")
                    self._log(f"[{ep:>4}] train={tl:.6f} val={vl:.6f} R²={r2:.4f}")
                    if ep % 3 == 0 or ep == get_state("model_config")["epochs"]:
                        self._update_plot(msg["train_losses"], msg["val_losses"])
                elif isinstance(msg, str):
                    if msg == "DONE":
                        self.is_running = False
                        self.train_btn.configure(state="normal", text="🚀 START TRAINING")
                        self.stop_btn.configure(state="disabled", text="🛑 STOP")
                        self.build_btn.configure(state="normal")
                        if self.stop_training_flag:
                            self._log("\n[ ABORTED ] Training stopped by user.")
                            self.train_lbl.configure(text="⚠ Training stopped.", text_color=COLORS["orange"])
                        else:
                            self._log("\n[ OK ] Training Complete.")
                            self.train_lbl.configure(text="✓ Training Complete.", text_color=COLORS["green"])
                        break
                    else:
                        self._log(msg)
        except queue.Empty:
            if self.is_running:
                self.after(50, self.process_queue)

    def _run_training_thread(self, cfg, X_train, y_train, X_val, y_val, use_es, es_pat, es_del, use_rlr, rlr_factor, rlr_pat, rlr_min):
        try:
            model = build_surrogate_model(
                cfg["input_dim"], 1, cfg["num_layers"], cfg["neurons"],
                cfg["act_hidden"], cfg["act_out"], cfg["dropout"], cfg["l1"], cfg["l2"]
            )
            criterion = get_keras_loss(cfg["loss"])
            optimizer = get_keras_optimizer(cfg["optimizer"], cfg["lr"])
            model.compile(optimizer=optimizer, loss=criterion)

            epochs = cfg["epochs"]
            st_callback = TkinterUpdateCallback(self.q, epochs, X_val, y_val, self)
            callbacks = [st_callback]

            if use_es:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=es_pat, min_delta=es_del, restore_best_weights=True, verbose=0))
            if use_rlr:
                callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=rlr_factor, patience=rlr_pat, min_lr=rlr_min, verbose=0))

            self.q.put("Starting Training...")
            start_time = time.time()

            model.fit(
                X_train, y_train, validation_data=(X_val, y_val),
                epochs=epochs, batch_size=cfg["batch_size"],
                callbacks=callbacks, verbose=0
            )

            elapsed = time.time() - start_time
            t_losses = st_callback.train_losses
            v_losses = st_callback.val_losses

            if len(t_losses) > 0:
                val_pred = model.predict(X_val, verbose=0).flatten()
                y_val_flat = y_val.flatten()
                ss_res = np.sum((y_val_flat - val_pred) ** 2)
                ss_tot = np.sum((y_val_flat - np.mean(y_val_flat)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                set_state("model", model)
                set_state("trained", True)
                set_state("train_losses", t_losses)
                set_state("val_losses", v_losses)
                set_state("training_metrics", {
                    "best_val_loss": min(v_losses) if v_losses else 0,
                    "final_train_loss": t_losses[-1] if t_losses else 0,
                    "r2": r2, "epochs_run": len(t_losses), "elapsed_seconds": elapsed
                })
                self._update_plot(t_losses, v_losses)

                summary = f"\n┌──────────────────────────────────────────┐\n│  TRAINING SUMMARY                        │\n├──────────────────────────────────────────┤\n│  Epochs run   : {len(t_losses):<25}│\n│  Best val loss: {min(v_losses):<25.6f}│\n│  Final R²     : {r2:<25.4f}│\n│  Time (s)     : {elapsed:<25.1f}│\n└──────────────────────────────────────────┘"
                self.q.put(summary)
            else:
                self.q.put("\nTraining aborted before first epoch completed.")

        except Exception as e:
            self.q.put(f"ERROR: {str(e)}")

        self.q.put("DONE")
