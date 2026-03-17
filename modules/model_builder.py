import customtkinter as ctk
import threading
import queue
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LeakyReLU, ELU
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

from models import ALGORITHM_REGISTRY, ALGO_NAMES
from models.nn_model import NeuralNetworkSurrogate
from utils.theme import COLORS, FONTS
from utils.state import get_state, set_state
from utils.plot_utils import add_save_button


# ─────────────────────── Keras helpers ───────────────────────────────────────

def get_keras_activation(name: str):
    if name == "LeakyReLU": return LeakyReLU()
    elif name == "ELU": return ELU()
    mapping = {"ReLU": "relu", "SELU": "selu", "Tanh": "tanh", "Sigmoid": "sigmoid",
               "GELU": "gelu", "SiLU (Swish)": "swish", "Linear": "linear"}
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
    if l1_val > 0 and l2_val > 0: return l1_l2(l1=l1_val, l2=l2_val)
    elif l1_val > 0: return l1(l1_val)
    elif l2_val > 0: return l2(l2_val)
    return None

def build_surrogate_model(input_dim, output_dim, num_layers, neurons, act_hidden, act_out, dropout_rate, l1_val, l2_val):
    model = Sequential()
    reg = _get_regularizer(l1_val, l2_val)
    for i in range(num_layers):
        kwargs = {"kernel_regularizer": reg} if reg else {}
        if i == 0:
            model.add(Dense(neurons, input_shape=(input_dim,), **kwargs))
        else:
            model.add(Dense(neurons, **kwargs))
        act = get_keras_activation(act_hidden)
        if isinstance(act, str): model.add(Activation(act))
        else: model.add(act)
        if dropout_rate > 0: model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim))
    out_act = get_keras_activation(act_out)
    if isinstance(out_act, str): model.add(Activation(out_act))
    else: model.add(out_act)
    return model


# ─────────────────────── Keras callback ──────────────────────────────────────

class TkinterUpdateCallback(Callback):
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
        val_pred = self.model.predict(self.X_val, verbose=0)
        r2 = r2_score(self.y_val, val_pred)
        self.q.put({
            "type": "epoch", "epoch": epoch + 1,
            "train_loss": train_loss, "val_loss": val_loss, "r2": r2,
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses)
        })


# ─────────────────────── Frame ───────────────────────────────────────────────

class ModelBuilderFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkLabel(self, text="MODEL BUILDER & TRAINING 🏗",
                                   font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(20, 10), sticky="w", padx=30)

        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)

        self.built_ui = False
        self.selected_algo = "Neural Network"
        self._pending_surrogate = None

        self.activation_options = ["ReLU", "LeakyReLU", "ELU", "SELU", "Tanh",
                                   "Sigmoid", "GELU", "SiLU (Swish)", "Linear"]
        self.loss_options = ["MeanSquaredError", "MeanAbsoluteError", "Huber", "LogCosh"]
        self.optim_options = ["Adam", "AdamW", "SGD", "RMSprop"]

        self.is_running = False
        self.stop_training_flag = False
        self.q = queue.Queue()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_show(self):
        if not get_state("preprocessed"):
            self._show_blocked("Preprocess data first.\n← Go to 'Preprocessing'")
            return
        if not self.built_ui:
            self._build_ui()
            self.built_ui = True
            # Restore widget values from session (switches algo quietly, no state reset)
            mb_cfg = get_state("model_builder_config")
            if mb_cfg:
                self.restore_from_session(mb_cfg)
            # Re-enable train button and redraw result if already trained
            if get_state("trained") and not self.is_running:
                self.train_btn.configure(state="normal")
                self._draw_static_training_result()
            elif get_state("model_ready"):
                self.train_btn.configure(state="normal")

        # External algo change requested by Hyperopt "Apply Best"
        target_algo = get_state("selected_algo")
        if target_algo and target_algo != self.selected_algo:
            self.algo_var.set(target_algo)
            self._on_algo_change()
            set_state("selected_algo", None)

        applied = get_state("applied_hpo_params")
        if applied:
            self._apply_hpo_params(applied)
            set_state("applied_hpo_params", None)

    # ── Session helpers ───────────────────────────────────────────────────────

    def get_session_config(self) -> dict:
        """Snapshot all current widget values for session serialization."""
        cfg = {"algo": self.selected_algo}
        if self.selected_algo == "Neural Network":
            cfg.update({
                "num_layers":     self.num_layers_entry.get(),
                "neurons":        self.neurons_entry.get(),
                "act_hidden":     self.act_var.get(),
                "act_out":        self.out_act_var.get(),
                "dropout":        self.dropout_entry.get(),
                "l1":             self.l1_entry.get(),
                "l2":             self.l2_entry.get(),
                "loss":           self.loss_var.get(),
                "optimizer":      self.optim_var.get(),
                "lr":             self.lr_entry.get(),
                "batch_size":     self.bs_entry.get(),
                "epochs":         self.ep_entry.get(),
                "use_es":         self.use_es.get(),
                "es_patience":    self.es_pat_e.get(),
                "es_min_delta":   self.es_del_e.get(),
                "use_reduce_lr":  self.use_reduce_lr.get(),
                "lr_factor":      self.lr_factor_e.get(),
                "lr_patience":    self.lr_pat_e.get(),
                "lr_min":         self.lr_min_e.get(),
                "lr_scheduler":   self.lr_sched_var.get(),
                "lr_decay_steps": self.lr_decay_steps_e.get(),
                "lr_decay_rate":  self.lr_decay_rate_e.get(),
            })
        elif self.selected_algo == "XGBoost":
            cfg.update({
                "n_estimators":    self.xgb_n_est.get(),
                "max_depth":       self.xgb_depth.get(),
                "learning_rate":   self.xgb_lr.get(),
                "subsample":       self.xgb_sub.get(),
                "colsample_bytree": self.xgb_col.get(),
                "reg_alpha":       self.xgb_alpha.get(),
                "reg_lambda":      self.xgb_lambda.get(),
            })
        elif self.selected_algo == "Random Forest":
            cfg.update({
                "n_estimators":      self.rf_n_est.get(),
                "max_depth":         self.rf_depth.get(),
                "min_samples_split": self.rf_split.get(),
                "min_samples_leaf":  self.rf_leaf.get(),
                "max_features":      self.rf_feat_var.get(),
            })
        elif self.selected_algo == "Gaussian Process":
            cfg.update({
                "kernel":     self.gpr_kernel_var.get(),
                "alpha":      self.gpr_alpha.get(),
                "n_restarts": self.gpr_restarts.get(),
            })
        return cfg

    def restore_from_session(self, config: dict) -> None:
        """Restore all widget values without resetting trained/model_ready flags."""
        if not config:
            return
        algo = config.get("algo", "Neural Network")

        # Switch algo quietly (rebuild param cards without clearing state flags)
        if algo != self.selected_algo:
            self.algo_var.set(algo)
            self.selected_algo = algo
            for w in self.param_cards_frame.winfo_children():
                w.destroy()
            dispatch = {
                "Neural Network": self._build_nn_cards,
                "XGBoost":        self._build_xgb_card,
                "Random Forest":  self._build_rf_card,
                "Gaussian Process": self._build_gpr_card,
            }
            dispatch.get(algo, self._build_nn_cards)()

        def _set(entry, val):
            if val is None:
                return
            entry.delete(0, "end")
            entry.insert(0, str(val))

        if algo == "Neural Network":
            _set(self.num_layers_entry, config.get("num_layers"))
            _set(self.neurons_entry,    config.get("neurons"))
            if "act_hidden"    in config: self.act_var.set(config["act_hidden"])
            if "act_out"       in config: self.out_act_var.set(config["act_out"])
            _set(self.dropout_entry,   config.get("dropout"))
            _set(self.l1_entry,        config.get("l1"))
            _set(self.l2_entry,        config.get("l2"))
            if "loss"          in config: self.loss_var.set(config["loss"])
            if "optimizer"     in config: self.optim_var.set(config["optimizer"])
            _set(self.lr_entry,        config.get("lr"))
            _set(self.bs_entry,        config.get("batch_size"))
            _set(self.ep_entry,        config.get("epochs"))
            if "use_es"        in config: self.use_es.set(config["use_es"])
            _set(self.es_pat_e,        config.get("es_patience"))
            _set(self.es_del_e,        config.get("es_min_delta"))
            if "use_reduce_lr" in config: self.use_reduce_lr.set(config["use_reduce_lr"])
            _set(self.lr_factor_e,     config.get("lr_factor"))
            _set(self.lr_pat_e,        config.get("lr_patience"))
            _set(self.lr_min_e,        config.get("lr_min"))
            if "lr_scheduler"  in config: self.lr_sched_var.set(config["lr_scheduler"])
            _set(self.lr_decay_steps_e, config.get("lr_decay_steps"))
            _set(self.lr_decay_rate_e,  config.get("lr_decay_rate"))

        elif algo == "XGBoost":
            _set(self.xgb_n_est,   config.get("n_estimators"))
            _set(self.xgb_depth,   config.get("max_depth"))
            _set(self.xgb_lr,      config.get("learning_rate"))
            _set(self.xgb_sub,     config.get("subsample"))
            _set(self.xgb_col,     config.get("colsample_bytree"))
            _set(self.xgb_alpha,   config.get("reg_alpha"))
            _set(self.xgb_lambda,  config.get("reg_lambda"))

        elif algo == "Random Forest":
            _set(self.rf_n_est,  config.get("n_estimators"))
            _set(self.rf_depth,  config.get("max_depth"))
            _set(self.rf_split,  config.get("min_samples_split"))
            _set(self.rf_leaf,   config.get("min_samples_leaf"))
            if "max_features" in config: self.rf_feat_var.set(config["max_features"])

        elif algo == "Gaussian Process":
            if "kernel" in config: self.gpr_kernel_var.set(config["kernel"])
            _set(self.gpr_alpha,    config.get("alpha"))
            _set(self.gpr_restarts, config.get("n_restarts"))

    def _show_blocked(self, message):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.built_ui = False
        lbl = ctk.CTkLabel(self.content_frame, text=f"[ BLOCKED ]\n{message}",
                           font=FONTS["header"], text_color=COLORS["red"])
        lbl.grid(row=0, column=0, pady=50, padx=20)

    # ── Main UI builder ───────────────────────────────────────────────────────

    def _build_ui(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # ── Algorithm Selector Card ──
        algo_card = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        algo_card.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 5))
        algo_card.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(algo_card, text="ALGORITHM", font=FONTS["header"],
                     text_color=COLORS["magenta"]).grid(
            row=0, column=0, padx=20, pady=(15, 5), sticky="w")

        ctk.CTkLabel(algo_card, text="Select Model Type").grid(
            row=1, column=0, padx=20, sticky="w")
        self.algo_var = ctk.StringVar(value=self.selected_algo)
        ctk.CTkComboBox(algo_card, values=ALGO_NAMES, variable=self.algo_var,
                        width=220, command=self._on_algo_change).grid(
            row=2, column=0, padx=20, pady=(0, 15), sticky="w")

        ctk.CTkLabel(algo_card,
                     text="Neural Network: deep learning, highly flexible\n"
                          "XGBoost: gradient boosting, great on tabular data\n"
                          "Random Forest: robust, provides uncertainty bands\n"
                          "Gaussian Process: full probabilistic uncertainty",
                     text_color=COLORS["text_dim"], font=("Helvetica", 11),
                     justify="left").grid(row=1, column=1, rowspan=2, padx=20, pady=10, sticky="w")

        # ── Param cards container (rebuilt on algo change) ──
        self.param_cards_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.param_cards_frame.grid(row=1, column=0, sticky="ew")
        self.param_cards_frame.grid_columnconfigure(0, weight=1)

        # ── Control buttons ──
        ctrl_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        ctrl_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=20)

        self.build_btn = ctk.CTkButton(
            ctrl_frame, text="⚡ BUILD", height=40, font=("Helvetica", 14, "bold"),
            fg_color=COLORS["cyan"], hover_color="#00ACC1", text_color="#000000",
            command=self._build_model)
        self.build_btn.pack(side="left")

        self.train_btn = ctk.CTkButton(
            ctrl_frame, text="🚀 START TRAINING", height=40,
            font=("Helvetica", 14, "bold"), state="disabled", command=self.start_training)
        self.train_btn.pack(side="left", padx=15)

        self.stop_btn = ctk.CTkButton(
            ctrl_frame, text="🛑 STOP", height=40, width=80,
            font=("Helvetica", 14, "bold"), fg_color=COLORS["red"],
            hover_color="#CC0000", state="disabled", command=self.stop_training)
        self.stop_btn.pack(side="left")

        self.train_lbl = ctk.CTkLabel(ctrl_frame, text="Model not built.",
                                      font=FONTS["body"], text_color=COLORS["text_dim"])
        self.train_lbl.pack(side="left", padx=25)

        self.pb = ctk.CTkProgressBar(ctrl_frame, width=250)
        self.pb.set(0)
        self.pb.pack(side="right")

        # ── Plot / Status area ──
        self.plot_frame = ctk.CTkFrame(self.content_frame, fg_color="#000", height=310)
        self.plot_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=5)
        self.plot_frame.grid_propagate(False)

        # ── Log box ──
        self.log_box = ctk.CTkTextbox(self.content_frame, height=180,
                                      font=FONTS["code"], fg_color="#0D1117",
                                      text_color=COLORS["text"])
        self.log_box.grid(row=4, column=0, sticky="ew", padx=20, pady=(10, 30))
        self.log_box.configure(state="disabled")

        # Build initial param cards for NN
        self._build_nn_cards()

    # ── Algo change ───────────────────────────────────────────────────────────

    def _on_algo_change(self, algo=None):
        self.selected_algo = self.algo_var.get()
        for w in self.param_cards_frame.winfo_children():
            w.destroy()
        set_state("model_ready", False)
        set_state("trained", False)
        self._pending_surrogate = None
        self.train_btn.configure(state="disabled")
        self.train_lbl.configure(text="Model not built.", text_color=COLORS["text_dim"])
        self.pb.set(0)

        dispatch = {
            "Neural Network": self._build_nn_cards,
            "XGBoost": self._build_xgb_card,
            "Random Forest": self._build_rf_card,
            "Gaussian Process": self._build_gpr_card,
        }
        dispatch.get(self.selected_algo, self._build_nn_cards)()

    # ── NN param cards ────────────────────────────────────────────────────────

    def _build_nn_cards(self):
        c = self.param_cards_frame

        # Architecture
        arch_card = ctk.CTkFrame(c, fg_color=COLORS["bg_card"])
        arch_card.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 5))
        arch_card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(arch_card, text="ARCHITECTURE", font=FONTS["header"],
                     text_color=COLORS["magenta"]).grid(
            row=0, column=0, columnspan=4, padx=20, pady=(15, 10), sticky="w")

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
        ctk.CTkComboBox(arch_card, values=self.activation_options,
                        variable=self.act_var).grid(row=2, column=2, padx=10, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(arch_card, text="Output Activation").grid(row=1, column=3, padx=20, sticky="w")
        self.out_act_var = ctk.StringVar(value="Linear")
        ctk.CTkComboBox(arch_card, values=self.activation_options,
                        variable=self.out_act_var).grid(row=2, column=3, padx=20, pady=(0, 15), sticky="ew")

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

        # Compile & Callbacks
        compile_card = ctk.CTkFrame(c, fg_color=COLORS["bg_card"])
        compile_card.grid(row=1, column=0, sticky="ew", padx=20, pady=5)
        compile_card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(compile_card, text="COMPILE & HYPERPARAMS", font=FONTS["header"],
                     text_color=COLORS["magenta"]).grid(
            row=0, column=0, columnspan=4, padx=20, pady=(15, 10), sticky="w")

        ctk.CTkLabel(compile_card, text="Loss Function").grid(row=1, column=0, padx=20, sticky="w")
        self.loss_var = ctk.StringVar(value="MeanSquaredError")
        ctk.CTkComboBox(compile_card, values=self.loss_options,
                        variable=self.loss_var).grid(row=2, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(compile_card, text="Optimizer").grid(row=1, column=1, padx=10, sticky="w")
        self.optim_var = ctk.StringVar(value="Adam")
        ctk.CTkComboBox(compile_card, values=self.optim_options,
                        variable=self.optim_var).grid(row=2, column=1, padx=10, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(compile_card, text="Learning Rate").grid(row=1, column=2, padx=10, sticky="w")
        self.lr_entry = ctk.CTkEntry(compile_card)
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=2, column=2, padx=10, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(compile_card, text="Batch Size").grid(row=3, column=0, padx=20, sticky="w")
        self.bs_entry = ctk.CTkEntry(compile_card)
        self.bs_entry.insert(0, "64")
        self.bs_entry.grid(row=4, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(compile_card, text="Epochs").grid(row=3, column=1, padx=10, sticky="w")
        self.ep_entry = ctk.CTkEntry(compile_card)
        self.ep_entry.insert(0, "200")
        self.ep_entry.grid(row=4, column=1, padx=10, pady=(0, 15), sticky="ew")

        sep = ctk.CTkFrame(compile_card, height=2, fg_color=COLORS["border"])
        sep.grid(row=5, column=0, columnspan=4, sticky="ew", padx=20, pady=10)
        ctk.CTkLabel(compile_card, text="CALLBACKS", font=FONTS["header"],
                     text_color=COLORS["cyan"]).grid(
            row=6, column=0, columnspan=4, padx=20, pady=(5, 10), sticky="w")

        self.use_es = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(compile_card, text="Early Stopping", variable=self.use_es).grid(
            row=7, column=0, padx=20, pady=10, sticky="w")
        ctk.CTkLabel(compile_card, text="ES Patience").grid(row=7, column=1, padx=10, sticky="w")
        self.es_pat_e = ctk.CTkEntry(compile_card)
        self.es_pat_e.insert(0, "20")
        self.es_pat_e.grid(row=7, column=1, padx=(100, 10), pady=10, sticky="w")
        ctk.CTkLabel(compile_card, text="ES Min Δ").grid(row=7, column=2, padx=10, sticky="w")
        self.es_del_e = ctk.CTkEntry(compile_card)
        self.es_del_e.insert(0, "0.00001")
        self.es_del_e.grid(row=7, column=2, padx=(80, 10), pady=10, sticky="w")

        self.use_reduce_lr = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(compile_card, text="Reduce LR on Plateau",
                        variable=self.use_reduce_lr).grid(row=8, column=0, padx=20, pady=(0, 20), sticky="w")
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

        sep2 = ctk.CTkFrame(compile_card, height=2, fg_color=COLORS["border"])
        sep2.grid(row=9, column=0, columnspan=4, sticky="ew", padx=20, pady=(0, 10))
        ctk.CTkLabel(compile_card, text="LR SCHEDULER", font=FONTS["header"],
                     text_color=COLORS["cyan"]).grid(
            row=10, column=0, columnspan=4, padx=20, pady=(0, 10), sticky="w")
        ctk.CTkLabel(compile_card, text="Scheduler Type").grid(row=11, column=0, padx=20, sticky="w")
        self.lr_sched_var = ctk.StringVar(value="None (fixed LR)")
        self.lr_sched_combo = ctk.CTkComboBox(
            compile_card, values=["None (fixed LR)", "CosineDecay", "ExponentialDecay"],
            variable=self.lr_sched_var, command=lambda _: None)
        self.lr_sched_combo.grid(row=12, column=0, padx=20, pady=(0, 15), sticky="ew")
        ctk.CTkLabel(compile_card, text="Decay Steps (CosineDecay)").grid(row=11, column=1, padx=10, sticky="w")
        self.lr_decay_steps_e = ctk.CTkEntry(compile_card, placeholder_text="e.g. 1000")
        self.lr_decay_steps_e.insert(0, "1000")
        self.lr_decay_steps_e.grid(row=12, column=1, padx=10, pady=(0, 15), sticky="ew")
        ctk.CTkLabel(compile_card, text="Decay Rate (ExponentialDecay)").grid(row=11, column=2, padx=10, sticky="w")
        self.lr_decay_rate_e = ctk.CTkEntry(compile_card, placeholder_text="e.g. 0.96")
        self.lr_decay_rate_e.insert(0, "0.96")
        self.lr_decay_rate_e.grid(row=12, column=2, padx=10, pady=(0, 15), sticky="ew")

    # ── XGBoost param card ────────────────────────────────────────────────────

    def _build_xgb_card(self):
        c = self.param_cards_frame
        card = ctk.CTkFrame(c, fg_color=COLORS["bg_card"])
        card.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 5))
        card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(card, text="XGBOOST PARAMETERS", font=FONTS["header"],
                     text_color=COLORS["magenta"]).grid(
            row=0, column=0, columnspan=4, padx=20, pady=(15, 10), sticky="w")

        fields = [
            ("N Estimators", "xgb_n_est", "300"),
            ("Max Depth", "xgb_depth", "6"),
            ("Learning Rate", "xgb_lr", "0.1"),
            ("Subsample", "xgb_sub", "0.8"),
            ("ColSample ByTree", "xgb_col", "0.8"),
            ("Reg Alpha (L1)", "xgb_alpha", "0.0"),
            ("Reg Lambda (L2)", "xgb_lambda", "1.0"),
        ]
        for i, (label, attr, default) in enumerate(fields):
            col = i % 4
            row_lbl = (i // 4) * 2 + 1
            row_ent = row_lbl + 1
            ctk.CTkLabel(card, text=label).grid(row=row_lbl, column=col, padx=20, sticky="w")
            entry = ctk.CTkEntry(card)
            entry.insert(0, default)
            entry.grid(row=row_ent, column=col, padx=20, pady=(0, 15), sticky="ew")
            setattr(self, attr, entry)

    # ── Random Forest param card ──────────────────────────────────────────────

    def _build_rf_card(self):
        c = self.param_cards_frame
        card = ctk.CTkFrame(c, fg_color=COLORS["bg_card"])
        card.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 5))
        card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(card, text="RANDOM FOREST PARAMETERS", font=FONTS["header"],
                     text_color=COLORS["magenta"]).grid(
            row=0, column=0, columnspan=4, padx=20, pady=(15, 10), sticky="w")

        ctk.CTkLabel(card, text="N Estimators").grid(row=1, column=0, padx=20, sticky="w")
        self.rf_n_est = ctk.CTkEntry(card); self.rf_n_est.insert(0, "200")
        self.rf_n_est.grid(row=2, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(card, text="Max Depth (empty = None)").grid(row=1, column=1, padx=20, sticky="w")
        self.rf_depth = ctk.CTkEntry(card, placeholder_text="None")
        self.rf_depth.grid(row=2, column=1, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(card, text="Min Samples Split").grid(row=1, column=2, padx=20, sticky="w")
        self.rf_split = ctk.CTkEntry(card); self.rf_split.insert(0, "2")
        self.rf_split.grid(row=2, column=2, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(card, text="Min Samples Leaf").grid(row=1, column=3, padx=20, sticky="w")
        self.rf_leaf = ctk.CTkEntry(card); self.rf_leaf.insert(0, "1")
        self.rf_leaf.grid(row=2, column=3, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(card, text="Max Features").grid(row=3, column=0, padx=20, sticky="w")
        self.rf_feat_var = ctk.StringVar(value="sqrt")
        ctk.CTkComboBox(card, values=["sqrt", "log2", "1.0"],
                        variable=self.rf_feat_var).grid(row=4, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(card,
                     text="Uncertainty bands: variance across trees is shown in 1D Sensitivity.",
                     text_color=COLORS["green"], font=("Helvetica", 11)).grid(
            row=5, column=0, columnspan=4, padx=20, pady=(0, 15), sticky="w")

    # ── GPR param card ────────────────────────────────────────────────────────

    def _build_gpr_card(self):
        c = self.param_cards_frame
        card = ctk.CTkFrame(c, fg_color=COLORS["bg_card"])
        card.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 5))
        card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(card, text="GAUSSIAN PROCESS PARAMETERS", font=FONTS["header"],
                     text_color=COLORS["magenta"]).grid(
            row=0, column=0, columnspan=4, padx=20, pady=(15, 10), sticky="w")

        ctk.CTkLabel(card, text="Kernel").grid(row=1, column=0, padx=20, sticky="w")
        self.gpr_kernel_var = ctk.StringVar(value="RBF")
        ctk.CTkComboBox(card, values=["RBF", "Matern", "RationalQuadratic"],
                        variable=self.gpr_kernel_var).grid(row=2, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(card, text="Alpha (noise level)").grid(row=1, column=1, padx=20, sticky="w")
        self.gpr_alpha = ctk.CTkEntry(card); self.gpr_alpha.insert(0, "1e-6")
        self.gpr_alpha.grid(row=2, column=1, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(card, text="N Restarts Optimizer").grid(row=1, column=2, padx=20, sticky="w")
        self.gpr_restarts = ctk.CTkEntry(card); self.gpr_restarts.insert(0, "3")
        self.gpr_restarts.grid(row=2, column=2, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(card,
                     text="⚠ GPR scales as O(n³). Best for datasets < 2000 samples.\n"
                          "Provides calibrated uncertainty estimates in 1D Sensitivity.",
                     text_color=COLORS["orange"], font=("Helvetica", 11)).grid(
            row=3, column=0, columnspan=4, padx=20, pady=(0, 15), sticky="w")

    # ── Build (validate config) ───────────────────────────────────────────────

    def _build_model(self):
        if self.selected_algo == "Neural Network":
            self._build_nn_model()
        else:
            self._build_sklearn_model()

    def _build_nn_model(self):
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

        layers_config = [{"units": neurons, "activation": act_hidden, "dropout": dropout}
                         for _ in range(num_layers)]
        set_state("layers_config", layers_config)

        model_config = {
            "num_layers": num_layers, "neurons": neurons,
            "act_hidden": act_hidden, "act_out": act_out,
            "dropout": dropout, "l1": l1_val, "l2": l2_val,
            "layers": layers_config, "loss": self.loss_var.get(),
            "optimizer": self.optim_var.get(), "lr": lr,
            "batch_size": bs, "epochs": ep, "input_dim": input_dim,
        }
        set_state("model_ready", True)
        set_state("model_config", model_config)
        set_state("trained", False)

        y_train = get_state("y_train")
        output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
        temp_model = build_surrogate_model(input_dim, output_dim, num_layers, neurons,
                                           act_hidden, act_out, dropout, l1_val, l2_val)
        set_state("model_params_count", temp_model.count_params())

        self.train_lbl.configure(
            text=f"✓ Built: {num_layers}x{neurons} [{temp_model.count_params()} params]",
            text_color=COLORS["green"])
        self.train_btn.configure(state="normal")
        self._display_model_summary(temp_model)

    def _build_sklearn_model(self):
        try:
            if self.selected_algo == "XGBoost":
                from models.xgb_model import XGBoostSurrogate
                surrogate = XGBoostSurrogate(
                    n_estimators=int(self.xgb_n_est.get()),
                    max_depth=int(self.xgb_depth.get()),
                    learning_rate=float(self.xgb_lr.get()),
                    subsample=float(self.xgb_sub.get()),
                    colsample_bytree=float(self.xgb_col.get()),
                    reg_alpha=float(self.xgb_alpha.get()),
                    reg_lambda=float(self.xgb_lambda.get()),
                )
                summary = (f"XGBoost | n_est={self.xgb_n_est.get()} "
                           f"depth={self.xgb_depth.get()} lr={self.xgb_lr.get()}")

            elif self.selected_algo == "Random Forest":
                from models.rf_model import RandomForestSurrogate
                depth_str = self.rf_depth.get().strip()
                max_depth = int(depth_str) if depth_str and depth_str.lower() != "none" else None
                surrogate = RandomForestSurrogate(
                    n_estimators=int(self.rf_n_est.get()),
                    max_depth=max_depth,
                    min_samples_split=int(self.rf_split.get()),
                    min_samples_leaf=int(self.rf_leaf.get()),
                    max_features=self.rf_feat_var.get(),
                )
                summary = (f"Random Forest | n_est={self.rf_n_est.get()} "
                           f"max_depth={max_depth}")

            elif self.selected_algo == "Gaussian Process":
                from models.gpr_model import GPRSurrogate
                surrogate = GPRSurrogate(
                    kernel=self.gpr_kernel_var.get(),
                    alpha=float(self.gpr_alpha.get()),
                    n_restarts=int(self.gpr_restarts.get()),
                )
                summary = (f"GPR | kernel={self.gpr_kernel_var.get()} "
                           f"alpha={self.gpr_alpha.get()}")
            else:
                raise ValueError(f"Unknown algo: {self.selected_algo}")

        except ValueError as e:
            self.train_lbl.configure(text=f"Error: {e}", text_color=COLORS["red"])
            return

        self._pending_surrogate = surrogate
        set_state("model_ready", True)
        set_state("trained", False)
        self.train_btn.configure(state="normal")
        self.train_lbl.configure(text=f"✓ {summary}", text_color=COLORS["green"])
        self._log_clear()
        self._log(f"Config ready: {summary}\nClick 'START TRAINING' to fit.")

    # ── Training dispatch ─────────────────────────────────────────────────────

    def start_training(self):
        if self.is_running:
            return
        if self.selected_algo == "Neural Network":
            self._start_nn_training()
        else:
            self._start_sklearn_training()

    def stop_training(self):
        if self.is_running:
            self.stop_training_flag = True
            self.stop_btn.configure(state="disabled", text="Stopping...")

    # ── NN Training ───────────────────────────────────────────────────────────

    def _start_nn_training(self):
        self.is_running = True
        self.stop_training_flag = False
        self.train_btn.configure(state="disabled", text="TRAINING...")
        self.stop_btn.configure(state="normal", text="🛑 STOP")
        self.build_btn.configure(state="disabled")
        self._log_clear()

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
        lr_schedule = self._build_lr_schedule(float(self.lr_entry.get()))

        t = threading.Thread(
            target=self._run_nn_thread,
            args=(cfg, X_train, y_train, X_val, y_val,
                  use_es, es_pat, es_del, use_rlr, rlr_factor, rlr_pat, rlr_min, lr_schedule)
        )
        t.start()
        self._init_plot()
        self.after(100, self.process_queue)

    def _build_lr_schedule(self, base_lr: float):
        choice = self.lr_sched_var.get()
        if choice == "CosineDecay":
            try: steps = int(self.lr_decay_steps_e.get())
            except ValueError: steps = 1000
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=base_lr, decay_steps=steps, alpha=0.0)
        elif choice == "ExponentialDecay":
            try: rate = float(self.lr_decay_rate_e.get())
            except ValueError: rate = 0.96
            try: steps = int(self.lr_decay_steps_e.get())
            except ValueError: steps = 1000
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=base_lr, decay_steps=steps,
                decay_rate=rate, staircase=False)
        return base_lr

    def _run_nn_thread(self, cfg, X_train, y_train, X_val, y_val,
                       use_es, es_pat, es_del, use_rlr, rlr_factor, rlr_pat, rlr_min,
                       lr_schedule=None):
        try:
            output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
            model = build_surrogate_model(
                cfg["input_dim"], output_dim, cfg["num_layers"], cfg["neurons"],
                cfg["act_hidden"], cfg["act_out"], cfg["dropout"], cfg["l1"], cfg["l2"]
            )
            criterion = get_keras_loss(cfg["loss"])
            effective_lr = lr_schedule if lr_schedule is not None else cfg["lr"]
            optimizer = get_keras_optimizer(cfg["optimizer"], effective_lr)
            model.compile(optimizer=optimizer, loss=criterion)

            epochs = cfg["epochs"]
            st_callback = TkinterUpdateCallback(self.q, epochs, X_val, y_val, self)
            callbacks = [st_callback]
            if use_es:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=es_pat,
                                               min_delta=es_del, restore_best_weights=True, verbose=0))
            if use_rlr:
                callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=rlr_factor,
                                                   patience=rlr_pat, min_lr=rlr_min, verbose=0))

            self.q.put("Starting Training...")
            start_time = time.time()
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=cfg["batch_size"],
                      callbacks=callbacks, verbose=0)
            elapsed = time.time() - start_time

            t_losses = st_callback.train_losses
            v_losses = st_callback.val_losses

            if len(t_losses) > 0:
                val_pred = model.predict(X_val, verbose=0)
                r2 = r2_score(y_val, val_pred)

                set_state("model", model)
                set_state("surrogate_model", NeuralNetworkSurrogate(model))
                set_state("trained", True)
                set_state("results_stale", True)
                set_state("train_losses", t_losses)
                set_state("val_losses", v_losses)
                set_state("training_metrics", {
                    "best_val_loss": min(v_losses), "final_train_loss": t_losses[-1],
                    "r2": r2, "epochs_run": len(t_losses), "elapsed_seconds": elapsed
                })
                self._update_plot(t_losses, v_losses)
                summary = (
                    f"\n┌──────────────────────────────────────────┐\n"
                    f"│  TRAINING SUMMARY                        │\n"
                    f"├──────────────────────────────────────────┤\n"
                    f"│  Epochs run   : {len(t_losses):<25}│\n"
                    f"│  Best val loss: {min(v_losses):<25.6f}│\n"
                    f"│  Final R²     : {r2:<25.4f}│\n"
                    f"│  Time (s)     : {elapsed:<25.1f}│\n"
                    f"└──────────────────────────────────────────┘"
                )
                self.q.put(summary)
            else:
                self.q.put("\nTraining aborted before first epoch completed.")

        except Exception as e:
            self.q.put(f"ERROR: {str(e)}")
        self.q.put("DONE")

    # ── sklearn Training ──────────────────────────────────────────────────────

    def _start_sklearn_training(self):
        self.is_running = True
        self.stop_training_flag = False
        self.train_btn.configure(state="disabled", text="TRAINING...")
        self.build_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")
        self._log_clear()
        self.pb.set(0)

        # Clear plot frame and show indeterminate state
        for w in self.plot_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.plot_frame, text="⏳ Training in progress...",
                     font=FONTS["header"], text_color=COLORS["cyan"]).pack(expand=True)

        t = threading.Thread(
            target=self._run_sklearn_thread,
            args=(self._pending_surrogate,)
        )
        t.start()
        self.after(100, self.process_queue)

    def _run_sklearn_thread(self, surrogate):
        try:
            X_train, y_train = get_state("X_train"), get_state("y_train")
            X_val, y_val = get_state("X_val"), get_state("y_val")

            self.q.put(f"Fitting {surrogate.algo_name}...")
            start = time.time()
            surrogate.fit(X_train, y_train, X_val, y_val)
            elapsed = time.time() - start

            y_val_pred = surrogate.predict(X_val)
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            r2 = r2_score(y_val, y_val_pred)

            set_state("surrogate_model", surrogate)
            set_state("trained", True)
            set_state("results_stale", True)
            set_state("training_metrics", {"r2": r2, "elapsed_seconds": elapsed})

            summary = (
                f"\n┌──────────────────────────────────────────┐\n"
                f"│  TRAINING SUMMARY ({surrogate.algo_name:<20}│\n"
                f"├──────────────────────────────────────────┤\n"
                f"│  Val R²       : {r2:<25.4f}│\n"
                f"│  Time (s)     : {elapsed:<25.1f}│\n"
                f"└──────────────────────────────────────────┘"
            )
            self.q.put(summary)

        except Exception as e:
            import traceback
            self.q.put(f"ERROR: {str(e)}\n{traceback.format_exc()}")
        self.q.put("DONE_SKLEARN")

    # ── Queue processor ───────────────────────────────────────────────────────

    def process_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if isinstance(msg, dict) and msg["type"] == "epoch":
                    ep, tl, vl, r2 = msg["epoch"], msg["train_loss"], msg["val_loss"], msg["r2"]
                    self.pb.set(ep / get_state("model_config")["epochs"])
                    self.train_lbl.configure(
                        text=f"Epoch {ep} | Train: {tl:.6f} | Val: {vl:.6f} | R²: {r2:.4f}")
                    self._log(f"[{ep:>4}] train={tl:.6f} val={vl:.6f} R²={r2:.4f}")
                    if ep % 3 == 0 or ep == get_state("model_config")["epochs"]:
                        self._update_plot(msg["train_losses"], msg["val_losses"])
                elif isinstance(msg, str):
                    if msg == "DONE":
                        self._finish_training(nn=True)
                        break
                    elif msg == "DONE_SKLEARN":
                        self._finish_training(nn=False)
                        break
                    else:
                        self._log(msg)
        except queue.Empty:
            if self.is_running:
                self.after(50, self.process_queue)

    def _finish_training(self, nn: bool):
        self.is_running = False
        self.train_btn.configure(state="normal", text="🚀 START TRAINING")
        self.stop_btn.configure(state="disabled", text="🛑 STOP")
        self.build_btn.configure(state="normal")

        if nn and self.stop_training_flag:
            self._log("\n[ ABORTED ] Training stopped by user.")
            self.train_lbl.configure(text="⚠ Training stopped.", text_color=COLORS["orange"])
        else:
            set_state("session_unsaved", True)
            metrics = get_state("training_metrics") or {}
            r2 = metrics.get("r2", 0)
            self._log("\n[ OK ] Training Complete.")
            self.train_lbl.configure(
                text=f"✓ Training Complete  |  Val R²: {r2:.4f}",
                text_color=COLORS["green"])
            self.pb.set(1.0)
            if not nn:
                # Show post-training status in plot area
                for w in self.plot_frame.winfo_children():
                    w.destroy()
                ctk.CTkLabel(self.plot_frame,
                             text=f"✓ {self.selected_algo} trained  |  Val R²: {r2:.4f}",
                             font=FONTS["title"], text_color=COLORS["green"]).pack(expand=True)

    # ── Plot helpers ──────────────────────────────────────────────────────────

    def _draw_static_training_result(self):
        """Re-draw the training result area from saved state (after session load)."""
        train_losses = get_state("train_losses", [])
        val_losses   = get_state("val_losses",   [])
        metrics      = get_state("training_metrics") or {}
        r2           = metrics.get("r2", 0)
        surrogate    = get_state("surrogate_model")
        algo         = surrogate.algo_name if surrogate else self.selected_algo

        for w in self.plot_frame.winfo_children():
            w.destroy()

        if train_losses:
            # NN path: draw loss curves
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
            fig.patch.set_facecolor(COLORS["bg_card"])
            ax.set_facecolor(COLORS["bg_card"])
            ax.set_xlabel("Epoch", color=COLORS["text"])
            ax.set_ylabel("Loss",  color=COLORS["text"])
            ax.plot(range(1, len(train_losses) + 1), train_losses,
                    color=COLORS["cyan"],   label="Train Loss", linewidth=1.5)
            ax.plot(range(1, len(val_losses)   + 1), val_losses,
                    color=COLORS["orange"], label="Val Loss",   linewidth=1.5)
            ax.legend()
            ax.set_title("Training History (restored from session)",
                         color=COLORS["text_dim"], fontsize=9)
            for spine in ax.spines.values(): spine.set_color(COLORS["border"])
            ax.tick_params(colors=COLORS["text"])
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            add_save_button(self.plot_frame, canvas, "training_history.png")
        else:
            # Sklearn path: show status text
            ctk.CTkLabel(
                self.plot_frame,
                text=f"✓ {algo} trained  |  Val R²: {r2:.4f}",
                font=FONTS["title"], text_color=COLORS["green"],
            ).pack(expand=True)

        # Restore train label
        self.train_lbl.configure(
            text=f"✓ Restored  |  {algo}  |  Val R²: {r2:.4f}",
            text_color=COLORS["green"],
        )
        self.pb.set(1.0)

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
        add_save_button(self.plot_frame, self.canvas, "training_loss.png")

    def _update_plot(self, t_losses, v_losses):
        x = range(1, len(t_losses) + 1)
        self.line_train.set_data(x, t_losses)
        self.line_val.set_data(x, v_losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.yview("end")
        self.log_box.configure(state="disabled")

    def _log_clear(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("0.0", "end")
        self.log_box.configure(state="disabled")

    def _display_model_summary(self, model):
        lines = []
        model.summary(print_fn=lambda x: lines.append(x))
        self._log_clear()
        self._log("─── MODEL SUMMARY ───\n" + "\n".join(lines))

    # ── Apply HPO params ──────────────────────────────────────────────────────

    def _apply_hpo_params(self, p):
        algo = p.get("__algo__", "Neural Network")
        if algo != self.selected_algo:
            self.train_lbl.configure(
                text=f"⚠ HPO was for {algo} but current algo is {self.selected_algo}.",
                text_color=COLORS["orange"])
            return

        if algo == "Neural Network":
            self._apply_nn_hpo(p)
        elif algo == "XGBoost":
            self._apply_xgb_hpo(p)
        elif algo == "Random Forest":
            self._apply_rf_hpo(p)
        elif algo == "Gaussian Process":
            self._apply_gpr_hpo(p)

    def _apply_nn_hpo(self, p):
        self.num_layers_entry.delete(0, "end"); self.num_layers_entry.insert(0, str(p.get("n_layers", 1)))
        self.neurons_entry.delete(0, "end"); self.neurons_entry.insert(0, str(p.get("neurons", 64)))
        self.act_var.set(p.get("activation", "ReLU"))
        self.out_act_var.set(p.get("out_act", "Linear"))
        self.dropout_entry.delete(0, "end"); self.dropout_entry.insert(0, str(p.get("dropout", 0.0)))
        self.l1_entry.delete(0, "end"); self.l1_entry.insert(0, str(p.get("l1", 0.0)))
        self.l2_entry.delete(0, "end"); self.l2_entry.insert(0, str(p.get("l2", 0.0)))
        self.loss_var.set(p.get("loss", "MeanSquaredError"))
        self.optim_var.set(p.get("optimizer", "Adam"))
        self.lr_entry.delete(0, "end"); self.lr_entry.insert(0, str(p.get("lr", 0.001)))
        self.bs_entry.delete(0, "end"); self.bs_entry.insert(0, str(p.get("batch_size", 64)))
        self.ep_entry.delete(0, "end"); self.ep_entry.insert(0, str(p.get("epochs", 100)))
        self.es_pat_e.delete(0, "end"); self.es_pat_e.insert(0, str(p.get("es_pat", 20)))
        self.es_del_e.delete(0, "end"); self.es_del_e.insert(0, str(p.get("es_del", 0.00001)))
        self.lr_factor_e.delete(0, "end"); self.lr_factor_e.insert(0, str(p.get("rlr_factor", 0.5)))
        self.lr_pat_e.delete(0, "end"); self.lr_pat_e.insert(0, str(p.get("rlr_pat", 10)))
        self.train_lbl.configure(text="✨ Applied Best HPO Params!", text_color=COLORS["cyan"])

    def _apply_xgb_hpo(self, p):
        self.xgb_n_est.delete(0, "end"); self.xgb_n_est.insert(0, str(p.get("n_estimators", 300)))
        self.xgb_depth.delete(0, "end"); self.xgb_depth.insert(0, str(p.get("max_depth", 6)))
        self.xgb_lr.delete(0, "end"); self.xgb_lr.insert(0, str(p.get("lr", 0.1)))
        self.xgb_sub.delete(0, "end"); self.xgb_sub.insert(0, str(p.get("subsample", 0.8)))
        self.xgb_col.delete(0, "end"); self.xgb_col.insert(0, str(p.get("colsample_bytree", 0.8)))
        self.train_lbl.configure(text="✨ Applied Best XGBoost HPO Params!", text_color=COLORS["cyan"])

    def _apply_rf_hpo(self, p):
        self.rf_n_est.delete(0, "end"); self.rf_n_est.insert(0, str(p.get("n_estimators", 200)))
        md = p.get("max_depth")
        self.rf_depth.delete(0, "end")
        if md is not None: self.rf_depth.insert(0, str(md))
        self.rf_split.delete(0, "end"); self.rf_split.insert(0, str(p.get("min_samples_split", 2)))
        self.rf_leaf.delete(0, "end"); self.rf_leaf.insert(0, str(p.get("min_samples_leaf", 1)))
        self.train_lbl.configure(text="✨ Applied Best RF HPO Params!", text_color=COLORS["cyan"])

    def _apply_gpr_hpo(self, p):
        self.gpr_kernel_var.set(p.get("kernel", "RBF"))
        self.gpr_alpha.delete(0, "end"); self.gpr_alpha.insert(0, str(p.get("alpha", 1e-6)))
        self.train_lbl.configure(text="✨ Applied Best GPR HPO Params!", text_color=COLORS["cyan"])
