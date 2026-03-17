import customtkinter as ctk
import tkinter as tk
import tkinter.ttk as ttk
import threading
import queue
import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
)

from modules.model_builder import build_surrogate_model, get_keras_loss, get_keras_optimizer
from models import ALGO_NAMES
from utils.theme import COLORS, FONTS
from utils.state import get_state, set_state
from utils.plot_utils import add_save_button

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────── NN quick train/eval ─────────────────────────────────

def _nn_train_eval(input_dim, num_layers, neurons, act_hidden, act_out, dropout, l1, l2,
                   X_train, y_train, X_val, y_val, lr, optimizer_name, loss_name,
                   batch_size, epochs, es_pat, es_del, rlr_factor, rlr_pat, rlr_min):
    from sklearn.metrics import mean_squared_error
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
    model = build_surrogate_model(input_dim, output_dim, num_layers, neurons,
                                  act_hidden, act_out, dropout, l1, l2)
    criterion = get_keras_loss(loss_name)
    optimizer = get_keras_optimizer(optimizer_name, lr)
    model.compile(optimizer=optimizer, loss=criterion)
    callbacks = []
    if es_pat > 0:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=es_pat,
                                       min_delta=es_del, restore_best_weights=True, verbose=0))
    if rlr_pat > 0:
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=rlr_factor,
                                           patience=rlr_pat, min_lr=rlr_min, verbose=0))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks, verbose=0)
    return min(history.history['val_loss'])


# ─────────────────────── Frame ───────────────────────────────────────────────

class HyperoptFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkLabel(self, text="HYPERPARAMETER OPTIMIZATION 🔍",
                                   font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(20, 10), sticky="w", padx=30)

        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure((0, 1), weight=1)

        self.built_ui = False
        self.q = queue.Queue()
        self.is_running = False

    def on_show(self):
        if not get_state("preprocessed"):
            self._show_blocked("Preprocess data first.\n← Go to 'Preprocessing'")
            return
        if not self.built_ui:
            self._build_ui()
            self.built_ui = True
            # Re-enable apply button when best_params are already in state
            # (e.g. after loading a session that had HPO results)
            if get_state("best_params"):
                self.apply_btn.configure(state="normal", text="✨ APPLY BEST PARAMS")

    def _show_blocked(self, message):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.built_ui = False
        lbl = ctk.CTkLabel(self.content_frame, text=f"[ BLOCKED ]\n{message}",
                           font=FONTS["header"], text_color=COLORS["red"])
        lbl.grid(row=0, column=0, pady=50, padx=20)

    # ── UI builder ────────────────────────────────────────────────────────────

    def _build_ui(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Strategy + Algorithm card
        c1 = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        c1.grid(row=0, column=0, columnspan=2, sticky="ew", pady=10, padx=20)
        c1.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(c1, text="Algorithm").grid(row=0, column=0, padx=20, pady=10, sticky="w")
        self.hpo_algo_var = ctk.StringVar(value="Neural Network")
        ctk.CTkComboBox(c1, values=ALGO_NAMES, variable=self.hpo_algo_var,
                        width=200, command=self._on_hpo_algo_change).grid(
            row=0, column=1, padx=20, pady=10, sticky="w")

        ctk.CTkLabel(c1, text="Optimization Method").grid(row=0, column=2, padx=20, pady=10, sticky="w")
        self.strat_var = ctk.StringVar(value="Optuna (TPE)")
        ctk.CTkComboBox(c1, values=["Optuna (TPE)", "Random Search"],
                        variable=self.strat_var).grid(row=0, column=3, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(c1, text="Trials").grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")
        self.trials_e = ctk.CTkEntry(c1)
        self.trials_e.insert(0, "15")
        self.trials_e.grid(row=1, column=1, padx=20, pady=(0, 10), sticky="w")

        # Dynamic search space + compile cards
        self.hpo_params_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.hpo_params_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.hpo_params_frame.grid_columnconfigure(0, weight=1)
        self._build_nn_hpo_params()

        # Run button
        self.run_btn = ctk.CTkButton(
            self.content_frame, text="⚡ RUN OPTIMIZATION", height=40,
            font=("Helvetica", 14, "bold"), command=self.start_optimization)
        self.run_btn.grid(row=2, column=0, columnspan=2, pady=20)

        # Log
        self.log_box = ctk.CTkTextbox(self.content_frame, height=200, font=FONTS["code"],
                                      fg_color="#0D1117", text_color=COLORS["text"])
        self.log_box.grid(row=3, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
        self.log_box.configure(state="disabled")

        # Results block
        resc = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        resc.grid(row=4, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
        resc.grid_columnconfigure(0, weight=1)

        self.apply_btn = ctk.CTkButton(
            resc, text="✓ APPLY BEST TO MODEL BUILDER",
            font=("Helvetica", 14, "bold"), fg_color=COLORS["green"],
            hover_color="#2E7D32", state="disabled", command=self._apply_best)
        self.apply_btn.grid(row=0, column=0, pady=10)

        contour_ctrl = ctk.CTkFrame(resc, fg_color="transparent")
        contour_ctrl.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 5))
        ctk.CTkLabel(contour_ctrl, text="Contour Param X:", font=("Helvetica", 11)).pack(side="left", padx=(0, 6))
        self.contour_x_var = ctk.StringVar(value="neurons")
        self.contour_x_combo = ctk.CTkComboBox(contour_ctrl, values=["neurons"],
                                               variable=self.contour_x_var, width=130)
        self.contour_x_combo.pack(side="left", padx=(0, 14))
        ctk.CTkLabel(contour_ctrl, text="Contour Param Y:", font=("Helvetica", 11)).pack(side="left", padx=(0, 6))
        self.contour_y_var = ctk.StringVar(value="lr")
        self.contour_y_combo = ctk.CTkComboBox(contour_ctrl, values=["lr"],
                                               variable=self.contour_y_var, width=130)
        self.contour_y_combo.pack(side="left")

        self.results_tabview = ctk.CTkTabview(resc, fg_color=COLORS["bg"], height=350)
        self.results_tabview.grid(row=2, column=0, sticky="nsew", pady=5, padx=10)
        for tab_name in ["Optimization History", "Best Trials", "Param Importances",
                         "Parallel Coords", "Contour Plot"]:
            self.results_tabview.add(tab_name)

    # ── Algo change in HPO ────────────────────────────────────────────────────

    def _on_hpo_algo_change(self, algo=None):
        for w in self.hpo_params_frame.winfo_children():
            w.destroy()
        algo = self.hpo_algo_var.get()
        dispatch = {
            "Neural Network": self._build_nn_hpo_params,
            "XGBoost": self._build_xgb_hpo_params,
            "Random Forest": self._build_rf_hpo_params,
            "Gaussian Process": self._build_gpr_hpo_params,
        }
        dispatch.get(algo, self._build_nn_hpo_params)()

    # ── NN search space ───────────────────────────────────────────────────────

    def _build_nn_hpo_params(self):
        c = self.hpo_params_frame

        ss = ctk.CTkFrame(c, fg_color=COLORS["bg_card"])
        ss.grid(row=0, column=0, sticky="ew", pady=10, padx=20)
        ss.grid_columnconfigure((0, 1, 2, 3), weight=1)
        ctk.CTkLabel(ss, text="ARCHITECTURE SEARCH SPACE", font=FONTS["header"],
                     text_color=COLORS["magenta"]).grid(row=0, column=0, columnspan=4, padx=20, pady=10, sticky="w")

        ctk.CTkLabel(ss, text="Min/Max Hidden Layers").grid(row=1, column=0, padx=20, pady=5, sticky="w")
        self.min_l = ctk.CTkEntry(ss, width=60); self.min_l.insert(0, "1")
        self.min_l.grid(row=1, column=1, pady=5, sticky="w")
        self.max_l = ctk.CTkEntry(ss, width=60); self.max_l.insert(0, "4")
        self.max_l.grid(row=1, column=1, padx=(70, 0), pady=5, sticky="w")

        ctk.CTkLabel(ss, text="Min/Max Neurons").grid(row=1, column=2, padx=20, pady=5, sticky="w")
        self.min_u = ctk.CTkEntry(ss, width=60); self.min_u.insert(0, "16")
        self.min_u.grid(row=1, column=3, pady=5, sticky="w")
        self.max_u = ctk.CTkEntry(ss, width=60); self.max_u.insert(0, "256")
        self.max_u.grid(row=1, column=3, padx=(70, 0), pady=5, sticky="w")

        ctk.CTkLabel(ss, text="Dropout Min/Max").grid(row=2, column=0, padx=20, pady=5, sticky="w")
        self.min_d = ctk.CTkEntry(ss, width=60); self.min_d.insert(0, "0.0")
        self.min_d.grid(row=2, column=1, pady=5, sticky="w")
        self.max_d = ctk.CTkEntry(ss, width=60); self.max_d.insert(0, "0.5")
        self.max_d.grid(row=2, column=1, padx=(70, 0), pady=5, sticky="w")

        ctk.CTkLabel(ss, text="L1 Max / L2 Max").grid(row=2, column=2, padx=20, pady=5, sticky="w")
        self.max_l1 = ctk.CTkEntry(ss, width=60); self.max_l1.insert(0, "0.01")
        self.max_l1.grid(row=2, column=3, pady=5, sticky="w")
        self.max_l2 = ctk.CTkEntry(ss, width=60); self.max_l2.insert(0, "0.01")
        self.max_l2.grid(row=2, column=3, padx=(70, 0), pady=5, sticky="w")

        ctk.CTkLabel(ss, text="Hidden Activations (comma-sep)").grid(row=3, column=0, padx=20, pady=5, sticky="w")
        self.act_list = ctk.CTkEntry(ss)
        self.act_list.insert(0, "ReLU, LeakyReLU, ELU, Tanh")
        self.act_list.grid(row=3, column=1, pady=5, sticky="ew")

        ctk.CTkLabel(ss, text="Output Activation").grid(row=3, column=2, padx=20, pady=5, sticky="w")
        self.out_act = ctk.StringVar(value="Linear")
        ctk.CTkComboBox(ss, values=["Linear", "Sigmoid", "ReLU", "Tanh"],
                        variable=self.out_act).grid(row=3, column=3, pady=5, sticky="ew")

        cc = ctk.CTkFrame(c, fg_color=COLORS["bg_card"])
        cc.grid(row=1, column=0, sticky="ew", pady=10, padx=20)
        cc.grid_columnconfigure((0, 1, 2, 3), weight=1)
        ctk.CTkLabel(cc, text="COMPILE & HYPERPARAMS", font=FONTS["header"],
                     text_color=COLORS["magenta"]).grid(row=0, column=0, columnspan=4, padx=20, pady=10, sticky="w")

        ctk.CTkLabel(cc, text="LR Min/Max").grid(row=1, column=0, padx=20, pady=5, sticky="w")
        self.min_lr = ctk.CTkEntry(cc, width=60); self.min_lr.insert(0, "1e-4")
        self.min_lr.grid(row=1, column=1, pady=5, sticky="w")
        self.max_lr = ctk.CTkEntry(cc, width=60); self.max_lr.insert(0, "1e-2")
        self.max_lr.grid(row=1, column=1, padx=(70, 0), pady=5, sticky="w")

        ctk.CTkLabel(cc, text="Batch Sizes (comma-sep)").grid(row=1, column=2, padx=20, pady=5, sticky="w")
        self.bs_list = ctk.CTkEntry(cc); self.bs_list.insert(0, "32, 64, 128")
        self.bs_list.grid(row=1, column=3, pady=5, sticky="ew")

        ctk.CTkLabel(cc, text="Optimizers (comma-sep)").grid(row=2, column=0, padx=20, pady=5, sticky="w")
        self.opt_list = ctk.CTkEntry(cc); self.opt_list.insert(0, "Adam, RMSprop")
        self.opt_list.grid(row=2, column=1, pady=5, sticky="ew")

        ctk.CTkLabel(cc, text="Loss Function").grid(row=2, column=2, padx=20, pady=5, sticky="w")
        self.loss_var = ctk.StringVar(value="MeanSquaredError")
        ctk.CTkComboBox(cc, values=["MeanSquaredError", "MeanAbsoluteError", "Huber", "LogCosh"],
                        variable=self.loss_var).grid(row=2, column=3, pady=5, sticky="ew")

        ctk.CTkLabel(cc, text="Epochs").grid(row=3, column=0, padx=20, pady=5, sticky="w")
        self.ep_e = ctk.CTkEntry(cc); self.ep_e.insert(0, "100")
        self.ep_e.grid(row=3, column=1, pady=5, sticky="ew")

        sep = ctk.CTkFrame(cc, height=2, fg_color=COLORS["border"])
        sep.grid(row=4, column=0, columnspan=4, sticky="ew", padx=20, pady=10)
        ctk.CTkLabel(cc, text="CALLBACKS", font=FONTS["header"],
                     text_color=COLORS["cyan"]).grid(row=5, column=0, columnspan=4, padx=20, pady=10, sticky="w")

        ctk.CTkLabel(cc, text="Early Stopping Patience").grid(row=6, column=0, padx=20, pady=5, sticky="w")
        self.es_pat_e = ctk.CTkEntry(cc); self.es_pat_e.insert(0, "20")
        self.es_pat_e.grid(row=6, column=1, pady=5, sticky="ew")
        ctk.CTkLabel(cc, text="ES Min Delta").grid(row=6, column=2, padx=20, pady=5, sticky="w")
        self.es_del_e = ctk.CTkEntry(cc); self.es_del_e.insert(0, "0.00001")
        self.es_del_e.grid(row=6, column=3, pady=5, sticky="ew")

        ctk.CTkLabel(cc, text="ReduceLR Factor").grid(row=7, column=0, padx=20, pady=5, sticky="w")
        self.lr_factor_e = ctk.CTkEntry(cc); self.lr_factor_e.insert(0, "0.5")
        self.lr_factor_e.grid(row=7, column=1, pady=5, sticky="ew")
        ctk.CTkLabel(cc, text="ReduceLR Patience").grid(row=7, column=2, padx=20, pady=5, sticky="w")
        self.lr_pat_e = ctk.CTkEntry(cc); self.lr_pat_e.insert(0, "10")
        self.lr_pat_e.grid(row=7, column=3, pady=5, sticky="ew")

    # ── XGBoost search space ──────────────────────────────────────────────────

    def _build_xgb_hpo_params(self):
        c = self.hpo_params_frame
        card = ctk.CTkFrame(c, fg_color=COLORS["bg_card"])
        card.grid(row=0, column=0, sticky="ew", pady=10, padx=20)
        card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(card, text="XGBOOST SEARCH SPACE", font=FONTS["header"],
                     text_color=COLORS["magenta"]).grid(row=0, column=0, columnspan=4, padx=20, pady=10, sticky="w")

        def _pair(parent, row, label, attr_min, attr_max, default_min, default_max, col=0):
            ctk.CTkLabel(parent, text=f"{label} Min/Max").grid(row=row, column=col, padx=20, pady=5, sticky="w")
            e1 = ctk.CTkEntry(parent, width=80); e1.insert(0, str(default_min))
            e1.grid(row=row, column=col + 1, pady=5, sticky="w")
            e2 = ctk.CTkEntry(parent, width=80); e2.insert(0, str(default_max))
            e2.grid(row=row, column=col + 1, padx=(90, 0), pady=5, sticky="w")
            setattr(self, attr_min, e1); setattr(self, attr_max, e2)

        _pair(card, 1, "N Estimators", "xgb_min_est", "xgb_max_est", 50, 500)
        _pair(card, 1, "Max Depth", "xgb_min_depth", "xgb_max_depth", 3, 10, col=2)
        _pair(card, 2, "Learning Rate", "xgb_min_lr", "xgb_max_lr", "1e-3", "3e-1")
        _pair(card, 2, "Subsample", "xgb_min_sub", "xgb_max_sub", 0.5, 1.0, col=2)
        _pair(card, 3, "ColSample", "xgb_min_col", "xgb_max_col", 0.5, 1.0)

        ctk.CTkLabel(card, text="Reg Alpha Max").grid(row=3, column=2, padx=20, pady=5, sticky="w")
        self.xgb_max_alpha = ctk.CTkEntry(card, width=80); self.xgb_max_alpha.insert(0, "1.0")
        self.xgb_max_alpha.grid(row=3, column=3, pady=5, sticky="w")

        ctk.CTkLabel(card, text="Reg Lambda Max").grid(row=4, column=0, padx=20, pady=5, sticky="w")
        self.xgb_max_lambda = ctk.CTkEntry(card, width=80); self.xgb_max_lambda.insert(0, "10.0")
        self.xgb_max_lambda.grid(row=4, column=1, pady=(5, 15), sticky="w")

    # ── RF search space ───────────────────────────────────────────────────────

    def _build_rf_hpo_params(self):
        c = self.hpo_params_frame
        card = ctk.CTkFrame(c, fg_color=COLORS["bg_card"])
        card.grid(row=0, column=0, sticky="ew", pady=10, padx=20)
        card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(card, text="RANDOM FOREST SEARCH SPACE", font=FONTS["header"],
                     text_color=COLORS["magenta"]).grid(row=0, column=0, columnspan=4, padx=20, pady=10, sticky="w")

        ctk.CTkLabel(card, text="N Estimators Min/Max").grid(row=1, column=0, padx=20, pady=5, sticky="w")
        self.rf_hpo_min_est = ctk.CTkEntry(card, width=80); self.rf_hpo_min_est.insert(0, "50")
        self.rf_hpo_min_est.grid(row=1, column=1, pady=5, sticky="w")
        self.rf_hpo_max_est = ctk.CTkEntry(card, width=80); self.rf_hpo_max_est.insert(0, "500")
        self.rf_hpo_max_est.grid(row=1, column=1, padx=(90, 0), pady=5, sticky="w")

        ctk.CTkLabel(card, text="Max Depth Min/Max").grid(row=1, column=2, padx=20, pady=5, sticky="w")
        self.rf_hpo_min_depth = ctk.CTkEntry(card, width=80); self.rf_hpo_min_depth.insert(0, "3")
        self.rf_hpo_min_depth.grid(row=1, column=3, pady=5, sticky="w")
        self.rf_hpo_max_depth = ctk.CTkEntry(card, width=80); self.rf_hpo_max_depth.insert(0, "20")
        self.rf_hpo_max_depth.grid(row=1, column=3, padx=(90, 0), pady=5, sticky="w")

        ctk.CTkLabel(card, text="Min Samples Split Min/Max").grid(row=2, column=0, padx=20, pady=5, sticky="w")
        self.rf_hpo_min_split = ctk.CTkEntry(card, width=80); self.rf_hpo_min_split.insert(0, "2")
        self.rf_hpo_min_split.grid(row=2, column=1, pady=5, sticky="w")
        self.rf_hpo_max_split = ctk.CTkEntry(card, width=80); self.rf_hpo_max_split.insert(0, "20")
        self.rf_hpo_max_split.grid(row=2, column=1, padx=(90, 0), pady=5, sticky="w")

        ctk.CTkLabel(card, text="Min Samples Leaf Min/Max").grid(row=2, column=2, padx=20, pady=5, sticky="w")
        self.rf_hpo_min_leaf = ctk.CTkEntry(card, width=80); self.rf_hpo_min_leaf.insert(0, "1")
        self.rf_hpo_min_leaf.grid(row=2, column=3, pady=5, sticky="w")
        self.rf_hpo_max_leaf = ctk.CTkEntry(card, width=80); self.rf_hpo_max_leaf.insert(0, "10")
        self.rf_hpo_max_leaf.grid(row=2, column=3, padx=(90, 0), pady=(5, 15), sticky="w")

    # ── GPR search space ──────────────────────────────────────────────────────

    def _build_gpr_hpo_params(self):
        c = self.hpo_params_frame
        card = ctk.CTkFrame(c, fg_color=COLORS["bg_card"])
        card.grid(row=0, column=0, sticky="ew", pady=10, padx=20)
        card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(card, text="GAUSSIAN PROCESS SEARCH SPACE", font=FONTS["header"],
                     text_color=COLORS["magenta"]).grid(row=0, column=0, columnspan=4, padx=20, pady=10, sticky="w")

        ctk.CTkLabel(card, text="Kernels (comma-sep)").grid(row=1, column=0, padx=20, pady=5, sticky="w")
        self.gpr_hpo_kernels = ctk.CTkEntry(card)
        self.gpr_hpo_kernels.insert(0, "RBF, Matern, RationalQuadratic")
        self.gpr_hpo_kernels.grid(row=1, column=1, pady=5, sticky="ew")

        ctk.CTkLabel(card, text="Alpha Min / Max").grid(row=1, column=2, padx=20, pady=5, sticky="w")
        self.gpr_hpo_min_alpha = ctk.CTkEntry(card, width=80); self.gpr_hpo_min_alpha.insert(0, "1e-8")
        self.gpr_hpo_min_alpha.grid(row=1, column=3, pady=5, sticky="w")
        self.gpr_hpo_max_alpha = ctk.CTkEntry(card, width=80); self.gpr_hpo_max_alpha.insert(0, "1e-2")
        self.gpr_hpo_max_alpha.grid(row=1, column=3, padx=(90, 0), pady=5, sticky="w")

        ctk.CTkLabel(card, text="N Restarts Min/Max").grid(row=2, column=0, padx=20, pady=5, sticky="w")
        self.gpr_hpo_min_restarts = ctk.CTkEntry(card, width=80); self.gpr_hpo_min_restarts.insert(0, "1")
        self.gpr_hpo_min_restarts.grid(row=2, column=1, pady=5, sticky="w")
        self.gpr_hpo_max_restarts = ctk.CTkEntry(card, width=80); self.gpr_hpo_max_restarts.insert(0, "5")
        self.gpr_hpo_max_restarts.grid(row=2, column=1, padx=(90, 0), pady=(5, 15), sticky="w")

        ctk.CTkLabel(card,
                     text="⚠ GPR optimization can be slow on large datasets.",
                     text_color=COLORS["orange"], font=("Helvetica", 11)).grid(
            row=3, column=0, columnspan=4, padx=20, pady=(0, 15), sticky="w")

    # ── Apply best ────────────────────────────────────────────────────────────

    def _apply_best(self):
        best = get_state("best_params")
        if best:
            best_dict = best.copy()
            best_dict["__algo__"] = self.hpo_algo_var.get()
            set_state("applied_hpo_params", best_dict)
            set_state("selected_algo", self.hpo_algo_var.get())
            set_state("session_unsaved", True)
            self._log("\n✨ Best parameters saved!\nGo to 'Model Builder' to see them applied.")
            self.apply_btn.configure(state="disabled", text="APPLIED")

    def _log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.yview("end")
        self.log_box.configure(state="disabled")

    # ── Start optimization ────────────────────────────────────────────────────

    def start_optimization(self):
        if self.is_running: return
        self.is_running = True
        self.run_btn.configure(state="disabled", text="RUNNING...")
        self.log_box.configure(state="normal"); self.log_box.delete("0.0", "end")
        self.log_box.configure(state="disabled")

        algo = self.hpo_algo_var.get()

        try:
            base_cfg = {
                "algo": algo,
                "strat": self.strat_var.get(),
                "trials": int(self.trials_e.get()),
                "X_train": get_state("X_train"),
                "y_train": get_state("y_train"),
                "X_val": get_state("X_val"),
                "y_val": get_state("y_val"),
            }

            if algo == "Neural Network":
                cfg = {**base_cfg,
                       "min_l": int(self.min_l.get()), "max_l": int(self.max_l.get()),
                       "min_u": int(self.min_u.get()), "max_u": int(self.max_u.get()),
                       "min_d": float(self.min_d.get()), "max_d": float(self.max_d.get()),
                       "max_l1": float(self.max_l1.get()), "max_l2": float(self.max_l2.get()),
                       "act_list": [x.strip() for x in self.act_list.get().split(",") if x.strip()],
                       "out_act": self.out_act.get(),
                       "min_lr": float(self.min_lr.get()), "max_lr": float(self.max_lr.get()),
                       "bs_list": [int(x.strip()) for x in self.bs_list.get().split(",") if x.strip()],
                       "opt_list": [x.strip() for x in self.opt_list.get().split(",") if x.strip()],
                       "loss": self.loss_var.get(), "epochs": int(self.ep_e.get()),
                       "es_pat": int(self.es_pat_e.get()), "es_del": float(self.es_del_e.get()),
                       "rlr_factor": float(self.lr_factor_e.get()),
                       "rlr_pat": int(self.lr_pat_e.get()), "rlr_min": 1e-6,
                       "input_dim": get_state("X_train").shape[1]}

            elif algo == "XGBoost":
                cfg = {**base_cfg,
                       "xgb_min_est": int(self.xgb_min_est.get()),
                       "xgb_max_est": int(self.xgb_max_est.get()),
                       "xgb_min_depth": int(self.xgb_min_depth.get()),
                       "xgb_max_depth": int(self.xgb_max_depth.get()),
                       "xgb_min_lr": float(self.xgb_min_lr.get()),
                       "xgb_max_lr": float(self.xgb_max_lr.get()),
                       "xgb_min_sub": float(self.xgb_min_sub.get()),
                       "xgb_max_sub": float(self.xgb_max_sub.get()),
                       "xgb_min_col": float(self.xgb_min_col.get()),
                       "xgb_max_col": float(self.xgb_max_col.get()),
                       "xgb_max_alpha": float(self.xgb_max_alpha.get()),
                       "xgb_max_lambda": float(self.xgb_max_lambda.get())}

            elif algo == "Random Forest":
                cfg = {**base_cfg,
                       "rf_min_est": int(self.rf_hpo_min_est.get()),
                       "rf_max_est": int(self.rf_hpo_max_est.get()),
                       "rf_min_depth": int(self.rf_hpo_min_depth.get()),
                       "rf_max_depth": int(self.rf_hpo_max_depth.get()),
                       "rf_min_split": int(self.rf_hpo_min_split.get()),
                       "rf_max_split": int(self.rf_hpo_max_split.get()),
                       "rf_min_leaf": int(self.rf_hpo_min_leaf.get()),
                       "rf_max_leaf": int(self.rf_hpo_max_leaf.get())}

            elif algo == "Gaussian Process":
                cfg = {**base_cfg,
                       "gpr_kernels": [x.strip() for x in self.gpr_hpo_kernels.get().split(",") if x.strip()],
                       "gpr_min_alpha": float(self.gpr_hpo_min_alpha.get()),
                       "gpr_max_alpha": float(self.gpr_hpo_max_alpha.get()),
                       "gpr_min_restarts": int(self.gpr_hpo_min_restarts.get()),
                       "gpr_max_restarts": int(self.gpr_hpo_max_restarts.get())}
            else:
                raise ValueError(f"Unknown algo: {algo}")

        except Exception as e:
            self._log(f"ERROR parsing inputs: {e}")
            self.is_running = False
            self.run_btn.configure(state="normal", text="⚡ RUN OPTIMIZATION")
            return

        t = threading.Thread(target=self._run_optuna_thread, args=(cfg,))
        t.start()
        self.after(100, self.process_queue)

    def process_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg == "DONE":
                    self.is_running = False
                    self.run_btn.configure(state="normal", text="⚡ RUN OPTIMIZATION")
                    self.apply_btn.configure(state="normal", text="✓ APPLY BEST TO MODEL BUILDER")
                    self._log("\n[ OK ] Optimization Complete.")
                    self._draw_results()
                    break
                else:
                    self._log(msg)
        except queue.Empty:
            if self.is_running:
                self.after(100, self.process_queue)

    # ── Results rendering ─────────────────────────────────────────────────────

    def _draw_results(self):
        study = get_state("optuna_study")
        if not study: return
        param_names = list(study.best_params.keys())
        self.contour_x_combo.configure(values=param_names)
        self.contour_y_combo.configure(values=param_names)
        if len(param_names) >= 2:
            self.contour_x_var.set(param_names[0])
            self.contour_y_var.set(param_names[1])
        self._draw_opt_history(self.results_tabview.tab("Optimization History"), study)
        self._draw_best_trials_table(self.results_tabview.tab("Best Trials"), study)
        self._draw_param_importances(self.results_tabview.tab("Param Importances"), study)
        self._draw_parallel_coords(self.results_tabview.tab("Parallel Coords"), study)
        self._draw_contour_plot(self.results_tabview.tab("Contour Plot"), study)

    def _embed_fig(self, parent, fig, default_name="plot.png"):
        for w in parent.winfo_children(): w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        add_save_button(parent, canvas, default_name)
        plt.close(fig)

    def _draw_opt_history(self, parent, study):
        for w in parent.winfo_children(): w.destroy()
        try:
            plt.style.use('dark_background')
            ax = plot_optimization_history(study)
            fig = ax.figure
            fig.set_size_inches(8, 3.2)
            fig.patch.set_facecolor(COLORS["bg_card"])
            ax.set_facecolor(COLORS["bg_card"])
            fig.tight_layout()
            self._embed_fig(parent, fig, "hpo_optimization_history.png")
        except Exception as e:
            ctk.CTkLabel(parent, text=f"Plot error: {e}", text_color=COLORS["red"]).pack(pady=20)

    def _draw_best_trials_table(self, parent, study, top_k=10):
        for w in parent.winfo_children(): w.destroy()
        trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))
        top = trials[:top_k]
        if not top:
            ctk.CTkLabel(parent, text="No completed trials.", text_color=COLORS["red"]).pack(pady=20)
            return
        param_names = list(study.best_params.keys())
        col_names = ["Rank", "Val Loss"] + param_names
        style = ttk.Style()
        style.theme_use("default")
        style.configure("HPO.Treeview", background="#1A1A2E", foreground=COLORS["text"],
                        fieldbackground="#1A1A2E", rowheight=24, font=("Helvetica", 10))
        style.configure("HPO.Treeview.Heading", background=COLORS["bg_card"],
                        foreground=COLORS["cyan"], font=("Helvetica", 10, "bold"), relief="flat")
        style.map("HPO.Treeview", background=[("selected", COLORS["primary_dark"])])
        container = tk.Frame(parent, bg="#1A1A2E")
        container.pack(fill="both", expand=True, padx=6, pady=6)
        tree = ttk.Treeview(container, columns=col_names, show="headings", style="HPO.Treeview")
        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        tree.column("Rank", width=50, anchor="center")
        tree.column("Val Loss", width=100, anchor="center")
        tree.heading("Rank", text="#")
        tree.heading("Val Loss", text="Val Loss")
        for p in param_names:
            tree.column(p, width=max(80, len(p) * 8), anchor="center")
            tree.heading(p, text=p)
        tree.tag_configure("best", foreground=COLORS["green"])
        tree.tag_configure("odd", background="#16213E")
        tree.tag_configure("even", background="#1A1A2E")
        for rank, trial in enumerate(top, 1):
            tag = ("best",) if rank == 1 else (("odd",) if rank % 2 == 1 else ("even",))
            vals = [rank, f"{trial.value:.6f}"] + [
                f"{trial.params.get(p, ''):.4g}" if isinstance(trial.params.get(p), float)
                else str(trial.params.get(p, "")) for p in param_names]
            tree.insert("", "end", values=vals, tags=tag)

    def _draw_param_importances(self, parent, study):
        for w in parent.winfo_children(): w.destroy()
        try:
            if len(study.trials) < 3:
                ctk.CTkLabel(parent, text="Need at least 3 completed trials.",
                             text_color=COLORS["orange"]).pack(pady=20)
                return
            plt.style.use("dark_background")
            ax = plot_param_importances(study)
            fig = ax.figure
            fig.set_size_inches(7, 3.5)
            fig.patch.set_facecolor(COLORS["bg_card"])
            ax.set_facecolor(COLORS["bg_card"])
            fig.tight_layout()
            self._embed_fig(parent, fig, "hpo_param_importances.png")
        except Exception as e:
            ctk.CTkLabel(parent, text=f"Importances error: {e}", text_color=COLORS["red"]).pack(pady=20)

    def _draw_parallel_coords(self, parent, study):
        for w in parent.winfo_children(): w.destroy()
        try:
            if len(study.trials) < 2:
                ctk.CTkLabel(parent, text="Need at least 2 trials.",
                             text_color=COLORS["orange"]).pack(pady=20)
                return
            plt.style.use("dark_background")
            ax = plot_parallel_coordinate(study)
            fig = ax.figure
            fig.set_size_inches(9, 3.5)
            fig.patch.set_facecolor(COLORS["bg_card"])
            fig.tight_layout()
            self._embed_fig(parent, fig, "hpo_parallel_coords.png")
        except Exception as e:
            ctk.CTkLabel(parent, text=f"Parallel coords error: {e}", text_color=COLORS["red"]).pack(pady=20)

    def _draw_contour_plot(self, parent, study):
        for w in parent.winfo_children(): w.destroy()
        try:
            if len(study.trials) < 3:
                ctk.CTkLabel(parent, text="Need at least 3 trials.",
                             text_color=COLORS["orange"]).pack(pady=20)
                return
            px, py = self.contour_x_var.get(), self.contour_y_var.get()
            if px == py:
                ctk.CTkLabel(parent, text="Select two different parameters.",
                             text_color=COLORS["orange"]).pack(pady=20)
                return
            plt.style.use("dark_background")
            ax = plot_contour(study, params=[px, py])
            fig = ax.figure
            fig.set_size_inches(7, 4)
            fig.patch.set_facecolor(COLORS["bg_card"])
            fig.tight_layout()
            self._embed_fig(parent, fig, "hpo_contour.png")
        except Exception as e:
            ctk.CTkLabel(parent, text=f"Contour error: {e}", text_color=COLORS["red"]).pack(pady=20)

    # ── Optuna thread (dispatches by algo) ────────────────────────────────────

    def _run_optuna_thread(self, c):
        algo = c.get("algo", "Neural Network")
        self.q.put(f"[START] {algo} optimization — {c['trials']} trials...")
        study = optuna.create_study(direction="minimize")

        if algo == "Neural Network":
            objective = self._make_nn_objective(c)
        elif algo == "XGBoost":
            objective = self._make_xgb_objective(c)
        elif algo == "Random Forest":
            objective = self._make_rf_objective(c)
        elif algo == "Gaussian Process":
            objective = self._make_gpr_objective(c)
        else:
            self.q.put(f"ERROR: Unknown algorithm {algo}")
            self.q.put("DONE")
            return

        try:
            for i in range(c["trials"]):
                study.optimize(objective, n_trials=1, show_progress_bar=False)
                best = study.best_trial
                val = study.trials[-1].value
                self.q.put(f"Trial {i+1:>3}/{c['trials']} | Loss: {val:.6f} | Best: {best.value:.6f}")
            set_state("best_params", study.best_params)
            set_state("optuna_study", study)
            self.q.put(f"\nBest Params: {study.best_params}")
        except Exception as e:
            self.q.put(f"ERROR: {str(e)}")

        self.q.put("DONE")

    def _make_nn_objective(self, c):
        def objective(trial):
            num_layers = trial.suggest_int("n_layers", c["min_l"], c["max_l"])
            neurons = trial.suggest_int("neurons", c["min_u"], c["max_u"], step=8)
            dropout = trial.suggest_float("dropout", c["min_d"], c["max_d"])
            l1 = trial.suggest_float("l1", 0.0, c["max_l1"]) if c["max_l1"] > 0 else 0.0
            l2 = trial.suggest_float("l2", 0.0, c["max_l2"]) if c["max_l2"] > 0 else 0.0
            act = trial.suggest_categorical("activation", c["act_list"]) if len(c["act_list"]) > 1 else c["act_list"][0]
            lr = trial.suggest_float("lr", c["min_lr"], c["max_lr"], log=True)
            bs = trial.suggest_categorical("batch_size", c["bs_list"]) if len(c["bs_list"]) > 1 else c["bs_list"][0]
            opt = trial.suggest_categorical("optimizer", c["opt_list"]) if len(c["opt_list"]) > 1 else c["opt_list"][0]
            return _nn_train_eval(
                c["input_dim"], num_layers, neurons, act, c["out_act"], dropout, l1, l2,
                c["X_train"], c["y_train"], c["X_val"], c["y_val"], lr, opt, c["loss"],
                bs, c["epochs"], c["es_pat"], c["es_del"], c["rlr_factor"], c["rlr_pat"], c["rlr_min"])
        return objective

    def _make_xgb_objective(self, c):
        def objective(trial):
            from xgboost import XGBRegressor
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.metrics import mean_squared_error

            n_est = trial.suggest_int("n_estimators", c["xgb_min_est"], c["xgb_max_est"], step=50)
            depth = trial.suggest_int("max_depth", c["xgb_min_depth"], c["xgb_max_depth"])
            lr = trial.suggest_float("lr", c["xgb_min_lr"], c["xgb_max_lr"], log=True)
            sub = trial.suggest_float("subsample", c["xgb_min_sub"], c["xgb_max_sub"])
            col = trial.suggest_float("colsample_bytree", c["xgb_min_col"], c["xgb_max_col"])
            alpha = trial.suggest_float("reg_alpha", 1e-8, c["xgb_max_alpha"], log=True)
            lam = trial.suggest_float("reg_lambda", 1e-8, c["xgb_max_lambda"], log=True)

            n_out = c["y_train"].shape[1] if c["y_train"].ndim > 1 else 1
            base = XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                subsample=sub, colsample_bytree=col, reg_alpha=alpha,
                                reg_lambda=lam, verbosity=0, n_jobs=-1)
            if n_out == 1:
                base.fit(c["X_train"], c["y_train"].ravel(),
                         eval_set=[(c["X_val"], c["y_val"].ravel())],
                         verbose=False)
                y_pred = base.predict(c["X_val"])
                return mean_squared_error(c["y_val"].ravel(), y_pred)
            else:
                model = MultiOutputRegressor(base, n_jobs=1)
                model.fit(c["X_train"], c["y_train"])
                y_pred = model.predict(c["X_val"])
                return mean_squared_error(c["y_val"], y_pred)
        return objective

    def _make_rf_objective(self, c):
        def objective(trial):
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error

            n_est = trial.suggest_int("n_estimators", c["rf_min_est"], c["rf_max_est"], step=50)
            depth = trial.suggest_int("max_depth", c["rf_min_depth"], c["rf_max_depth"])
            split = trial.suggest_int("min_samples_split", c["rf_min_split"], c["rf_max_split"])
            leaf = trial.suggest_int("min_samples_leaf", c["rf_min_leaf"], c["rf_max_leaf"])

            n_out = c["y_train"].shape[1] if c["y_train"].ndim > 1 else 1
            model = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                         min_samples_split=split, min_samples_leaf=leaf,
                                         n_jobs=-1, random_state=42)
            y_tr = c["y_train"].ravel() if n_out == 1 else c["y_train"]
            model.fit(c["X_train"], y_tr)
            y_pred = model.predict(c["X_val"])
            y_val = c["y_val"].ravel() if n_out == 1 else c["y_val"]
            return mean_squared_error(y_val, y_pred)
        return objective

    def _make_gpr_objective(self, c):
        def objective(trial):
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.metrics import mean_squared_error
            from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel

            kernels_list = c["gpr_kernels"] if len(c["gpr_kernels"]) > 1 else c["gpr_kernels"]
            kernel_name = trial.suggest_categorical("kernel", kernels_list)
            alpha = trial.suggest_float("alpha", c["gpr_min_alpha"], c["gpr_max_alpha"], log=True)
            n_restarts = trial.suggest_int("n_restarts", c["gpr_min_restarts"], c["gpr_max_restarts"])

            k_map = {"RBF": ConstantKernel(1.0) * RBF(1.0),
                     "Matern": ConstantKernel(1.0) * Matern(nu=1.5),
                     "RationalQuadratic": ConstantKernel(1.0) * RationalQuadratic()}
            kernel = k_map.get(kernel_name, ConstantKernel(1.0) * RBF(1.0))

            n_out = c["y_train"].shape[1] if c["y_train"].ndim > 1 else 1
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                           n_restarts_optimizer=n_restarts, normalize_y=True)
            if n_out == 1:
                gpr.fit(c["X_train"], c["y_train"].ravel())
                y_pred = gpr.predict(c["X_val"])
                return mean_squared_error(c["y_val"].ravel(), y_pred)
            else:
                model = MultiOutputRegressor(gpr, n_jobs=1)
                model.fit(c["X_train"], c["y_train"])
                y_pred = model.predict(c["X_val"])
                return mean_squared_error(c["y_val"], y_pred)
        return objective
