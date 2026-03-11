import customtkinter as ctk
import threading
import queue
import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from modules.model_builder import build_surrogate_model, get_keras_loss, get_keras_optimizer
from utils.theme import COLORS, FONTS
from utils.state import get_state, set_state

optuna.logging.set_verbosity(optuna.logging.WARNING)

def _train_eval(input_dim, layers_cfg, X_train, y_train, X_val, y_val, lr, batch_size, epochs, loss_name):
    """Quick train + eval for HPO trial. Returns best val loss."""
    model = build_surrogate_model(input_dim, 1, layers_cfg)
    criterion = get_keras_loss(loss_name)
    optimizer = get_keras_optimizer("Adam", lr)
    
    model.compile(optimizer=optimizer, loss=criterion)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop], verbose=0
    )
    return min(history.history['val_loss'])

class HyperoptFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        self.header = ctk.CTkLabel(self, text="HYPERPARAMETER OPTIMIZATION 🔍", font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(30, 20), sticky="w", padx=30)
        
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

    def _show_blocked(self, message):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.built_ui = False
        lbl = ctk.CTkLabel(self.content_frame, text=f"[ BLOCKED ]\n{message}", font=FONTS["header"], text_color=COLORS["red"])
        lbl.grid(row=0, column=0, pady=50, padx=20)

    def _build_ui(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
        # Strategy
        c1 = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        c1.grid(row=0, column=0, columnspan=2, sticky="ew", pady=10, padx=20)
        
        ctk.CTkLabel(c1, text="Optimization Method").grid(row=0, column=0, padx=20, pady=10)
        self.strat_var = ctk.StringVar(value="Optuna (TPE)")
        ctk.CTkComboBox(c1, values=["Optuna (TPE)", "Random Search", "Grid Search"], variable=self.strat_var).grid(row=0, column=1, padx=20, pady=10)
        
        # Search Space
        ss = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        ss.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10, padx=20)
        ss.grid_columnconfigure((0,1,2,3), weight=1)
        
        # Row 1
        ctk.CTkLabel(ss, text="Min Layers").grid(row=0, column=0, pady=5)
        self.min_l = ctk.CTkEntry(ss); self.min_l.insert(0, "1"); self.min_l.grid(row=0, column=1, pady=5)
        
        ctk.CTkLabel(ss, text="Max Layers").grid(row=0, column=2, pady=5)
        self.max_l = ctk.CTkEntry(ss); self.max_l.insert(0, "4"); self.max_l.grid(row=0, column=3, pady=5)
        
        # Row 2
        ctk.CTkLabel(ss, text="Min Neurons").grid(row=1, column=0, pady=5)
        self.min_u = ctk.CTkEntry(ss); self.min_u.insert(0, "16"); self.min_u.grid(row=1, column=1, pady=5)
        
        ctk.CTkLabel(ss, text="Max Neurons").grid(row=1, column=2, pady=5)
        self.max_u = ctk.CTkEntry(ss); self.max_u.insert(0, "256"); self.max_u.grid(row=1, column=3, pady=5)

        # Configs
        ctk.CTkLabel(ss, text="Trials").grid(row=2, column=0, pady=5)
        self.trials_e = ctk.CTkEntry(ss); self.trials_e.insert(0, "15"); self.trials_e.grid(row=2, column=1, pady=5)
        
        ctk.CTkLabel(ss, text="Epochs p. trial").grid(row=2, column=2, pady=5)
        self.ep_e = ctk.CTkEntry(ss); self.ep_e.insert(0, "30"); self.ep_e.grid(row=2, column=3, pady=5)

        # Run block
        self.run_btn = ctk.CTkButton(self.content_frame, text="⚡ RUN OPTIMIZATION", height=40, command=self.start_optimization)
        self.run_btn.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Terminal Log
        self.log_box = ctk.CTkTextbox(self.content_frame, height=200, font=FONTS["code"], fg_color="#0D1117", text_color=COLORS["text"])
        self.log_box.grid(row=3, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
        self.log_box.configure(state="disabled")

    def _log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.yview("end")
        self.log_box.configure(state="disabled")

    def start_optimization(self):
        if self.is_running: return
        self.is_running = True
        self.run_btn.configure(state="disabled", text="RUNNING...")
        self.log_box.configure(state="normal"); self.log_box.delete("0.0", "end"); self.log_box.configure(state="disabled")
        
        # Parse inputs
        try:
            cfg = {
                "strat": self.strat_var.get(),
                "min_l": int(self.min_l.get()), "max_l": int(self.max_l.get()),
                "min_u": int(self.min_u.get()), "max_u": int(self.max_u.get()),
                "trials": int(self.trials_e.get()), "epochs": int(self.ep_e.get()),
                "input_dim": get_state("X_train").shape[1],
                "X_train": get_state("X_train"), "y_train": get_state("y_train"),
                "X_val": get_state("X_val"), "y_val": get_state("y_val")
            }
        except:
            self._log("ERROR: Invalid numeric inputs.")
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
                    self._log("\n[ OK ] Optimization Complete.")
                    break
                else:
                    self._log(msg)
        except queue.Empty:
            if self.is_running:
                self.after(100, self.process_queue)

    def _run_optuna_thread(self, c):
        self.q.put(f"[START] Beginning {c['strat']}...")
        
        study = optuna.create_study(direction="minimize")
        activations = ["ReLU", "LeakyReLU", "ELU", "Tanh"]
        
        def objective(trial):
            n_layers = trial.suggest_int("n_layers", c["min_l"], c["max_l"])
            layers_cfg = []
            for i in range(n_layers):
                u = trial.suggest_int(f"units_{i}", c["min_u"], c["max_u"], step=8)
                d = trial.suggest_float(f"dropout_{i}", 0.0, 0.5, step=0.1)
                act = trial.suggest_categorical(f"act_{i}", activations)
                layers_cfg.append({"units": u, "activation": act, "dropout": d})

            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            bs = trial.suggest_categorical("batch_size", [32, 64, 128])

            val_loss = _train_eval(c["input_dim"], layers_cfg, c["X_train"], c["y_train"], c["X_val"], c["y_val"], lr, bs, c["epochs"], "MeanSquaredError")
            return val_loss

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
