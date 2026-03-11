import customtkinter as ctk
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
import threading
import queue
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from modules.model_builder import build_surrogate_model, get_keras_loss, get_keras_optimizer
from utils.theme import COLORS, FONTS
from utils.state import get_state, set_state


class TkinterUpdateCallback(Callback):
    """Custom Keras callback to send metrics to Tkinter via Queue."""
    def __init__(self, q, epochs, X_val, y_val):
        super().__init__()
        self.q = q
        self.epochs = epochs
        self.X_val = X_val
        self.y_val = y_val
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss', 0.0)
        train_loss = logs.get('loss', 0.0)
        
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Compute R² manually
        val_pred = self.model.predict(self.X_val, verbose=0).flatten()
        y_val_flat = self.y_val.flatten()
        ss_res = np.sum((y_val_flat - val_pred) ** 2)
        ss_tot = np.sum((y_val_flat - np.mean(y_val_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Send to queue
        self.q.put({
            "type": "epoch",
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "r2": r2,
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses)
        })


class TrainingFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        # Header
        self.header = ctk.CTkLabel(self, text="TRAINING DASHBOARD 🚀", font=FONTS["title"], text_color=COLORS["cyan"])
        self.header.grid(row=0, column=0, pady=(30, 20), sticky="w", padx=30)
        
        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        
        self.built_ui = False
        self.is_running = False
        self.q = queue.Queue()

    def on_show(self):
        if not get_state("model_ready"):
            self._show_blocked("Configure a model first.\n← Go to 'Model Builder'")
            return
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

        # Early Stopping Frame
        es_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"])
        es_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        es_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        self.use_es = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(es_frame, text="Enable Early Stopping", variable=self.use_es).grid(row=0, column=0, padx=20, pady=15)
        
        ctk.CTkLabel(es_frame, text="Patience").grid(row=0, column=1)
        self.es_pat_e = ctk.CTkEntry(es_frame, width=80)
        self.es_pat_e.insert(0, "20")
        self.es_pat_e.grid(row=1, column=1, padx=20, pady=(0,15))
        
        ctk.CTkLabel(es_frame, text="Min Delta").grid(row=0, column=2)
        self.es_del_e = ctk.CTkEntry(es_frame, width=80)
        self.es_del_e.insert(0, "0.00001")
        self.es_del_e.grid(row=1, column=2, padx=20, pady=(0,15))

        # Controls & Metrics
        ctrl_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        ctrl_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        
        self.train_btn = ctk.CTkButton(ctrl_frame, text="⚡ START TRAINING", height=40, command=self.start_training)
        self.train_btn.pack(side="left", padx=10)
        
        self.status_lbl = ctk.CTkLabel(ctrl_frame, text="Ready.", text_color=COLORS["text_dim"])
        self.status_lbl.pack(side="left", padx=20)
        
        self.pb = ctk.CTkProgressBar(ctrl_frame, width=300)
        self.pb.set(0)
        self.pb.pack(side="right", padx=10)

        # Matplotlib Plot
        self.plot_frame = ctk.CTkFrame(self.content_frame, fg_color="#000", height=400)
        self.plot_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        self.plot_frame.pack_propagate(False)
        
        # Terminal Logging
        self.log_box = ctk.CTkTextbox(self.content_frame, height=200, font=FONTS["code"], fg_color="#0D1117", text_color=COLORS["text"])
        self.log_box.grid(row=3, column=0, sticky="ew", padx=20, pady=20)
        self.log_box.configure(state="disabled")

    def _log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.yview("end")
        self.log_box.configure(state="disabled")

    def start_training(self):
        if self.is_running: return
        self.is_running = True
        self.train_btn.configure(state="disabled", text="TRAINING...")
        self.log_box.configure(state="normal"); self.log_box.delete("0.0", "end"); self.log_box.configure(state="disabled")
        
        cfg = get_state("model_config")
        X_train, y_train = get_state("X_train"), get_state("y_train")
        X_val, y_val = get_state("X_val"), get_state("y_val")
        
        es_pat = int(self.es_pat_e.get())
        es_del = float(self.es_del_e.get())
        use_es = self.use_es.get()
        
        # Start Thread
        t = threading.Thread(target=self._run_training_thread, args=(cfg, X_train, y_train, X_val, y_val, use_es, es_pat, es_del))
        t.start()
        
        # Init plots
        self._init_plot()
        
        self.after(100, self.process_queue)

    def _init_plot(self):
        for w in self.plot_frame.winfo_children():
            w.destroy()
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.fig.patch.set_facecolor(COLORS["bg_card"])
        self.ax.set_facecolor(COLORS["bg_card"])
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        
        self.line_train, = self.ax.plot([], [], color=COLORS['cyan'], label="Train Loss")
        self.line_val, = self.ax.plot([], [], color=COLORS['orange'], label="Val Loss")
        self.ax.legend()
        
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
                if isinstance(msg, dict):
                    if msg["type"] == "epoch":
                        ep = msg["epoch"]
                        tl = msg["train_loss"]
                        vl = msg["val_loss"]
                        r2 = msg["r2"]
                        
                        self.pb.set(ep / get_state("model_config")["epochs"])
                        self.status_lbl.configure(text=f"Epoch {ep} | Train: {tl:.6f} | Val: {vl:.6f} | R²: {r2:.4f}")
                        
                        log_line = f"[{ep:>4}] train={tl:.6f} val={vl:.6f} R²={r2:.4f}"
                        self._log(log_line)
                        
                        # Only redraw plot occasionally to prevent GUI freeze
                        if ep % 3 == 0 or ep == get_state("model_config")["epochs"]:
                            self._update_plot(msg["train_losses"], msg["val_losses"])
                            
                elif isinstance(msg, str):
                    if msg == "DONE":
                        self.is_running = False
                        self.train_btn.configure(state="normal", text="⚡ START TRAINING")
                        self._log("\n[ OK ] Training Complete.")
                        self.status_lbl.configure(text="✓ Training Complete.", text_color=COLORS["green"])
                        break
                    else:
                        self._log(msg)
        except queue.Empty:
            if self.is_running:
                self.after(50, self.process_queue)

    def _run_training_thread(self, cfg, X_train, y_train, X_val, y_val, use_es, es_pat, es_del):
        try:
            model = build_surrogate_model(cfg["input_dim"], 1, cfg["layers"])
            criterion = get_keras_loss(cfg["loss"])
            optimizer = get_keras_optimizer(cfg["optimizer"], cfg["lr"])
            model.compile(optimizer=optimizer, loss=criterion)

            epochs = cfg["epochs"]
            st_callback = TkinterUpdateCallback(self.q, epochs, X_val, y_val)
            callbacks = [st_callback]
            
            if use_es:
                es = EarlyStopping(monitor='val_loss', patience=es_pat, min_delta=es_del, restore_best_weights=True, verbose=1)
                callbacks.append(es)

            self.q.put("Starting Training...")
            start_time = time.time()

            # Keras Blocks thread
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=cfg["batch_size"],
                callbacks=callbacks,
                verbose=0
            )

            # Finished
            elapsed = time.time() - start_time
            t_losses = st_callback.train_losses
            v_losses = st_callback.val_losses
            
            # Final R2
            val_pred = model.predict(X_val, verbose=0).flatten()
            y_val_flat = y_val.flatten()
            ss_res = np.sum((y_val_flat - val_pred) ** 2)
            ss_tot = np.sum((y_val_flat - np.mean(y_val_flat)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Save state
            set_state("model", model)
            set_state("trained", True)
            set_state("train_losses", t_losses)
            set_state("val_losses", v_losses)
            set_state("training_metrics", {
                "best_val_loss": min(v_losses) if v_losses else 0,
                "final_train_loss": t_losses[-1] if t_losses else 0,
                "r2": r2,
                "epochs_run": len(t_losses),
                "elapsed_seconds": elapsed
            })
            
            self._update_plot(t_losses, v_losses)

            summary = f"""
┌──────────────────────────────────────────┐
│  TRAINING SUMMARY                        │
├──────────────────────────────────────────┤
│  Epochs run   : {len(t_losses):<25}│
│  Best val loss: {min(v_losses):<25.6f}│
│  Final R²     : {r2:<25.4f}│
│  Time (s)     : {elapsed:<25.1f}│
└──────────────────────────────────────────┘"""
            self.q.put(summary)
            
        except Exception as e:
            self.q.put(f"ERROR: {str(e)}")
            
        self.q.put("DONE")
