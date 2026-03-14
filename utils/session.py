"""
Session persistence for Surrogate Builder.
Saves / loads the full AppState to / from a .smproj zip file.
"""

import json
import os
import zipfile
import tempfile
import datetime
import shutil

import numpy as np


# ── constants ────────────────────────────────────────────────────────────────

SCHEMA_VERSION = 1
RECENT_FILE = os.path.join(os.path.expanduser("~"), ".smtraining", "recent_sessions.json")


# ── JSON serializer ───────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    return str(obj)


def _write_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


# ── save ──────────────────────────────────────────────────────────────────────

def save_session(path: str) -> None:
    """Serialize the current AppState to a .smproj zip at *path*.

    Raises RuntimeError with a user-readable message on failure.
    The zip is built atomically: a temp dir is packed then moved.
    """
    from utils.state import get_state, set_state
    import joblib
    import pandas as pd

    session_name = (
        get_state("session_name")
        or os.path.splitext(os.path.basename(path))[0]
    )
    now = datetime.datetime.now().isoformat(timespec="seconds")

    tmpdir = tempfile.mkdtemp()
    try:
        # ── manifest / session.json ───────────────────────────────────────
        surrogate = get_state("surrogate_model")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "session_name":   session_name,
            "created_at":     now,   # overwritten below if updating existing file
            "saved_at":       now,
            "algorithm":      surrogate.algo_name if surrogate else None,
            "app_state": {
                "data_loaded":       get_state("data_loaded",      False),
                "preprocessed":      get_state("preprocessed",     False),
                "model_ready":       get_state("model_ready",      False),
                "trained":           get_state("trained",          False),
                "input_columns":     get_state("input_columns",    []),
                "output_column":     get_state("output_column",    []),
                "train_losses":      get_state("train_losses",     []),
                "val_losses":        get_state("val_losses",       []),
                "training_metrics":  get_state("training_metrics", {}),
                "best_params":       get_state("best_params"),
                "layers_config":     get_state("layers_config"),
                "model_config":      get_state("model_config"),
                "model_params_count": get_state("model_params_count", 0),
            },
            "preprocessing_config": get_state("preprocessing_config"),
        }

        # Preserve original created_at when re-saving an existing file
        if os.path.isfile(path):
            try:
                with zipfile.ZipFile(path, "r") as zin:
                    existing = json.loads(zin.read("session.json"))
                    manifest["created_at"] = existing.get("created_at", now)
            except Exception:
                pass

        _write_json(os.path.join(tmpdir, "session.json"), manifest)

        # ── data/ ─────────────────────────────────────────────────────────
        df: "pd.DataFrame | None" = get_state("df")
        if df is not None:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            df.to_csv(os.path.join(data_dir, "dataframe.csv"), index=False)

        # ── splits/ ───────────────────────────────────────────────────────
        split_arrays = {
            k: get_state(k)
            for k in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test")
            if get_state(k) is not None
        }
        if split_arrays:
            splits_dir = os.path.join(tmpdir, "splits")
            os.makedirs(splits_dir)
            np.savez_compressed(os.path.join(splits_dir, "arrays.npz"), **split_arrays)

        # ── scalers/ ──────────────────────────────────────────────────────
        scaler_X = get_state("scaler_X")
        scaler_y = get_state("scaler_y")
        if scaler_X is not None or scaler_y is not None:
            scalers_dir = os.path.join(tmpdir, "scalers")
            os.makedirs(scalers_dir)
            if scaler_X is not None:
                joblib.dump(scaler_X, os.path.join(scalers_dir, "scaler_X.pkl"))
            if scaler_y is not None:
                joblib.dump(scaler_y, os.path.join(scalers_dir, "scaler_y.pkl"))

        # ── pca/ ──────────────────────────────────────────────────────────
        pca_X = get_state("pca_X")
        pca_y = get_state("pca_y")
        if pca_X is not None or pca_y is not None:
            pca_dir = os.path.join(tmpdir, "pca")
            os.makedirs(pca_dir)
            if pca_X is not None:
                joblib.dump(pca_X, os.path.join(pca_dir, "pca_X.pkl"))
            if pca_y is not None:
                joblib.dump(pca_y, os.path.join(pca_dir, "pca_y.pkl"))

        # ── model/ ────────────────────────────────────────────────────────
        if surrogate is not None and get_state("trained", False):
            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(model_dir)
            surrogate.save(model_dir)
            with open(os.path.join(model_dir, "algo.txt"), "w") as f:
                f.write(surrogate.algo_name)

        # ── zip everything ────────────────────────────────────────────────
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
            for root, _, files in os.walk(tmpdir):
                for fname in files:
                    abs_p = os.path.join(root, fname)
                    zout.write(abs_p, os.path.relpath(abs_p, tmpdir))

        # Update session state
        set_state("session_path",           path)
        set_state("session_name",           session_name)
        set_state("session_unsaved",        False)
        set_state("session_last_saved_at",  now)

        _add_to_recent(session_name, path, now)

    except Exception as e:
        raise RuntimeError(f"Session save failed: {e}") from e
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── load ──────────────────────────────────────────────────────────────────────

def load_session(path: str) -> dict:
    """Deserialize a .smproj into AppState.

    Returns the full session dict (including preprocessing_config etc.) so the
    caller can restore UI widget values.
    Raises RuntimeError on version mismatch or corrupt archive.
    """
    from utils.state import set_state, AppState, init_all_defaults
    from models import ALGORITHM_REGISTRY
    import joblib
    import pandas as pd

    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmpdir)

        # ── session.json ──────────────────────────────────────────────────
        sess_path = os.path.join(tmpdir, "session.json")
        if not os.path.isfile(sess_path):
            raise RuntimeError("Not a valid .smproj file (session.json missing).")
        with open(sess_path) as f:
            session = json.load(f)

        if session.get("schema_version", 1) > SCHEMA_VERSION:
            raise RuntimeError(
                f"Session was saved with a newer version of Surrogate Builder "
                f"(schema v{session['schema_version']}). Please update the app."
            )

        # Reset AppState to clean defaults
        AppState._state.clear()
        init_all_defaults()

        # Restore scalar / list / dict keys from app_state block
        app_state = session.get("app_state", {})
        for k, v in app_state.items():
            set_state(k, v)

        set_state("preprocessing_config", session.get("preprocessing_config"))

        # ── data/ ─────────────────────────────────────────────────────────
        csv_path = os.path.join(tmpdir, "data", "dataframe.csv")
        if os.path.isfile(csv_path):
            set_state("df", pd.read_csv(csv_path))

        # ── splits/ ───────────────────────────────────────────────────────
        npz_path = os.path.join(tmpdir, "splits", "arrays.npz")
        if os.path.isfile(npz_path):
            sp = np.load(npz_path)
            for k in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
                if k in sp:
                    set_state(k, sp[k])

        # ── scalers/ ──────────────────────────────────────────────────────
        for name in ("scaler_X", "scaler_y"):
            pkl = os.path.join(tmpdir, "scalers", f"{name}.pkl")
            if os.path.isfile(pkl):
                set_state(name, joblib.load(pkl))

        # ── pca/ ──────────────────────────────────────────────────────────
        for name in ("pca_X", "pca_y"):
            pkl = os.path.join(tmpdir, "pca", f"{name}.pkl")
            if os.path.isfile(pkl):
                set_state(name, joblib.load(pkl))

        # ── model/ ────────────────────────────────────────────────────────
        model_dir = os.path.join(tmpdir, "model")
        if os.path.isdir(model_dir) and app_state.get("trained"):
            algo_txt = os.path.join(model_dir, "algo.txt")
            algo_name = session.get("algorithm", "Neural Network")
            if os.path.isfile(algo_txt):
                with open(algo_txt) as f:
                    algo_name = f.read().strip()
            try:
                cls = ALGORITHM_REGISTRY.get(algo_name)
                if cls:
                    surrogate = cls.load(model_dir)
                    set_state("surrogate_model", surrogate)
                    if hasattr(surrogate, "_model") and surrogate._model is not None:
                        set_state("model", surrogate._model)
                    # Signal model_builder to switch to restored algo
                    set_state("selected_algo", algo_name)
            except Exception as e:
                print(f"[Session] Could not restore model '{algo_name}': {e}")

        # Mark Results tab stale so it rebuilds on next visit
        set_state("results_stale", True)

        # Session metadata
        set_state("session_path",           path)
        set_state("session_name",           session.get("session_name", "Loaded Session"))
        set_state("session_unsaved",        False)
        set_state("session_last_saved_at",  session.get("saved_at", ""))

        _add_to_recent(
            session.get("session_name", os.path.splitext(os.path.basename(path))[0]),
            path,
            session.get("saved_at", ""),
        )

        return session

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Session load failed: {e}") from e
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── reset ─────────────────────────────────────────────────────────────────────

def reset_session() -> None:
    """Wipe all data-bearing AppState keys back to init_defaults values."""
    from utils.state import AppState, init_all_defaults
    AppState._state.clear()
    init_all_defaults()


# ── recent sessions registry ──────────────────────────────────────────────────

def _add_to_recent(name: str, path: str, saved_at: str, max_entries: int = 10) -> None:
    os.makedirs(os.path.dirname(RECENT_FILE), exist_ok=True)
    existing = load_recent_sessions()
    existing = [e for e in existing if e.get("path") != path]
    existing.insert(0, {"name": name, "path": path, "saved_at": saved_at})
    try:
        with open(RECENT_FILE, "w") as f:
            json.dump(existing[:max_entries], f, indent=2)
    except Exception:
        pass


def load_recent_sessions() -> list:
    """Return list of recently saved sessions (existing files only)."""
    if not os.path.isfile(RECENT_FILE):
        return []
    try:
        with open(RECENT_FILE) as f:
            entries = json.load(f)
        return [e for e in entries if os.path.isfile(e.get("path", ""))]
    except Exception:
        return []
