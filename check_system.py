
import os
import sys
import importlib

def check_file(path):
    exists = os.path.exists(path)
    print(f"[{'OK' if exists else 'MISSING'}] File: {path}")
    return exists

def check_module(module_name):
    try:
        importlib.import_module(module_name)
        print(f"[OK] Module: {module_name}")
        return True
    except ImportError as e:
        print(f"[MISSING] Module: {module_name} ({e})")
        return False


def log(msg):
    print(msg)
    with open("system_report.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def main():
    # Clear report file
    with open("system_report.txt", "w", encoding="utf-8") as f:
        f.write("=== System Check ===\n")
    
    print("=== System Check ===")
    
    # 1. Check Files
    log("\n--- Checking Critical Files ---")
    files_to_check = [
        "app_integreted.py",
        "combined_inference.py",
        "best_fish_model.pt",
        "fish_freshness_model.pkl",
        "species_encoder.pkl",
        "freshness_encoder.pkl",
        "tabular_label_encoders.pkl",
        "schema.sql",
        ".env"
    ]
    
    all_files_ok = True
    for f in files_to_check:
        exists = os.path.exists(f)
        log(f"[{'OK' if exists else 'MISSING'}] File: {f}")
        if not exists:
            all_files_ok = False
            
    # 2. Check Modules
    log("\n--- Checking Python Dependencies ---")
    modules_to_check = [
        "streamlit",
        "pandas",
        "numpy",
        "PIL", # Pillow
        "plotly",
        "psycopg2",
        "torch",
        "torchvision",
        "xgboost",
        "cv2", # opencv-python
        "dotenv", # python-dotenv
        "qrcode",
        "twilio"
    ]
    
    all_modules_ok = True
    for m in modules_to_check:
        try:
            importlib.import_module(m)
            log(f"[OK] Module: {m}")
        except ImportError as e:
            log(f"[MISSING] Module: {m} ({e})")
            all_modules_ok = False

    # 3. Check Model Loading (if modules and files exist)
    if all_files_ok and all_modules_ok:
        log("\n--- Checking Model Loading ---")
        try:
            sys.path.append(".")
            from combined_inference import CombinedFishPredictor
            log("Attempting to initialize CombinedFishPredictor...")
            predictor = CombinedFishPredictor()
            if predictor.cnn_model and predictor.tabular_model:
                log("[OK] Models loaded successfully.")
            else:
                log("[WARNING] Predictor initialized but some models failed to load.")
        except Exception as e:
            log(f"[ERROR] Failed to load models: {e}")
    else:
        log("\n[SKIP] Skipping model loading check due to missing files or modules.")

    log("\n=== Check Complete ===")

if __name__ == "__main__":
    main()
