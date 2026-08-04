"""
Script: header_generator_mlp.py
Module role:
    Convert the latest quantized MLP TensorFlow Lite artifact into a C/C++
    header for firmware integration.

Technical summary:
    This script resolves the latest versioned quantized model, serializes the
    binary contents as a byte array, writes a guarded header file, and records
    the generated artifact in the versioned output structure.

Inputs:
    - Latest quantized TensorFlow Lite model artifact
    - Project path and versioning utilities from utils.global_utils.paths_mlp and
      utils.global_utils.versioning

Outputs:
    - environment_model_data_mlp.h
    - Version manifest for the generated header artifact

Notes:
    This script assumes the repository project structure and the referenced
    utility modules. The computational logic and export procedure are
    preserved.
"""
import os,sys 
from pathlib import Path

# --- Bootstrap: allows importing utils/ locally and in the runner ---
ROOT = os.environ.get("RUNNER_PROJECT_ROOT")
if not ROOT:
    HERE = Path(__file__).resolve()
    for base in [HERE, *HERE.parents, Path.cwd(), *Path.cwd().parents]:
        if (base / "utils").exists():
            ROOT = str(base); break
if ROOT and ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -----------------------------------------------------------------
from utils.global_utils.paths_mlp import PROJECT_ROOT, QUANTIZED_MODEL, HEADER_GENERATOR  # uses the project root in a stable manner
from utils.global_utils.versioning import create_versioned_dir, resolve_latest, update_latest, write_manifest

# === Versioned directories for the current execution ===
run_dir = create_versioned_dir(HEADER_GENERATOR, strategy="counter")
version_path = resolve_latest(QUANTIZED_MODEL)

def convert_tflite_to_header(tflite_model_path, output_header_path, variable_name="model_data"):
    """Serialize a TensorFlow Lite model file into a C/C++ header array."""
    with open(tflite_model_path, "rb") as f:
        model_data = f.read()

    header_guard = os.path.basename(output_header_path).replace(".", "_").upper()

    with open(output_header_path, "w") as f:
        f.write(f"#ifndef {header_guard}\n")
        f.write(f"#define {header_guard}\n\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"alignas(8) const unsigned char {variable_name}[] = {{")

        # Compact header generation: all bytes are written in a single line
        byte_array = ",".join(f"0x{b:02x}" for b in model_data)
        f.write(byte_array)

        f.write("};\n\n")
        f.write(f"const int {variable_name}_len = {len(model_data)};\n\n")
        f.write(f"#endif // {header_guard}\n")

# Usage:
convert_tflite_to_header(
    tflite_model_path= version_path/"environment_quantized_model_mlp.tflite",
    output_header_path=run_dir/"environment_model_data_mlp.h",
    variable_name="environment_model"
)

print("[INFO] Generated file: environment_model_data_mlp.h")

# === Post-execution: update 'latest' and manifest ===
try:
    update_latest(run_dir)
except Exception as _e:
    print("[WARN] Unable to update 'latest':", _e)
try:
    write_manifest(run_dir, run=str(run_dir))
except Exception as _e:
    print("[WARN] Unable to write manifest.json:", _e)