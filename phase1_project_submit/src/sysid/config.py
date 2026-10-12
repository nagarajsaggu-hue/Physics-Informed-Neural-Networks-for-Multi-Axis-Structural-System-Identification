"""Configuration loading.

Everything the pipeline needs comes from config.yaml plus the two metadata
files written by the ANSYS extraction. Nothing is hard-coded in the training
script, so a change of assumption is a change to one YAML file.
"""

import json
from pathlib import Path

import yaml


def project_root() -> Path:
    """Repository root, found by walking up from this file."""
    return Path(__file__).resolve().parents[2]


def load_config(path=None) -> dict:
    """Read config.yaml and resolve relative paths against the repo root."""
    root = project_root()
    path = Path(path) if path else root / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for key in ("csv", "meta", "modal"):
        p = cfg["data"].get(key)
        if p:
            cfg["data"][key] = str(root / p)
            
    # Ensure the dynamic output directory exists before returning
    out_dir = root / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["output"]["dir"] = str(out_dir)
    
    cfg["_root"] = str(root)
    return cfg


def load_json(path):
    """Read a JSON file, returning None if it is absent.

    The modal reference is optional - the pipeline still runs without it,
    it just cannot produce that row of the comparison table.
    """
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)
