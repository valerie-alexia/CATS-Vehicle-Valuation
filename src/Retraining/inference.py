"""Compatibility entry point: new UI and API share the v2 service."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.Retraining.service import load_active, predict_price, validate_payload
MOBIL_DIR = ROOT / 'Models/Mobil123'
MOTOR_DIR = ROOT / 'Models/Momotor'
SPEC_REF_PATH = ROOT / 'Data/reference/spec_mobil.csv'
