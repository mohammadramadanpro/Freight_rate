from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42

WEIGHT_CAP = 47_500.0
DISTANCE_FLAG_RATIO = 1.5

# Rate-per-mile bounds outside which a posted rate is treated as a suspected corruption.
# Used for reporting and for the outlier experiments only; never applied to validation rows.
SUSPECT_RPM_LOW = 1.0
SUSPECT_RPM_HIGH = 5.0

# Each fold trains on every month before the holdout month.
ROLLING_HOLDOUT_MONTHS = ["2025-07", "2025-08", "2025-09", "2025-10"]
