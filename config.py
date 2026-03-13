"""Central configuration for the Pancake Printer system."""

import os

# Printer serial connection
PRINTER_PORT = os.environ.get("PRINTER_PORT", "/dev/ttyUSB0")
PRINTER_BAUD = int(os.environ.get("PRINTER_BAUD", "115200"))
PRINTER_TIMEOUT = int(os.environ.get("PRINTER_TIMEOUT", "30"))

# Print bed dimensions (mm)
BED_WIDTH = float(os.environ.get("BED_WIDTH", "200.0"))
BED_HEIGHT = float(os.environ.get("BED_HEIGHT", "200.0"))

# G-code motion parameters
FEED_RATE_TRAVEL = int(os.environ.get("FEED_RATE_TRAVEL", "3000"))    # mm/min
FEED_RATE_DISPENSE = int(os.environ.get("FEED_RATE_DISPENSE", "600"))  # mm/min
EXTRUSION_RATE = float(os.environ.get("EXTRUSION_RATE", "0.05"))      # E per mm
Z_HEIGHT = float(os.environ.get("Z_HEIGHT", "1.0"))                   # nozzle height mm

# Tone settings
DEFAULT_NUM_TONES = int(os.environ.get("DEFAULT_NUM_TONES", "3"))

# Tool codes for each batter dispenser
TOOL_CODES = ["T0", "T1", "T2"]

# Dry-run mode (log G-code instead of sending to printer)
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"

# Flask
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5005"))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
