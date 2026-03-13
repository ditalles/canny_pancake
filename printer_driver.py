"""Printer communication and job management for Marlin/GRBL CNC printers."""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class PrintJob:
    """Represents a single print job."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "queued"  # queued | printing | done | error
    progress: float = 0.0  # 0.0 to 100.0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    gcode: str = ""


class PrinterConnection:
    """Serial connection to a Marlin/GRBL printer."""

    def __init__(self, port=None, baud=None, timeout=None):
        self.port = port or config.PRINTER_PORT
        self.baud = baud or config.PRINTER_BAUD
        self.timeout = timeout or config.PRINTER_TIMEOUT
        self._serial = None
        self._lock = threading.Lock()

    def connect(self):
        """Open serial connection to the printer.

        Returns True on success, False on failure.
        """
        try:
            import serial
        except ImportError:
            logger.error("pyserial not installed. Run: pip install pyserial")
            return False

        ports_to_try = [self.port, "/dev/ttyUSB0", "/dev/ttyACM0"]
        for port in ports_to_try:
            try:
                self._serial = serial.Serial(port, self.baud, timeout=self.timeout)
                time.sleep(2)  # Wait for Marlin to initialize
                # Flush startup messages
                while self._serial.in_waiting:
                    self._serial.readline()
                logger.info("Connected to printer on %s", port)
                self.port = port
                return True
            except (serial.SerialException, OSError):
                continue

        logger.error("Failed to connect to printer on any port")
        return False

    def disconnect(self):
        """Close serial connection."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._serial = None
            logger.info("Disconnected from printer")

    @property
    def is_connected(self):
        return self._serial is not None and self._serial.is_open

    def send_line(self, line):
        """Send a single G-code line and wait for 'ok' response.

        Args:
            line: G-code command string

        Returns:
            Response string from printer

        Raises:
            RuntimeError: If not connected or timeout waiting for response
        """
        if not self.is_connected:
            raise RuntimeError("Printer not connected")

        # Strip comments and whitespace
        line = line.split(";")[0].strip()
        if not line:
            return "ok"

        with self._lock:
            self._serial.write((line + "\n").encode())
            response = self._serial.readline().decode().strip()

            if not response:
                raise RuntimeError(f"Timeout waiting for response to: {line}")

            return response

    def send_gcode(self, gcode, progress_callback=None):
        """Send complete G-code to the printer line by line.

        Args:
            gcode: Complete G-code string
            progress_callback: Called with (current_line, total_lines) after each line
        """
        lines = [l.strip() for l in gcode.split("\n") if l.strip() and not l.strip().startswith(";")]
        total = len(lines)

        for i, line in enumerate(lines):
            self.send_line(line)
            if progress_callback:
                progress_callback(i + 1, total)


class DryRunConnection:
    """Mock printer connection that logs G-code instead of sending it."""

    def __init__(self):
        self.port = "DRY_RUN"
        self._lines_sent = []

    def connect(self):
        logger.info("Dry-run mode: no printer connection")
        return True

    def disconnect(self):
        pass

    @property
    def is_connected(self):
        return True

    def send_line(self, line):
        line = line.split(";")[0].strip()
        if line:
            self._lines_sent.append(line)
        return "ok"

    def send_gcode(self, gcode, progress_callback=None):
        lines = [l.strip() for l in gcode.split("\n") if l.strip() and not l.strip().startswith(";")]
        total = len(lines)
        for i, line in enumerate(lines):
            self.send_line(line)
            if progress_callback:
                progress_callback(i + 1, total)
            time.sleep(0.01)  # Simulate print time


class PrintManager:
    """Manages print jobs and printer communication."""

    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()
        self._connection = None

    def get_connection(self):
        """Get or create the printer connection."""
        if self._connection is None:
            if config.DRY_RUN:
                self._connection = DryRunConnection()
            else:
                self._connection = PrinterConnection()
        return self._connection

    def connect_printer(self):
        """Connect to the printer. Returns True on success."""
        conn = self.get_connection()
        if not conn.is_connected:
            return conn.connect()
        return True

    @property
    def printer_status(self):
        """Return printer connection status dict."""
        conn = self.get_connection()
        return {
            "connected": conn.is_connected,
            "port": conn.port,
            "dry_run": config.DRY_RUN,
        }

    def create_job(self, gcode):
        """Create a new print job. Returns the job."""
        job = PrintJob(gcode=gcode)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get_job(self, job_id):
        """Get a job by ID. Returns None if not found."""
        return self._jobs.get(job_id)

    def start_job(self, job_id):
        """Start printing a job in a background thread.

        Returns True if started, False if job not found or printer unavailable.
        """
        job = self.get_job(job_id)
        if not job:
            return False

        conn = self.get_connection()
        if not conn.is_connected:
            if not conn.connect():
                job.status = "error"
                job.error_message = "Cannot connect to printer"
                return False

        thread = threading.Thread(target=self._run_job, args=(job, conn), daemon=True)
        thread.start()
        return True

    def _run_job(self, job, connection):
        """Execute a print job (runs in background thread)."""
        job.status = "printing"
        job.progress = 0.0

        def on_progress(current, total):
            job.progress = (current / total) * 100.0

        try:
            connection.send_gcode(job.gcode, progress_callback=on_progress)
            job.status = "done"
            job.progress = 100.0
            logger.info("Job %s completed", job.id)
        except Exception as e:
            job.status = "error"
            job.error_message = str(e)
            logger.error("Job %s failed: %s", job.id, e)
