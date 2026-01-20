#!/usr/bin/env python3
"""
Deploy Watcher Script for Auto-Deploy

This script:
1. Starts the FastAPI server (server.py)
2. Monitors the restart_flag.txt file
3. When the flag is detected, restarts the server

Usage:
    python backend/deploy_watcher.py

The server process is managed by this watcher, so:
- Don't run server.py directly when using auto-deploy
- This script will keep the server running and restart it when needed
"""

import os
import sys
import time
import signal
import subprocess
import platform
from pathlib import Path
from datetime import datetime

# Configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
RESTART_FLAG_FILE = SCRIPT_DIR / "restart_flag.txt"
SERVER_SCRIPT = SCRIPT_DIR / "server.py"
CHECK_INTERVAL = 2  # seconds
LOG_FILE = BASE_DIR / "logs" / "deploy_watcher.log"

# Ensure logs directory exists
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(message: str):
    """Log message with timestamp to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")


def get_python_command():
    """Get the appropriate Python command for this platform."""
    if platform.system() == 'Windows':
        return [sys.executable]
    return ["python3"]


class ServerManager:
    """Manages the FastAPI server process."""

    def __init__(self):
        self.process = None
        self.running = True

    def start_server(self):
        """Start the server process."""
        if self.process and self.process.poll() is None:
            log("Server already running")
            return

        log("Starting server...")
        python_cmd = get_python_command()

        try:
            # Start uvicorn directly for better control
            self.process = subprocess.Popen(
                python_cmd + ["-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"],
                cwd=str(SCRIPT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )
            log(f"Server started with PID: {self.process.pid}")

            # Start a thread to read and log server output
            import threading
            def log_output():
                if self.process and self.process.stdout:
                    for line in iter(self.process.stdout.readline, ''):
                        if line:
                            print(f"[SERVER] {line.rstrip()}")
                    self.process.stdout.close()

            output_thread = threading.Thread(target=log_output, daemon=True)
            output_thread.start()

        except Exception as e:
            log(f"Failed to start server: {e}")
            raise

    def stop_server(self):
        """Stop the server process gracefully."""
        if not self.process:
            return

        log("Stopping server...")

        try:
            if platform.system() == 'Windows':
                # On Windows, use taskkill for cleaner shutdown
                subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(self.process.pid)],
                    capture_output=True,
                    timeout=10
                )
            else:
                # On Unix, send SIGTERM first, then SIGKILL if needed
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)

            log("Server stopped")
        except Exception as e:
            log(f"Error stopping server: {e}")
        finally:
            self.process = None

    def restart_server(self):
        """Restart the server."""
        log("Restarting server...")
        self.stop_server()
        time.sleep(1)  # Brief pause before restart
        self.start_server()
        log("Server restarted successfully")

    def check_restart_flag(self):
        """Check if restart flag exists and process it."""
        if RESTART_FLAG_FILE.exists():
            try:
                with open(RESTART_FLAG_FILE, 'r') as f:
                    flag_time = f.read().strip()
                log(f"Restart flag detected (created at: {flag_time})")

                # Remove the flag file first
                RESTART_FLAG_FILE.unlink()
                log("Restart flag removed")

                # Restart the server
                self.restart_server()
                return True

            except Exception as e:
                log(f"Error processing restart flag: {e}")
                # Try to remove flag even on error
                try:
                    RESTART_FLAG_FILE.unlink()
                except:
                    pass

        return False

    def check_server_health(self):
        """Check if server process is still running."""
        if self.process and self.process.poll() is not None:
            log(f"Server process died with exit code: {self.process.returncode}")
            self.process = None
            return False
        return True

    def run(self):
        """Main watcher loop."""
        log("=" * 50)
        log("Deploy Watcher starting...")
        log(f"Server script: {SERVER_SCRIPT}")
        log(f"Restart flag: {RESTART_FLAG_FILE}")
        log(f"Check interval: {CHECK_INTERVAL}s")
        log("=" * 50)

        # Remove any existing restart flag
        if RESTART_FLAG_FILE.exists():
            RESTART_FLAG_FILE.unlink()
            log("Removed stale restart flag")

        # Start the server
        self.start_server()

        # Main monitoring loop
        try:
            while self.running:
                # Check for restart flag
                self.check_restart_flag()

                # Check server health
                if not self.check_server_health():
                    log("Server not running, restarting...")
                    time.sleep(2)
                    self.start_server()

                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log("Keyboard interrupt received")
        finally:
            log("Deploy watcher shutting down...")
            self.stop_server()
            log("Deploy watcher stopped")


def signal_handler(signum, frame):
    """Handle termination signals."""
    log(f"Received signal {signum}")
    sys.exit(0)


def main():
    """Entry point."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Change to backend directory for imports
    os.chdir(SCRIPT_DIR)

    manager = ServerManager()
    manager.run()


if __name__ == "__main__":
    main()
