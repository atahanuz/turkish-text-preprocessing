#!/usr/bin/env python3
"""
Startup script to run both the Python server and React app simultaneously.
"""

import subprocess
import sys
import os
import signal
import time

def start_processes():
    """Start both server.py and the React app."""
    processes = []

    try:
        # Start the Python server
        print("Starting Python server...")
        print("="*50)
        server_process = subprocess.Popen(
            [sys.executable, "server.py"]
        )
        processes.append(("Server", server_process))
        print(f"\n✓ Server started (PID: {server_process.pid})")

        # Give the server a moment to start
        time.sleep(1)

        # Start the React app
        print("\nStarting React app...")
        print("="*50)
        gui_path = os.path.join(os.getcwd(), "GUI")

        if not os.path.exists(gui_path):
            print(f"Error: GUI folder not found at {gui_path}")
            server_process.terminate()
            sys.exit(1)

        react_process = subprocess.Popen(
            ["npm", "start"],
            cwd=gui_path
        )
        processes.append(("React App", react_process))
        print(f"\n✓ React app started (PID: {react_process.pid})")

        print("\n" + "="*50)
        print("Both processes are running!")
        print("Press Ctrl+C to stop both processes")
        print("="*50 + "\n")

        # Monitor both processes
        while True:
            # Check if any process has terminated
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n{name} has stopped unexpectedly (exit code: {proc.returncode})")
                    # Terminate all other processes
                    for other_name, other_proc in processes:
                        if other_proc.poll() is None:
                            other_proc.terminate()
                    sys.exit(1)

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nShutting down processes...")
        for name, proc in processes:
            if proc.poll() is None:  # Process is still running
                print(f"Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    print(f"✓ {name} stopped")
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}...")
                    proc.kill()
                    proc.wait()
        print("\nAll processes stopped.")
        sys.exit(0)

    except Exception as e:
        print(f"\nError: {e}")
        # Clean up any running processes
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        sys.exit(1)

if __name__ == "__main__":
    start_processes()
