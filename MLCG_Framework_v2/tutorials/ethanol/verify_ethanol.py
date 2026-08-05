#!/usr/bin/env python3
"""Compatibility entry point for the ethanol NVE scaling test.

Use run_energy_scaling.py directly.  This wrapper deliberately does not import
ESPResSo because the scaling driver launches the selected pypresso executable.
"""
from run_energy_scaling import main

if __name__ == "__main__":
    main()
