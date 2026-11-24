#!/usr/bin/env python3
"""Root wrapper forwarding to tests/test_task4_corrections.py"""
import os
import sys
import subprocess

if __name__ == "__main__":
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "test_task4_corrections.py")
    sys.exit(subprocess.call([sys.executable, test_path] + sys.argv[1:]))
