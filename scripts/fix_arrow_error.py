"""
Automated fix for ArrowKeyError in Jupyter notebooks

This script fixes the pandas/pyarrow compatibility issue by upgrading
both packages to compatible versions.

Usage:
    In a Jupyter notebook: %run fix_arrow_error.py
    From command line: python fix_arrow_error.py
"""

import sys
import subprocess


def run_command(cmd):
    """Run a pip command and return success status."""
    try:
        subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    print("=" * 70)
    print("ArrowKeyError Fix Script")
    print("=" * 70)
    print(f"\nPython: {sys.executable}")
    print(f"Version: {sys.version}\n")

    # Check current versions
    print("Checking current versions:")
    print("-" * 70)

    try:
        import pandas as pd

        print(f"pandas: {pd.__version__}")
    except ImportError:
        print("pandas: NOT INSTALLED")

    try:
        import pyarrow as pa

        print(f"pyarrow: {pa.__version__}")
    except ImportError:
        print("pyarrow: NOT INSTALLED")

    print("\nUpgrading packages...")
    print("-" * 70)

    # Upgrade both pandas and pyarrow together
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pandas>=2.2.0",
        "pyarrow>=14.0.0",
    ]

    print("Running: " + " ".join(cmd))

    if run_command(cmd):
        print("✓ Packages upgraded successfully")
    else:
        print("✗ Upgrade failed")
        print("\nTrying alternative method...")

        # Try uninstalling first
        for pkg in ["pandas", "pyarrow"]:
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        # Reinstall
        if run_command(cmd):
            print("✓ Packages reinstalled successfully")
        else:
            print("✗ Reinstall failed")
            print("\nPlease try manually:")
            print(f"  {sys.executable} -m pip install --upgrade pandas pyarrow")
            return

    # Verify fix
    print("\nVerifying fix:")
    print("-" * 70)

    # Force reimport
    if "pandas" in sys.modules:
        del sys.modules["pandas"]
    if "pyarrow" in sys.modules:
        del sys.modules["pyarrow"]

    try:
        import pandas as pd
        import pyarrow as pa

        print(f"pandas: {pd.__version__}")
        print(f"pyarrow: {pa.__version__}")

        # Test parquet
        import tempfile
        import os

        df = pd.DataFrame({"test": [1, 2, 3]})
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_file = f.name

        df.to_parquet(temp_file, index=False)
        df2 = pd.read_parquet(temp_file)
        os.unlink(temp_file)

        print("✓ Parquet support working!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("\nYou may need to restart your Jupyter kernel:")
        print("  Kernel → Restart Kernel")

    print("\n" + "=" * 70)
    print("IMPORTANT: Restart your Jupyter kernel to apply changes!")
    print("  Kernel → Restart Kernel")
    print("=" * 70)


if __name__ == "__main__":
    main()
