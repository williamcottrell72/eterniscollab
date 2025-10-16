"""
Quick fix script for Jupyter notebook dependency issues

Run this in your notebook if you get import errors:
    %run fix_jupyter_dependencies.py

Or run from command line:
    python fix_jupyter_dependencies.py
"""

import sys
import subprocess


def install_package(package):
    """Install a package using the current Python interpreter."""
    print(f"Installing {package}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False


def check_package(package):
    """Check if a package is installed and importable."""
    try:
        mod = __import__(package)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {package:20s} {version}")
        return True
    except ImportError:
        print(f"✗ {package:20s} NOT INSTALLED")
        return False


def main():
    print("=" * 70)
    print("Jupyter Dependencies Fix Script")
    print("=" * 70)
    print(f"\nPython executable: {sys.executable}")
    print(f"Python version: {sys.version}\n")

    # Required packages for polymarket_data.py
    required_packages = [
        "pandas",
        "numpy",
        "requests",
        "pyarrow",  # Critical for parquet support
    ]

    print("Checking required packages:")
    print("-" * 70)

    missing = []
    for package in required_packages:
        if not check_package(package):
            missing.append(package)

    if missing:
        print(f"\n{len(missing)} package(s) missing: {', '.join(missing)}")
        print("\nAttempting to install missing packages...")
        print("-" * 70)

        for package in missing:
            install_package(package)

        print("\nRechecking packages:")
        print("-" * 70)
        for package in required_packages:
            check_package(package)

    else:
        print("\n✓ All required packages are installed!")

    # Test parquet support
    print("\nTesting parquet support:")
    print("-" * 70)
    try:
        import pandas as pd
        import tempfile
        import os

        df = pd.DataFrame({"test": [1, 2, 3], "value": [4, 5, 6]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_file = f.name

        # Write parquet
        df.to_parquet(temp_file, index=False)
        print(f"✓ Writing parquet file: {temp_file}")

        # Read parquet
        df_read = pd.read_parquet(temp_file)
        print(f"✓ Reading parquet file")

        # Verify data
        assert df.equals(df_read), "Data mismatch!"
        print(f"✓ Data integrity verified")

        # Cleanup
        os.unlink(temp_file)
        print(f"✓ Parquet support is working!")

    except Exception as e:
        print(f"✗ Parquet test failed: {e}")
        print(f"\nYou may need to install pyarrow manually:")
        print(f"  {sys.executable} -m pip install pyarrow")

    print("\n" + "=" * 70)
    print("Setup complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
