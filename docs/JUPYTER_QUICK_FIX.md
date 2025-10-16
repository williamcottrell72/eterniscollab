# Quick Fix for Jupyter Notebook

## Problem
```
ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.
Missing optional dependency 'pyarrow'.
```

## Solution

### Option 1: Run this in a notebook cell (FASTEST)

```python
import sys
!{sys.executable} -m pip install pyarrow
```

Then restart your kernel: **Kernel → Restart Kernel**

### Option 2: Run the fix script

In a notebook cell:
```python
%run fix_jupyter_dependencies.py
```

### Option 3: Install from terminal

```bash
# Find which Python your notebook uses
jupyter kernelspec list

# Install pyarrow
pip install pyarrow

# Or if using conda
conda install pyarrow
```

### Option 4: Verify and install in one cell

```python
# Run this entire cell
import sys

# Check current Python
print(f"Python: {sys.executable}")

# Try importing pyarrow
try:
    import pyarrow
    print(f"✓ pyarrow {pyarrow.__version__} is installed")
except ImportError:
    print("Installing pyarrow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
    print("✓ pyarrow installed. Please restart the kernel.")
```

## After Installing

**IMPORTANT:** Restart your Jupyter kernel after installing:
- **Kernel → Restart Kernel** (from menu)
- Or click the restart button in toolbar

Then try your code again:
```python
from polymarket_data import download_polymarket_prices_by_slug
from datetime import datetime

df = download_polymarket_prices_by_slug(
    market_slug="fed-rate-hike-in-2025",
    outcome_index=0,
    start_date=datetime(2024, 12, 1),
    end_date=datetime(2024, 12, 2),
    fidelity=1
)
```

## Why This Happens

Your Jupyter notebook kernel uses a different Python environment than your terminal. Installing packages in the terminal doesn't affect the notebook kernel unless you restart it.

## Long-term Solution

Create a dedicated kernel with all dependencies:

```bash
# Install all dependencies
pip install -r requirements.txt

# Create a new kernel
python -m ipykernel install --user --name=eterniscollab --display-name="Python (eterniscollab)"

# Restart Jupyter and select the new kernel
```

Then in Jupyter: **Kernel → Change Kernel → Python (eterniscollab)**
