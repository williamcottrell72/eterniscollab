# Fix: ArrowKeyError in Jupyter Notebook

## The Error
```
ArrowKeyError: No type extension with name arrow.py_extension_type found
```

## Root Cause
This is a **version compatibility issue** between pandas and pyarrow in your Jupyter kernel environment (`ml-env`). The versions are out of sync.

## Solution 1: Upgrade Both Packages (RECOMMENDED)

Run this in a notebook cell:

```python
import sys
!{sys.executable} -m pip install --upgrade pandas pyarrow
```

Then **restart your kernel**: Kernel → Restart Kernel

## Solution 2: Install Compatible Versions

Run this in a notebook cell:

```python
import sys
# Install specific compatible versions
!{sys.executable} -m pip install pandas==2.2.2 pyarrow==14.0.2
```

Then **restart your kernel**: Kernel → Restart Kernel

## Solution 3: Clear Cache and Reinstall

Sometimes the issue is cached imports. Run this:

```python
import sys
import subprocess

# Uninstall both
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pandas", "pyarrow"])

# Reinstall fresh
subprocess.run([sys.executable, "-m", "pip", "install", "pandas>=2.2.0", "pyarrow>=14.0.0"])

print("Done! Please restart your kernel.")
```

## Solution 4: Use Conda (if you're in a conda env)

From terminal:

```bash
# Activate your environment
conda activate ml-env

# Install via conda (better compatibility)
conda install pandas pyarrow -c conda-forge

# Or update existing
conda update pandas pyarrow -c conda-forge
```

Then restart Jupyter kernel.

## Solution 5: Create Fresh Environment

If nothing works, create a clean environment:

```bash
# Create new conda environment
conda create -n polymarket python=3.12 pandas pyarrow jupyter ipykernel -c conda-forge

# Activate it
conda activate polymarket

# Install other dependencies
pip install requests

# Create Jupyter kernel
python -m ipykernel install --user --name=polymarket --display-name="Python (polymarket)"

# Start Jupyter and select the new kernel
jupyter notebook
```

## Verify Fix

After upgrading and restarting kernel, run this:

```python
import sys
print(f"Python: {sys.executable}\n")

import pandas as pd
import pyarrow as pa

print(f"pandas: {pd.__version__}")
print(f"pyarrow: {pa.__version__}\n")

# Test parquet
import tempfile
import os

df = pd.DataFrame({"test": [1, 2, 3]})
with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
    temp_file = f.name

try:
    df.to_parquet(temp_file, index=False)
    df2 = pd.read_parquet(temp_file)
    os.unlink(temp_file)
    print("✓ Parquet support working!")
except Exception as e:
    print(f"✗ Still broken: {e}")
```

## Known Compatible Versions

These versions work well together:

| pandas | pyarrow | Status |
|--------|---------|--------|
| 2.2.2  | 14.0.2  | ✓ Compatible |
| 2.2.x  | 14.x    | ✓ Compatible |
| 2.1.x  | 13.x    | ✓ Compatible |
| 2.0.x  | 12.x    | ✓ Compatible |

Avoid mixing old pandas with new pyarrow or vice versa.

## Why This Happens

The error occurs when:
1. pandas and pyarrow are from different release cycles
2. Cached/stale imports in Jupyter kernel
3. Multiple Python environments with different versions

The extension type registration mechanism changed between versions, causing this conflict.

## Still Having Issues?

### Check your actual Python environment:

```python
import sys
print(sys.executable)
```

### Then install packages for THAT specific Python:

```bash
# Use the exact path shown above
/path/to/your/python -m pip install --upgrade pandas pyarrow
```

### Nuclear option - reinstall everything:

```python
import sys
import subprocess

packages = ["pandas", "pyarrow", "numpy", "requests"]

for pkg in packages:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", pkg])
    subprocess.run([sys.executable, "-m", "pip", "install", pkg])

print("Reinstalled all packages. Restart kernel.")
```

## After Fix Works

Your original code will work:

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

print(f"✓ Downloaded {len(df)} data points")
```
