# Installation Guide

## Quick Start

```bash
pip install -r requirements.txt
```

## Dependencies

All required dependencies are listed in `requirements.txt`:

```
openai>=1.0.0
anthropic>=0.18.0
pydantic-ai-slim>=0.0.14
pydantic>=2.0.0
httpx>=0.24.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
numpy>=1.24.0
plotly>=5.0.0
pandas>=2.0.0
jupyter>=1.0.0
ipykernel>=6.0.0
requests>=2.31.0
pyarrow>=12.0.0  # Required for parquet file support
```

## Common Issues

### Issue: "Missing optional dependency 'pyarrow'"

**Problem:** You're running code in a Jupyter notebook with a kernel that doesn't have pyarrow installed.

**Solution 1 - Install in your Jupyter kernel:**

Run this in a notebook cell:
```python
import sys
!{sys.executable} -m pip install pyarrow
```

**Solution 2 - Install via conda (if using conda):**
```bash
conda install pyarrow
```

**Solution 3 - Install in base environment:**
```bash
pip install pyarrow
```

**Solution 4 - Verify installation:**
```python
import pyarrow
print(f"pyarrow version: {pyarrow.__version__}")
```

### Issue: Jupyter Kernel Doesn't Have Packages

**Problem:** Your Jupyter kernel uses a different Python environment than where you installed packages.

**Solution:**

1. Check which Python your notebook is using:
```python
import sys
print(sys.executable)
```

2. Install packages for that specific Python:
```bash
/path/to/your/python -m pip install -r requirements.txt
```

3. Or create a new kernel with all packages:
```bash
python -m ipykernel install --user --name=eterniscollab --display-name="Python (eterniscollab)"
```

Then select this kernel in Jupyter.

### Issue: Import errors in notebooks

**Quick Fix - Install directly in notebook:**
```python
import sys
!{sys.executable} -m pip install pyarrow pandas requests
```

## Environment Setup

### Option 1: pip (Recommended for most users)

```bash
# Clone the repo
cd eterniscollab

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pyarrow; print('Success!')"
```

### Option 2: conda

```bash
# Create conda environment (optional)
conda create -n eterniscollab python=3.12
conda activate eterniscollab

# Install dependencies
pip install -r requirements.txt

# Or use conda for some packages
conda install pyarrow pandas numpy plotly jupyter
pip install -r requirements.txt  # For remaining packages
```

### Option 3: For Jupyter Notebooks

```bash
# Install dependencies
pip install -r requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name=eterniscollab

# Start Jupyter
jupyter notebook

# In Jupyter: Kernel -> Change Kernel -> eterniscollab
```

## Verify Installation

Run this test script:

```python
# test_installation.py
import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}\n")

packages = [
    "pandas",
    "numpy",
    "requests",
    "pyarrow",
    "plotly",
    "pydantic",
    "openai",
    "anthropic"
]

print("Checking packages:")
for package in packages:
    try:
        mod = __import__(package)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {package:20s} {version}")
    except ImportError:
        print(f"✗ {package:20s} NOT INSTALLED")

print("\nTesting parquet support:")
try:
    import pandas as pd
    import tempfile
    import os

    df = pd.DataFrame({"a": [1, 2, 3]})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        temp_file = f.name

    df.to_parquet(temp_file)
    df2 = pd.read_parquet(temp_file)
    os.unlink(temp_file)

    print("✓ Parquet support working!")
except Exception as e:
    print(f"✗ Parquet support failed: {e}")
```

Run it:
```bash
python test_installation.py
```

## Troubleshooting

### Still getting import errors?

1. **Check Python version:**
   ```bash
   python --version  # Should be 3.12+
   ```

2. **Check where packages are installed:**
   ```bash
   pip show pyarrow
   ```

3. **Try installing in user directory:**
   ```bash
   pip install --user -r requirements.txt
   ```

4. **Clear pip cache:**
   ```bash
   pip cache purge
   pip install --force-reinstall pyarrow
   ```

5. **Use the EXACT Python that Jupyter uses:**
   ```python
   # In Jupyter notebook
   import sys
   print(sys.executable)

   # Then in terminal
   /path/shown/above -m pip install pyarrow
   ```

## Need Help?

If you continue to have issues:

1. Check which Python you're running:
   ```bash
   which python
   which python3
   ```

2. Check if you have multiple Python installations:
   ```bash
   ls -la $(which python)
   ls -la $(which python3)
   ```

3. Try creating a fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
