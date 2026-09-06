#!/usr/bin/env python
"""Verify all project dependencies are installed."""

import importlib
import sys

# List of critical packages to verify
packages = [
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('sklearn', 'scikit-learn'),
    ('scipy', 'scipy'),
    ('pvlib', 'pvlib'),
    ('yaml', 'pyyaml'),
    ('joblib', 'joblib'),
    ('requests', 'requests'),
    ('pathlib', None),  # Built-in
    ('collections', None),  # Built-in
]

print("=" * 70)
print("DEPENDENCY VERIFICATION REPORT")
print("=" * 70)
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print()

missing = []
available = []

for import_name, package_name in packages:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'Built-in')
        available.append(f"✓ {import_name:20} ({version})")
    except ImportError as e:
        missing.append(f"✗ {import_name:20} - {str(e)}")

print("AVAILABLE PACKAGES:")
for item in sorted(available):
    print(f"  {item}")

if missing:
    print("\nMISSING PACKAGES:")
    for item in sorted(missing):
        print(f"  {item}")
else:
    print("\n✓ All required packages are available!")

print()
print("=" * 70)
