#!/bin/bash

echo "[INFO] Checking dependencies..."
python -c "import Cython" 2>/dev/null || (
    echo "[INFO] Installing Cython..."
    pip install Cython
)
python -c "import numpy" 2>/dev/null || (
    echo "[INFO] Installing numpy..."
    pip install numpy
)

echo "[INFO] Cleaning previous builds..."
rm -rf build/ *.c *.cpp *.so __pycache__/

# For now, to keep the PATHs simple
echo "[INFO] Building Cython extensions..."
python setup.py build_ext --build-lib=./

# echo "[INFO] Building Cython extensions..."
# python setup.py build_ext --build-lib=build/

# echo "[INFO] Organizing build artifacts..."
# mv *.cpp build/ 2>/dev/null || true
# mv *.so build/ 2>/dev/null || true
# rm -rf build/temp.* 2>/dev/null || true

# if ls build/qqs*.so 1> /dev/null 2>&1; then
#     echo "[SUCCESS] Build completed successfully"
# else
#     echo "[ERROR] Build failed - extensions not found"
#     exit 1
# fi
