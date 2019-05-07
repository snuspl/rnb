# Convenient shell script for building the Python-C++ bridge .so file.
#
# The build requires pybind11 headers (https://github.com/pybind/pybind11),
# and assumes that Python3 is available within a Conda environment.

# assume pybind11 headers are located in the following path 
PYBIND11_INCLUDE=../pybind11/include

# if CUDA_HOME is not set we search for /usr/local/cuda
if [[ -z "${CUDA_HOME}" ]]; then
  CUDA_HOME=/usr/local/cuda
fi


NAME=cupti
SRC=utils/${NAME}.cpp
OBJ=utils/${NAME}.so

$CUDA_HOME/bin/nvcc -O3 -shared -std=c++11 -Xcompiler -fPIC            \
    -I$PYBIND11_INCLUDE `python3-config --includes`                    \
    -I$CUDA_HOME/extras/CUPTI/include  -I$CONDA_PREFIX/include $SRC    \
    -L$CONDA_PREFIX/lib -lcuda -L$CUDA_HOME/extras/CUPTI/lib64 -lcupti \
    -o $OBJ
