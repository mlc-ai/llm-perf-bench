echo "export MLC_HOME=$MLC_HOME" >>~/.bashrc
source ~/.bashrc && micromamba activate python311

# Workaround the issue:
# > libcuda.so.1, needed by tvm/libtvm_runtime.so, not found (try using -rpath or -rpath-link)

CUDA_STUB=/usr/local/cuda/lib64/stubs/
ln -s $CUDA_STUB/libcuda.so $CUDA_STUB/libcuda.so.1
export LD_LIBRARY_PATH=$CUDA_STUB:$LD_LIBRARY_PATH

cd $MLC_HOME && mkdir build && cd build && touch config.cmake
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >>config.cmake
echo "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" >>config.cmake
echo "set(USE_CUDA ON)" >>config.cmake
echo "set(USE_VULKAN OFF)" >>config.cmake
echo "set(USE_METAL OFF)" >>config.cmake
echo "set(USE_OPENCL OFF)" >>config.cmake
cmake .. && make -j$(nproc)

rm $CUDA_STUB/libcuda.so.1
