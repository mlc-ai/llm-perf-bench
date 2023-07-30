echo "export TVM_HOME=$MLC_HOME/3rdparty/tvm" >>~/.bashrc
echo "export TVM_LIBRARY_PATH=\$TVM_HOME/build/" >>~/.bashrc
echo "export PYTHONPATH=\$TVM_HOME/python" >>~/.bashrc

source ~/.bashrc && micromamba activate python311

cd $TVM_HOME && mkdir build && cd build && cp ../cmake/config.cmake .
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >>config.cmake
echo "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" >>config.cmake
echo "set(USE_GTEST OFF)" >>config.cmake
echo "set(USE_CUDA ON)" >>config.cmake
echo "set(USE_LLVM ON)" >>config.cmake
echo "set(USE_VULKAN OFF)" >>config.cmake
echo "set(USE_CUTLASS ON)" >>config.cmake
cmake .. && make -j$(nproc)
