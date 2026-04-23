cmake_minimum_required(VERSION 3.10)
project(FNO_CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directory
include_directories(include)

# Create a library for the core FNO components
add_library(fno_core 
    src/FFT.cpp 
    src/SpectralConv1D.cpp
)

# Build the main executable
add_executable(fno_main src/main.cpp)
target_link_libraries(fno_main fno_core)
