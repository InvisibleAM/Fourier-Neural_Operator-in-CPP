# Fourier Neural Operator (FNO) in C++


## What is FNO and Why Should You Care?

Many physics problems require solving differential equations—for example, how heat diffuses through a material, how waves propagate, or how fluids flow. Traditional approaches compute these solutions by discretizing the domain and solving large systems of equations, which is computationally expensive.

**Fourier Neural Operators** learn to approximate the solution operator directly by working in the frequency domain. Once trained on example solutions, they can generate predictions for new initial conditions in a fraction of the time required by classical numerical methods.


## How FNO Works

The fundamental idea is to transform the problem into a domain where operations are simpler:

1. **Input** → Convert data to frequency domain (decompose into frequency components, like analyzing the frequency spectrum of a signal)
2. **Process** → Apply transformations in frequency space (multiply by learnable weights)
3. **Output** → Transform back to spatial domain (reconstruct the physical solution)

The key advantage: **Frequency space representation makes solving physics problems computationally more efficient.**

## Project Structure

```
fno_cpp/
├── CMakeLists.txt              Build configuration and compilation rules
├── include/fno/
│   ├── Types.hpp               Type definitions and constants
│   ├── FFT.hpp                 Fast Fourier Transform interface
│   └── SpectralConv1D.hpp      Spectral convolution layer interface
├── src/
│   ├── FFT.cpp                 FFT algorithm implementation
│   ├── SpectralConv1D.cpp      FNO layer implementation
│   └── main.cpp                Demo application and test case
└── build/                      Directory for compiled binaries
```

## What Each File Does

| File | Purpose |
|------|---------|
| **Types.hpp** | Defines common data types used throughout the code |
| **FFT.cpp** | Converts data to/from frequency space (Fast Fourier Transform) |
| **SpectralConv1D.cpp** | The core FNO algorithm that solves physics problems |
| **main.cpp** | Example: solves a wave equation to show how it works |
## Quick Start (5 Minutes)

### Step 1: Set Up

Make sure you have:
- **A C++ compiler** (comes with most systems)
- **CMake** (build tool) — [Install here](https://cmake.org/download/)

### Step 2: Build

Open a terminal in the project directory and run:

```bash
mkdir build
cd build
cmake ..
make
```

Expected output indicating successful compilation:
```
[100%] Built target fno_main
```

### Step 3: Run

```bash
./fno_main
```

You'll see predictions compared to exact answers:
```
FNO Prediction vs Exact Solution
x: 0.000 | Exact: -1.000 | FNO: -1.000
x: 0.125 | Exact: -0.707 | FNO: -0.707
x: 0.250 | Exact: 0.000 | FNO: 0.000
...
MSE: 1.570e-32
```

## Understanding the Example: 1D Advection

This demonstration solves a fundamental PDE: the **1D advection (transport) equation**.

### Simulation Parameters

- **Grid size**: 64 spatial points (discretization of the domain)
- **Wave speed**: 1.0 unit per time step (constant velocity)
- **Time**: 0.25 time units (prediction horizon)
- **Frequency modes**: 8 (using only 8 frequency components instead of 64 to represent the solution)

### Workflow

1. **Initializes** a wave pattern with sinusoidal initial condition
2. **Applies** the FNO operator to predict how the wave evolves (advection by 0.25 units)
3. **Compares** the FNO prediction with the analytically computed solution
4. **Quantifies accuracy** using Mean Squared Error (demonstrates near-machine-precision agreement)

---
## Building and Running

### Prerequisites

- C++ compiler with C++17 support (GCC, Clang, MSVC)
- CMake 3.10+
- Standard library only (no external dependencies)

### Build Steps

```bash
cd /path/to/fno_cpp
mkdir build
cd build
cmake ..
make
```

**Expected Output:**
```
-- Configuring done
-- Generating done
[100%] Built target fno_main
```

### Running the Application

```bash
./fno_main
```

**Expected Output:**
```
FNO Prediction vs Exact Solution
x: 0.000 | Exact: -1.000 | FNO: -1.000
x: 0.125 | Exact: -0.707 | FNO: -0.707
x: 0.250 | Exact: 0.000 | FNO: 0.000
x: 0.375 | Exact: 0.707 | FNO: 0.707
x: 0.500 | Exact: 1.000 | FNO: 1.000
x: 0.625 | Exact: 0.707 | FNO: 0.707
x: 0.750 | Exact: 0.000 | FNO: 0.000
x: 0.875 | Exact: -0.707 | FNO: -0.707
MSE: 1.570e-32
```

### Rebuild

```bash
make clean    # Remove compiled files
make          # Rebuild
```


### Building with External Libraries

To add libraries (e.g., Eigen, PyTorch C++), update `CMakeLists.txt`:

```cmake
find_package(Eigen3 REQUIRED)
target_link_libraries(fno_core Eigen3::Eigen)
```

## References

- [Fourier Neural Operator for Parametric PDEs](https://arxiv.org/abs/2010.08895) - Li et al., ICLR 2021
- Cooley-Tukey FFT Algorithm: O(n log n) complexity
- PDE Solving with Neural Operators: Emergent deep learning technique


