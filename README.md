# Fourier Neural Operator (FNO) in C++


Many physics problems require solving differential equations — for example, how heat diffuses through a material, how waves propagate, or how fluids flow. Traditional approaches compute these solutions by discretizing the domain and solving large systems of equations, which is computationally expensive.

Fourier Neural Operators learn to approximate the solution operator directly by working in the frequency domain. Once trained on example solutions, they can generate predictions for new initial conditions in a fraction of the time required by classical numerical methods.


### How FNO Works

The idea is to transform the problem into a domain where operations are simpler:

1. **Input** - Convert data to frequency domain (decompose into frequency components)
2. **Process** - Apply transformations in frequency space (multiply by learnable weights)
3. **Output** - Transform back to spatial domain 

Frequency space representation makes solving computationally more efficient.



| **Types.hpp** | Defines common data types used throughout the code |
| **FFT.cpp** | Converts data to/from frequency space (Fast Fourier Transform) |
| **SpectralConv1D.cpp** | The core FNO algorithm that solves physics problems |
| **main.cpp** | Example: solves a wave equation to show how it works |


## Usage

Make sure you have:
- **A C++ compiler** 
- **CMake** (build tool) — [Install here](https://cmake.org/download/)

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

Then run:

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

## Example: 1D Advection

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
