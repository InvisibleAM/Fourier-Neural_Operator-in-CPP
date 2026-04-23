# Fourier Neural Operator (FNO) in C++

## Mathematical Foundation

### 1. Fourier Neural Operators

FNOs learn mappings between function spaces by operating in the Fourier domain:

$$\mathcal{F}: u(x) \rightarrow v(x)$$

where $u(x)$ is the input function and $v(x)$ is the output function.

**Key Equation - Spectral Convolution:**

$$v_j^{\ell+1}(x) = \sigma \left( \sum_{k=1}^{d} \int_{\Omega} e^{2\pi i k x \cdot \xi} \mathcal{R}^{\ell}_k(\xi) \left( \int_{\Omega} e^{-2\pi i k x \cdot \xi} v_j^{\ell}(x) dx \right) d\xi \right)$$

Simplified for 1D:

$$y(x) = \mathcal{FFT}^{-1}\left[ \mathcal{R}(\omega) \odot \mathcal{FFT}[u(x)] \right]$$

where:
- $\mathcal{FFT}$ = Fast Fourier Transform
- $\mathcal{R}(\omega)$ = Learnable weights in Fourier space (complex)
- $\odot$ = Element-wise multiplication
- $\mathcal{FFT}^{-1}$ = Inverse FFT (returns to spatial domain)

### 2. Wave Equation Example

The code solves the 1D advection equation (constant-speed wave equation):

$$\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0$$

**Exact Solution:**
$$u(x, t) = u_0(x - ct)$$

The FNO learns this operator by encoding the phase shift $e^{-2\pi i k c t}$ directly into the spectral weights $\mathcal{R}_k$.

### 3. Fast Fourier Transform (Cooley-Tukey)

The recursive FFT algorithm reduces complexity from $O(n^2)$ to $O(n \log n)$:

**Algorithm:**
1. Divide sequence into even and odd indices
2. Recursively transform both halves
3. Combine results using twiddle factors: $W_n^k = e^{2\pi i k/n}$

**For Inversion:** Apply FFT with inverted angle, then divide by 2 at each stage.



### File Descriptions

#### `include/fno/Types.hpp`
Centralized type definitions for consistency across the codebase:
- `Complex` = `std::complex<double>` (complex numbers for FFT)
- `Tensor1D` = `std::vector<double>` (real-valued 1D tensors)
- `Complex1D` = `std::vector<Complex>` (spectral tensors)
- `PI` = $\pi$ constant for mathematical operations

**Why this matters:** Changing precision (e.g., `double` to `float`) requires only one edit.

#### `include/fno/FFT.hpp` & `src/FFT.cpp`
Implements the 1D Cooley-Tukey Fast Fourier Transform:

```cpp
void fft1d(Complex1D& x, bool invert);
```

**Parameters:**
- `x`: Vector of complex numbers (modified in-place)
- `invert`: `false` for forward FFT, `true` for inverse FFT

**Algorithm Steps:**
1. Base case: If size ≤ 1, return
2. Split input into even and odd indices
3. Recursively transform both halves
4. Combine using twiddle factors: $e^{-2\pi i k / n}$
5. For inversion, divide by 2 at each stage

**Complexity:** $O(n \log n)$ where $n$ = sequence length (must be power of 2)

#### `include/fno/SpectralConv1D.hpp` & `src/SpectralConv1D.cpp`
Spectral convolution layer implementing the FNO operator:

**Class Members:**
```cpp
int modes;                    // Number of Fourier modes to retain
int spatial_res;              // Spatial resolution (grid points)
Complex1D weights_R;          // Learnable spectral weights
```

**Key Methods:**

1. **`SpectralConv1D(int m, int res)`** - Constructor
   - Initializes with `m` Fourier modes and `res` spatial points

2. **`inject_exact_pde_operator(double wave_speed, double target_time)`** - Ground truth injection
   - Encodes the exact solution of the advection equation
   - Sets: $\mathcal{R}_k = e^{-2\pi i k c t}$
   - Useful for validation and testing

3. **`forward(const Tensor1D& input)`** - Forward pass
   - **Step 1:** Convert input to complex (real part only)
   - **Step 2:** Apply forward FFT → spectral domain
   - **Step 3:** Multiply by spectral weights (convolution)
   - **Step 4:** Mirror conjugates (ensure real output)
   - **Step 5:** Apply inverse FFT → spatial domain
   - **Step 6:** Extract real part
   - **Returns:** Tensor1D with transformed output

#### `src/main.cpp`
Demonstration that applies FNO to a simple wave equation:

**Configuration:**
```cpp
int spatial_resolution = 64;  // 64 grid points
int modes = 8;                // Use 8 Fourier modes
double c = 1.0;              // Wave speed
double t = 0.25;             // Time step
```

**Workflow:**
1. Create FNO layer with given parameters
2. Inject exact PDE operator (ground truth)
3. Generate initial condition: $u_0(x) = \sin(2\pi x)$
4. Compute exact solution: $u(x,t) = \sin(2\pi(x - ct))$
5. Run FNO forward pass
6. Compare predictions vs exact solution (compute MSE)

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


