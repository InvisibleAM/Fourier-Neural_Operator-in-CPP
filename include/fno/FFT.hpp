#pragma once
#include "fno/Types.hpp"

namespace fno {
    // Computes the 1D Cooley-Tukey Fast Fourier Transform
    // Input size must be a power of 2.
    void fft1d(Complex1D& x, bool invert);
}
