#pragma once
#include <vector>
#include <complex>

namespace fno {
    const double PI = 3.14159265358979323846;
    using Complex = std::complex<double>;
    using Tensor1D = std::vector<double>;
    using Complex1D = std::vector<Complex>;
}
