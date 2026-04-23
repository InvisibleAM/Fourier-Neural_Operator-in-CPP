#include "fno/FFT.hpp"
#include <cmath>

namespace fno {
    void fft1d(Complex1D& x, bool invert) {
        int n = x.size();
        if (n <= 1) return;

        Complex1D even(n / 2), odd(n / 2);
        for (int i = 0; 2 * i < n; i++) {
            even[i] = x[2 * i];
            odd[i] = x[2 * i + 1];
        }

        fft1d(even, invert);
        fft1d(odd, invert);

        double angle = 2 * PI / n * (invert ? 1 : -1);
        Complex w(1), wn(cos(angle), sin(angle));
        for (int i = 0; 2 * i < n; i++) {
            x[i] = even[i] + w * odd[i];
            x[i + n / 2] = even[i] - w * odd[i];
            if (invert) {
                x[i] /= 2;
                x[i + n / 2] /= 2;
            }
            w *= wn;
        }
    }
}
