#include "fno/SpectralConv1D.hpp"
#include "fno/FFT.hpp"
#include <cmath>

namespace fno {
    SpectralConv1D::SpectralConv1D(int m, int res) : modes(m), spatial_res(res) {
        weights_R.resize(modes, Complex(0,0));
    }

    void SpectralConv1D::inject_exact_pde_operator(double wave_speed, double target_time) {
        for(int k = 0; k < modes; ++k) {
            double theta = -2.0 * PI * k * wave_speed * target_time;
            weights_R[k] = Complex(cos(theta), sin(theta));
        }
    }

    Tensor1D SpectralConv1D::forward(const Tensor1D& input) {
        Complex1D spectral_repr(spatial_res);
        for(int x = 0; x < spatial_res; ++x) {
            spectral_repr[x] = Complex(input[x], 0.0);
        }
        
        fft1d(spectral_repr, false);

        Complex1D spectral_out(spatial_res, Complex(0,0));
        for(int k = 0; k < modes; ++k) {
            spectral_out[k] = spectral_repr[k] * weights_R[k];
        }
        
        // Mirror for real signals
        for(int k = 1; k < modes; ++k) {
             spectral_out[spatial_res - k] = std::conj(spectral_out[k]);
        }

        fft1d(spectral_out, true);

        Tensor1D output(spatial_res);
        for(int x = 0; x < spatial_res; ++x) {
            output[x] = spectral_out[x].real();
        }

        return output;
    }
}
