#pragma once
#include "fno/Types.hpp"

namespace fno {
    class SpectralConv1D {
    private:
        int modes;
        int spatial_res;
        Complex1D weights_R;

    public:
        SpectralConv1D(int m, int res);
        
        // Helper for injecting mathematical ground truth
        void inject_exact_pde_operator(double wave_speed, double target_time);
        
        // Forward pass execution
        Tensor1D forward(const Tensor1D& input);
    };
}
