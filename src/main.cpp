#include <iostream>
#include <cmath>
#include <iomanip>
#include "fno/Types.hpp"
#include "fno/SpectralConv1D.hpp"

using namespace fno;

int main() {
    int spatial_resolution = 64; 
    int modes = 8;               
    double c = 1.0;              
    double t = 0.25;             

    SpectralConv1D fno_layer(modes, spatial_resolution);
    fno_layer.inject_exact_pde_operator(c, t);

    Tensor1D grid(spatial_resolution);
    Tensor1D initial_condition(spatial_resolution);
    Tensor1D exact_solution(spatial_resolution);

    for(int i = 0; i < spatial_resolution; ++i) {
        grid[i] = (double)i / spatial_resolution;
        initial_condition[i] = std::sin(2.0 * PI * grid[i]);
        exact_solution[i] = std::sin(2.0 * PI * (grid[i] - c * t)); 
    }

    Tensor1D fno_prediction = fno_layer.forward(initial_condition);

    double mse = 0.0;
    std::cout << "FNO Prediction vs Exact Solution\n";
    for(int i = 0; i < spatial_resolution; i += 8) {
        double diff = std::abs(exact_solution[i] - fno_prediction[i]);
        mse += diff * diff;
        std::cout << "x: " << std::fixed << std::setprecision(3) << grid[i] 
                  << " | Exact: " << exact_solution[i] 
                  << " | FNO: " << fno_prediction[i] << "\n";
    }

    std::cout << "MSE: " << std::scientific << (mse / spatial_resolution) << "\n";
    return 0;
}
