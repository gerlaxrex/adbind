#include "Variable.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

using namespace adbind;

// Linear function with weights
std::shared_ptr<Variable> function(float x, float y, 
                                   std::shared_ptr<Variable> w1, 
                                   std::shared_ptr<Variable> w2, 
                                   std::shared_ptr<Variable> b) {
    auto term1 = w1 * x;
    auto term2 = w2 * y;
    return relu(b + term1 + term2);
}

// The real process we're trying to approximate
double real_process(float x, float y) {
    return 1.0 + 2.5 * x - 0.32 * y + std::cos(x);
}

int main() {
    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    int N_SAMPLES = 200;
    
    std::vector<std::pair<float, float>> inputs;
    for (int i = 0; i < N_SAMPLES; i++) {
        inputs.push_back({dis(gen), dis(gen)});
    }
    
    auto w1 = std::make_shared<Variable>(0.2);
    auto w2 = std::make_shared<Variable>(0.3);
    auto b = std::make_shared<Variable>(0.0);
    
    const int epochs = 10;
    const double lr = 0.1;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "#### Epoch " << epoch << std::endl;
        
        for (int idx = 0; idx < inputs.size(); idx++) {
            auto [x, y] = inputs[idx];
            auto yHat = function(x, y, w1, w2, b);
            double yReal = real_process(x, y);
            auto yRealVar = std::make_shared<Variable>(yReal);
            
            auto diff = yHat - yRealVar;
            auto loss = diff * diff;
            loss->backward();

            if (idx % 20 == 0) {
                std::cout << "Loss variable: " << loss << std::endl;
            }
            
            // GD
            w1->setValue(w1->getValue() - lr * w1->getGrad());
            w2->setValue(w2->getValue() - lr * w2->getGrad());
            b->setValue(b->getValue() - lr * b->getGrad());
            
            // Zero grad
            loss->reset();
            w1->reset();
            w2->reset();
            b->reset();
        }
    }
    
    std::cout << "\nFinal weights and bias:" << std::endl;
    std::cout << "w1: " << w1->getValue() << " (target: 2.5)" << std::endl;
    std::cout << "w2: " << w2->getValue() << " (target: -0.32)" << std::endl;
    std::cout << "b: " << b->getValue() << " (target: 1.0)" << std::endl;
    
    std::cout << "\nPredictions vs. Real values:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::setw(15) << "Prediction" << std::setw(15) << "Real Value" << std::endl;
    
    for (int i = 0; i < 5; i++) {
        auto [x, y] = inputs[i];
        auto yHat = function(x, y, w1, w2, b);
        double yReal = real_process(x, y);
        
        std::cout << std::setw(15) << yHat->getValue() << std::setw(15) << yReal << std::endl;
    }
    
    return 0;
}