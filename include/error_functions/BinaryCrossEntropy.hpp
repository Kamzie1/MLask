#pragma once
#include "types.hpp"
#include <cmath>

namespace mlask{

/** @brief Binary Cross-Entropy Error Function */
struct BinaryCrossEntropy{
    float_t operator()(float_t result, float_t expected)const{
        return -1*(result*std::log(expected) + (1-result)*std::log(1-expected));
    }
};

/** @brief Derived Binary Cross-Entropy Error Function */
struct DerivedBinaryCrossEntropy{
    float_t operator()(float_t result, float_t expected)const{
        float_t epsilon = 1e-5;
        float_t res = std::max<float_t>(epsilon, std::min(1.0f - epsilon, result));
        return (res - expected) /( res*(1.0f - res));
    }
};
}
