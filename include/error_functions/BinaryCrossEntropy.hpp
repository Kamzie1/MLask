#pragma once
#include "types.hpp"
#include <cmath>

namespace mlask{

/** @brief Binary Cross-Entropy Error Function */
struct BinaryCrossEntropy{
    float_t operator()(float_t result, float_t expected)const{
        return -1*(result*log(expected) + (1-result)*log(1-expected));
    }
};

/** @brief Derived Binary Cross-Entropy Error Function */
struct DerivedBinaryCrossEntropy{
    float_t operator()(float_t result, float_t expected)const{
        return (expected - result) / expected*(1 - expected);
    }
};
}
