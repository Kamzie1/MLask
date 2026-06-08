#pragma once
#include "types.hpp"
#include <cmath>

namespace mlask{

/** @brief Categorical Cross-Entropy Error Function */
struct CategoricalCrossEntropy{
    float_t operator()(float_t result, float_t expected)const{
        return -1*result*log(expected);
    }
};

/** @brief Derived Categorical Cross-Entropy Error Function */
struct DerivedCategoricalCrossEntropy{
    float_t operator()(float_t result, float_t expected)const{
        return -1 * result / expected;
    }
};
}
