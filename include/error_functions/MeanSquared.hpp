#pragma once
#include "types.hpp"

namespace mlask{

/** @brief Mean Squared Error Function */
struct MeanSquared{
    float_t operator()(float_t result, float_t expected)const{
        return 0.5 * (expected - result) * (expected - result);
    }
};

/** @brief Derived Mean Squared Error Function */
struct DerivedMeanSquared{
    float_t operator()(float_t result, float_t expected)const{
        return result - expected;
    }
};
}
