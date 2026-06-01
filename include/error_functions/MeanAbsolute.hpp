#pragma once
#include "types.hpp"
#include <cmath>

namespace mlask{

/** @brief Mean Absolute Error Function */
struct MeanAbsolute{
    float_t operator()(float_t result, float_t expected)const{
        return abs(expected - result);
    }
};

/** @brief Derived Mean Absolute Error Function */
struct DerivedMeanAbsolute{
    float_t operator()(float_t result, float_t expected)const{
        if(result > expected)   return -1;
        if(result < expected)   return 1;
        return 0;
    }
};
}
