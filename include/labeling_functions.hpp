#pragma once
#include "types.hpp"

/**
 * @brief Labels values in range [0.5, 1] to 1 and values in range [0, 0.5) to 0.
 * @param x The floating-point result from the neural network.
 * @return The rounded integer result.
 */
inline int post_process_round(float_t x) {
    return (int)(x + 0.5f);
}

/**
 * @brief Labels positive values as 1 and negative values as 0.
 * @param x The floating-point result from the neural network.
 * @return The sign integer result.
 */
inline int post_process_sign(float_t x) {
    return (x>=0) ? 1 : 0;
}
