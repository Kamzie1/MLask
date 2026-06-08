#pragma once
#include "types.hpp"
#include <cstddef>

namespace mlask{

struct ConfusionMatrix{
    std::size_t TP;
    std::size_t TN;
    std::size_t FP;
    std::size_t FN;

    /** @brief evaluates the result
    * @param prediction the neural network solution
    * @param expected ewxpected solution
    */
    void evaluate(int prediction, int expected);

    
    /** @brief accuracy of the model */
    float_t accuracy()const;
    /** @brief precision of the model */
    float_t precision()const;
    /** @brief recall of the model */
    float_t recall()const;
    /** @brief F1 score of the model */
    float_t F1_score()const;

    friend std::ostream & operator<<(std::ostream &os, const ConfusionMatrix& conf);
    
};

}
