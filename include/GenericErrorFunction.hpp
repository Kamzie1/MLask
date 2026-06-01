#pragma once
#include "ErrorFunction.hpp"
namespace mlask{
/** @brief Class representing an error function
 * Children should define a way to move forawrd and backward, but for simplicity they define a function that operates on a single x_1 and x_2 from result vector and expected vector, instead of on the entire vector
*/
template <ErrorFunction err>
class GenericErrorFunction{
private:
public:
    /** @brief calculates derived error
    * @param result neural network result
    * @param expected expected result
    * @return derived error
    */
    static vectorOut error(vectorOut result, vectorOut expected){
        return result.binaryExpr(expected, err());
    }
    /** @brief get the error as a scalar value
    * @param result neural network result in a vector form
    * @param expected expected result in a vector form
    * @return error as a scalar value
    */
    static float_t error_scalar(vectorOut result, vectorOut expected){
        return error(result, expected).sum() / result.size();
    }
};
}
