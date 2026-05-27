#pragma once
#include "Layer.hpp"
#include "Eigen/Dense"

namespace mlask{

/**
 * @brief ActivationFunction class
 * @brief Class representing an activation function via two functions passed in constructor
 * @brief This class is not convertible to ONNX and should preferably be used only for prototyping.
 * @brief To create your own activation function, you can inherit from Layer abstract class and define the forward and backward methods.
 */
class ActivationFunction : public Layer{
    actfunc func_;
    actfunc derv_;

    vectorIn_ input_;
public:
    /**
     * @param func an activation function
     * @param derv is a derived form of func
     * @param in size of the input, in=0 means the input is not restricted to certain size
     */
    ActivationFunction(actfunc func, actfunc derv, std::size_t in = 0):func_(func), derv_(derv) {in_ = in; out_ = in;}
    /** @brief Performs forward propagation
     * @param input the input vector
     * @return the output vector
     */
    vectorOut_ forward(vectorIn_ input) override;
    /** @brief Performs backward propagation
     * @param error the error vector
     * @return the gradient vector
     */
    vectorIn_ backward(vectorOut_ error) override;
    /** @brief No parameters to fit in an activation function, so this method does nothing */
    void fit(float_t learning_rate) override {}
};
}
