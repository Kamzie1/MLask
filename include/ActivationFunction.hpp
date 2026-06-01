#pragma once
#include "Layer.hpp"

namespace mlask{
/** @brief base class for Activation Functions, meant for simplicity in definition.
 * @details it strips children from fit method override must and also make it easy to define forrward and backward passes via activate and derived functions that act on a single value, rather than entire vector.
 */
class ActivationFunction : public Layer{
private:
    vectorIn input_;
    /** @brief activation function signature. see Relu.cpp
     * @param in x
     * @return y
     */
    virtual float_t activate(float_t input) = 0;
    /** @brief derived version of activate function. see Relu.cpp
     * @param in x
     * @return y
     */
    virtual float_t derived(float_t input) = 0;
public:
    /** @brief Defines a way to move forward in a neural network */
    vectorOut forward(vectorIn)override;
    /** @brief Defines a way to backpropagate error in backropagation algorithm */
    vectorIn backward(vectorOut)override;
    /** @brief Describes how the layer 'learns', meaning it defines how layer updates itself. */
    void fit(float_t learning_rate)override{};
    /** @brief Returns a string representation of the layer. */
    std::string str()const override{ return "Activation Function"; }
};
}
