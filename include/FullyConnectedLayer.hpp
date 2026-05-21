#pragma once
#include "Layer.hpp"
#include "Eigen/Dense"
#include <ostream>

namespace mlask{

///  Class representing fully connected layer.
///  Meaning a layer with in_ number of neurons as an input and out_ number of neurons as an output,
///  where all these neurons are connected with each other.
template<std::size_t in_, std::size_t out_>
class FullyConnectedLayer : public Layer{
private:
    Eigen::Matrix<float_t, out_, in_> weights_;
    Eigen::Matrix<float_t, out_, 1> bias_;
     vectorIn_ input_;

    Eigen::Matrix<float_t, out_, in_> weightsChange_;
    Eigen::Matrix<float_t, out_, 1> biasChange_;
    std::size_t epochs_;
public:
    FullyConnectedLayer(){
        weights_.setRandom();
        bias_.setRandom();
        weightsChange_.setZero();
        biasChange_.setZero();
        epochs_ = 0;
    }
    /// A function defining moving foraward in neural network
    vectorOut_ forward(vectorIn_ input) override;
    /// A backtrack for backpropagation algorithm
    vectorIn_ backward(vectorOut_ error) override;
    /// A function that updates weights and biases
    void fit(float_t learning_rate) override;

    Eigen::Matrix<float_t, out_, in_> weights(){ return weights_; }
    Eigen::Matrix<float_t, in_, 1> bias(){ return bias_; }

    std::ostream& print(std::ostream& os)const override{return os<<weights_<<std::endl<<bias_<<std::endl;}
    friend std::ostream& operator<<(std::ostream& os, const FullyConnectedLayer& fullyConnectedLayer){ return fullyConnectedLayer.print(os); }
};

template <std::size_t in_, std::size_t out_>
vectorOut_  FullyConnectedLayer<in_, out_>::forward(vectorIn_ input){
    epochs_ += 1;
    input_ = input;
    return  weights_ * input + bias_;
}

template <std::size_t in_, std::size_t out_>
vectorIn_  FullyConnectedLayer<in_, out_>::backward(vectorOut_ error){
    biasChange_ = biasChange_ + error;
    weightsChange_ = weightsChange_ + error * input_.transpose();
    return weights_.transpose() * error;
}

template <std::size_t in_, std::size_t out_>
void FullyConnectedLayer<in_, out_>::fit(float_t learning_rate){
    weights_ = weights_ - (weightsChange_ / epochs_ * learning_rate);
    bias_ = bias_ - (biasChange_ / epochs_ * learning_rate);
    weightsChange_.setZero();
    biasChange_.setZero();
    epochs_ = 0;
}

}
