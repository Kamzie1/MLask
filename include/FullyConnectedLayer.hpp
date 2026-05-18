#pragma once
#include "Layer.hpp"
#include "Eigen/Dense"
#include <ostream>

#define vectorIn FullyConnectedLayer<in_,out_>::vectorIn_
#define vectorOut FullyConnectedLayer<in_,out_>::vectorOut_

namespace mlask{

template<std::size_t in_, std::size_t out_>
class FullyConnectedLayer : public Layer{
private:
    Eigen::Matrix<float_t, out_, in_> weights_;
    Eigen::Matrix<float_t, out_, 1> bias_;
    Eigen::Matrix<float_t, in_, 1> input_;

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
    vectorOut_ forward(vectorIn_ input) override;
    vectorIn_ backward(vectorOut_ error) override;
    void fit(float_t learning_rate) override;

    Eigen::Matrix<float_t, out_, in_> weights(){ return weights_; }
    Eigen::Matrix<float_t, in_, 1> bias(){ return bias_; }

    std::ostream& print(std::ostream& os)const{return os<<weights_<<std::endl<<bias_<<std::endl;}
    template<std::size_t in, std::size_t out>
    friend std::ostream& operator<<(std::ostream& os, const FullyConnectedLayer<in, out>& fullyConnectedLayer);
};

template <std::size_t in_, std::size_t out_>
vectorOut  FullyConnectedLayer<in_, out_>::forward(vectorIn input){
    epochs_ += 1;
    input_ = input;
    return  weights_ * input + bias_;
}

template <std::size_t in_, std::size_t out_>
vectorIn  FullyConnectedLayer<in_, out_>::backward(vectorOut error){
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

template<std::size_t in, std::size_t out>
std::ostream&  operator<<(std::ostream& os, const FullyConnectedLayer<in, out>& fullyConnectedLayer){
    return fullyConnectedLayer.print(os);
}

}
