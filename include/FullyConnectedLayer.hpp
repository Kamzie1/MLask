#pragma once
#include "Layer.hpp"
#include "Eigen/Dense"
#include <Eigen/Core>
#include <cstddef>
#include <iomanip>
namespace mlask{

/**
 * @brief Class representing fully connected layer.
 * @brief Meaning a layer with in number of neurons as an input and out number of neurons as an output,
 * @brief where all the neurons are connected with each other.
 * @tparam in The number of neurons in the input
 * @tparam out The number of neurons in the output
 */
template<std::size_t in, std::size_t out>
class FullyConnectedLayer : public Layer{
private:
    Eigen::Matrix<float_t, out, in> weights_;
    Eigen::Matrix<float_t, out, 1> bias_;
     vectorIn_ input_;

    Eigen::Matrix<float_t, out, in> weightsChange_;
    Eigen::Matrix<float_t, out, 1> biasChange_;
    std::size_t epochs_;
public:
    /// @brief A constructor for fully connected layer, initializes weights and biases with random values, and changes with zeros
    FullyConnectedLayer(){
        weights_.setRandom();
        bias_.setRandom();
        weightsChange_.setZero();
        biasChange_.setZero();
        epochs_ = 0;
        in_ = in;
        out_ = out;
    }
    /** @brief A function defining moving foraward in neural network */
    vectorOut_ forward(vectorIn_ input) override;
    /** @brief A backtrack for backpropagation algorithm */
    vectorIn_ backward(vectorOut_ error) override;
    /** @brief A function that updates weights and biases */
    void fit(float_t learning_rate) override;

    /**
     * @brief A function that tries to convert the layer to ONNX format, returns true if successful, false otherwise
     * @param graph A pointer to the ONNX graph to which the layer should be added
     * @param input The name of the input tensor for this layer in the ONNX graph
     * @param output The name of the output tensor for this layer in the ONNX graph
     */
    bool tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output) const override;

    /* @brief Returns a string representation of the layer */
    std::string str() const override;
    /* getters */
    Eigen::Matrix<float_t, out, in> weights(){ return weights_; }
    Eigen::Matrix<float_t, in, 1> bias(){ return bias_; }
};

template <std::size_t in, std::size_t out>
vectorOut_  FullyConnectedLayer<in, out>::forward(vectorIn_ input){
    epochs_ += 1;
    input_ = input;
    return  weights_ * input + bias_;
}

template <std::size_t in, std::size_t out>
vectorIn_  FullyConnectedLayer<in, out>::backward(vectorOut_ error){
    biasChange_ = biasChange_ + error;
    weightsChange_ = weightsChange_ + error * input_.transpose();
    return weights_.transpose() * error;
}

template <std::size_t in, std::size_t out>
void FullyConnectedLayer<in, out>::fit(float_t learning_rate){
    weights_ = weights_ - (weightsChange_ / epochs_ * learning_rate);
    bias_ = bias_ - (biasChange_ / epochs_ * learning_rate);
    weightsChange_.setZero();
    biasChange_.setZero();
    epochs_ = 0;
}

template <std::size_t in, std::size_t out>
bool FullyConnectedLayer<in, out>::tryConvertToONNX(onnx::GraphProto* graph, std::string input, std::string output)const{
    onnx::TensorProto* w = graph->add_initializer();
    w->set_name(input + "_weight");
    w->set_data_type(onnx::TensorProto::FLOAT);
    w->add_dims(in);
    w->add_dims(out);
    constexpr int weightsLayout = (in > 1) ? Eigen::RowMajor : Eigen::ColMajor;
    Eigen::Matrix<float, out, in, weightsLayout | Eigen::DontAlign> onnxWeights = weights_;
    w->set_raw_data(onnxWeights.data(), onnxWeights.size() * sizeof(float_t));

    onnx::TensorProto* b = graph->add_initializer();
    b->set_name(input + "_bias");
    b->set_data_type(onnx::TensorProto::FLOAT);
    b->add_dims(out);
    b->set_raw_data(bias_.data(), bias_.size() * sizeof(float_t));

    onnx::NodeProto* matmul_node = graph->add_node();
    matmul_node->set_op_type("MatMul");
    matmul_node->add_input(input);
    matmul_node->add_input(input + "_weight");
    matmul_node->add_output(input + "_result");

    onnx::NodeProto* add_node = graph->add_node();
    add_node->set_op_type("Add");
    add_node->add_input(input + "_result");
    add_node->add_input(input + "_bias");
    add_node->add_output(output);
    return true;
}

template<std::size_t in, std::size_t out>
std::string FullyConnectedLayer<in, out>::str() const{
    std::stringstream weights_stream;
    weights_stream << std::fixed << std::setprecision(3) << weights_;
    std::stringstream bias_stream;
    bias_stream << std::fixed << std::setprecision(3) << bias_;
    return "FullyConnectedLayer: " + std::to_string(in) + " -> " + std::to_string(out) + "\nWeights:\n" + weights_stream.str() + "\nBias:\n" + bias_stream.str();
}
}
