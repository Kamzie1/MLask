#pragma once
#include "GenericErrorFunction.hpp"
#include "Layer.hpp"
#include "InternalActivationFunction.hpp"
#include "ErrorFunction.hpp"
#include <Eigen/Core>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>
#include "FullyConnectedLayer.hpp"
#include "exceptions.hpp"
#include <iostream>
#include <filesystem>

#ifdef DEBUG

#define LOG(X) std::cout<<"[INFO] "<<(X)<<std::endl;
#define WARN(X) std::cout<<"[WARN] "<<(X)<<std::endl;
#define ERR(X) std::cerr<<"[ERROR] "<<(X)<<std::endl;

#else

#define LOG(X)
#define WARN(X)
#define ERR(X)

#endif


namespace mlask {
/** @brief Class representing a neural network model */
class Model {
  private:
    std::size_t in_;
    std::size_t out_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::size_t epochs_;
    bool log_;
    std::size_t epoch_ = 0;

  public:
    /**
     * @brief Constructs a model
     * @param in number of input neurons
     * @param out number of output neurons
     * @param size number of layers in the model (optional for optimization)
     * @param epochs number of epochs to train the model (optional for logging)
     * @param log whether to log the training process (optional for logging)
     */
    Model(std::size_t in, std::size_t out, std::size_t size = 0, std::size_t epochs=0, bool log=false);

    /**
     * @brief Adds a layer to the model(Layer is an abstract class, see Layer.hpp for more details)
     * @param layer The layer to add
     * @throws ArchitectureError if the input size of the layer does not match the output size of the previous layer
     */
    void addLayer(std::unique_ptr<Layer> layer);

    /**
     * @brief Adds a layer to the model(Layer is an abstract class, see Layer.hpp for more details)
     * @tparam layer The layer to add, a class needs to derive from layer abstract class, see layer.hpp and TLayer concept
     * @details This one doesn't check if in and out are valid
     */
    template<TLayer layer, typename... Args>
    void addLayer(Args&&... args);

    /**
     * @brief creates and adds fullyConnectedLayer with given in and out neurons
     * @tparam in number of input neurons
     * @tparam out number of output neurons
     * @throws ArchitectureError if the input size of the layer does not match the output size of the previous layer
     */
    template<std::size_t in, std::size_t out>
    void addFullyConnectedLayer();
    /**
     * @brief creates and adds ActivationFunction layer with give function form and derived form
     * @param func The activation function
     * @param derv The derivative of the activation function
     */
    void addLambdaActivationFunction( actfunc func, actfunc derv);
    /**
     * @brief creates and adds ActivationFunction layer of type activationFunction. Only mlask built-in are available
     * @param activationFunction The type of activation function to add
     * @throws ArchitectureError if the activation function type is not supported. See InternalActivationFunction.hpp for more details
     */
    void addActivationFunction(InternalActivationFunction activationFunction);

    /**
     * @brief performs backpropagation algorithm
     * @param input the input vector for the model
     * @param expected the expected output vector for the given input
     * @tparam err ErrorFunction used to calculate error, notice that this should be derived version
     */
    template<ErrorFunction err>
    void backprop(vectorIn input, vectorOut expected);

    /**
     * @brief Fits the entire model(every layer added)
     * @param learning_rate The learning rate for the model
     */
    void fit(float_t learning_rate);
    /**
     * @brief calculates the output for given input
     * @param input The input vector for the model
     * @return The output vector for the given input
     */
    vectorOut forward(vectorIn input) const;

    /**
     * @brief calculates an error of a model
     * @param input The input vector for the model
     * @param expected The expected vector of values
     * @tparam err the error function (see ErrorFunction.hpp or include/error_functions)
     * @return the error based on an error function
     */
    template<ErrorFunction err>
    [[nodiscard]] float_t error(vectorIn input, vectorOut expected)const;


    /**
     * @brief calculates an error of a model based on many inputs
     * @param input The input matrix for the model
     * @param expected The expected matrix of values
     * @tparam err the error function (see ErrorFunction.hpp or include/error_functions)
     * @return the error based on an error function
     */
    template<ErrorFunction err>
    [[nodiscard]] float_t whole_error(Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic> input, Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic> expected)const;

    /**
     * @brief gets the layer at a given index
     * @param index The index of the layer to get
     * @return A pointer to the layer at the given index
     */
    const Layer* getLayer(std::size_t index)const{ return layers_[index].get(); }
    const Layer* operator[](std::size_t index)const { return getLayer(index); }

    /** @brief Returns a string representation of the model */
    [[nodiscard]] std::string str()const;

    /**
     * @brief exports the model to a file in ONNX format
     * @param path The path to the file where the model will be saved
     * @param name The name of the model to be used in the ONNX file (optional, default is "MLask Model")
     * @throws throws ExportError if the export failed
     */
    void exportToONNX(std::filesystem::path path, std::string name = "MLask Model") const;

private:
    /**
     * @brief converts the model to ONNX format
     * @param model The ONNX model to convert to
     * @param name The name of the model to be used in the ONNX file
     * @throws throws ExportError if the export failed
     * @return true if the conversion was successful, false otherwise
     */
    void convertToONNX(onnx::ModelProto& model, std::string name)const;
    /** @brief gets the output size of the last layer in the model. For validation purposes. */
    inline std::size_t lastOut()const{
        if(layers_.empty()){
            return in_;
        }
        return layers_.back()->getOut();
    }
};


template <std::size_t in, std::size_t out>
void Model::addFullyConnectedLayer(){
    std::size_t last_out = lastOut();
    if(in != last_out){
        ERR("Input size of the layer does not match the output size of the previous layer");
        throw ArchitectureError("Input size of the layer does not match the output size of the previous layer", layers_.size());
    }
    else{
        layers_.push_back(std::make_unique<FullyConnectedLayer<in, out>>());
        LOG("Added FullyConnected Layer");
    }
}

template<TLayer layer, typename... Args>
void Model::addLayer(Args&&... args){
    layers_.push_back(std::make_unique<layer>(std::forward<Args>(args)...));
}

template<ErrorFunction err>
float_t Model::error(vectorIn input, vectorOut expected)const{
    vectorOut result = forward(input);
    return GenericErrorFunction<err>::error_scalar(result, expected);
}

template<ErrorFunction err>
float_t Model::whole_error(Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic> inputs, Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic> expected_values)const{
    float_t error_value = 0;
    std::size_t size = inputs.rows();
    for(std::size_t i=0;i<size;i++){
        error_value += error<err>(inputs.row(i).transpose(), expected_values.row(i).transpose());
    }
    return error_value / size;
}

template<ErrorFunction err>
void Model::backprop(vectorIn input, vectorOut expected){
    for (const std::unique_ptr<Layer> &layer : layers_) {
        input = layer->forward(input);
    }
    vectorOut error = GenericErrorFunction<err>::error(input, expected);
    for (auto it = layers_.rbegin(); it!=layers_.rend(); ++it){
        error = (*it).get()->backward(error);
    }
}

} // namespace mlask
