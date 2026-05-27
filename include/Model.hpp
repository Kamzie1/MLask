#pragma once
#include "Layer.hpp"
#include "InternalActivationFunction.hpp"
#include <Eigen/Core>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>
#include "FullyConnectedLayer.hpp"
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
class Model {
  private:
    std::size_t in_;
    std::size_t out_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::size_t epochs_;
    bool log_;
    std::size_t epoch_ = 0;

  public:
    ///@brief Constructs a model
    /// @param in number of input neurons
    /// @param out number of output neurons
    /// @param size number of layers in the model (optional for optimization)
    /// @param epochs number of epochs to train the model (optional for logging)
    /// @param log whether to log the training process (optional for logging)
    Model(std::size_t in, std::size_t out, std::size_t size = 0, std::size_t epochs=0, bool log=false);

    ///@brief Adds a layer to the model(Layer is an abstract class, see Layer.hpp for more details)
    /// @param layer The layer to add
    void addLayer(std::unique_ptr<Layer> layer);

    ///@brief creates and adds fullyConnectedLayer with given in and out neurons
    /// @param in number of input neurons
    /// @param out number of output neurons
    template<std::size_t in, std::size_t out>
    void addFullyConnectedLayer();
    ///@brief creates and adds ActivationFunction layer with give function form and derived form
    /// @param func The activation function
    /// @param derv The derivative of the activation function
    void addActivationFunctionWithLambdas( actfunc func, actfunc derv);
    ///@brief creates and adds ActivationFunction layer of type activationFunction
    /// @param activationFunction The type of activation function to add
    void addActivationFunction(InternalActivationFunction activationFunction);

    ///@brief performs backpropagation algorithm
    /// @param input the input vector for the model
    /// @param expected the expected output vector for the given input
    /// @param err the error function to calculate the error between the expected and actual output
    void backprop(vectorIn_ input, vectorOut_ expected, err_function err);
    ///@brief Fits the entire model(every layer added)
    /// @param learning_rate The learning rate for the model
    void fit(float_t learning_rate);
    ///@brief calculates the output for given input
    /// @param input The input vector for the model
    /// @return The output vector for the given input
    vectorOut_ forward(vectorIn_ input) const;

    ///@brief gets the layer at a given index
    /// @param index The index of the layer to get
    /// @return A pointer to the layer at the given index
    const Layer* getLayer(std::size_t index)const{ return layers_[index].get(); }
    const Layer* operator[](std::size_t index)const { return getLayer(index); }

    ///@brief exports the model to a file in ONNX format
    /// @param path The path to the file where the model will be saved
    /// @param name The name of the model to be used in the ONNX file (optional, default is "MLask Model")
    void exportToONNX(std::filesystem::path path, std::string name = "MLask Model") const;

private:
    ///@brief converts the model to ONNX format
    /// @param model The ONNX model to convert to
    /// @param name The name of the model to be used in the ONNX file
    /// @return true if the conversion was successful, false otherwise
    bool tryConvertToONNX(onnx::ModelProto& model, std::string name)const;
    ///@brief gets the output size of the last layer in the model. For validation purposes.
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
        throw std::invalid_argument("Input size of the layer does not match the output size of the previous layer");
    }
    else{
        layers_.push_back(std::make_unique<FullyConnectedLayer<in, out>>());
        LOG("Added FullyConnected Layer");
    }
}

} // namespace mlask
