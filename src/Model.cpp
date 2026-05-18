#include "Model.hpp"
#include <Eigen/Core>
#include <memory>

namespace mlask {
Model::Model(std::size_t size): layers_(size) {}
void Model::addLayer(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

Eigen::Matrix<float_t, Eigen::Dynamic, 1>
Model::forward(Eigen::Matrix<float_t, Eigen::Dynamic, 1> input) const {
    for (const std::unique_ptr<Layer> &layer : layers_) {
        input = layer->forward(input);
    }
    return input;
}

void Model::backprop(Eigen::Matrix<float_t, Eigen::Dynamic, 1> input, Eigen::Matrix<float_t, Eigen::Dynamic, 1> expected, err_function err){
    for (const std::unique_ptr<Layer> &layer : layers_) {
        input = layer->forward(input);
    }
    Eigen::Matrix<float_t, Eigen::Dynamic, 1> error = err(expected, input);
    for (auto it = layers_.rbegin(); it!=layers_.rend(); ++it){
        error = (*it).get()->backward(error);
    }
}
void Model::fit(float_t learning_rate){
    for (const std::unique_ptr<Layer> &layer : layers_) {
        layer->fit(learning_rate);
    }
}
} // namespace mlask
