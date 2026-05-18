#pragma once
#include "Layer.hpp"
#include <Eigen/Core>
#include <cstddef>
#include <memory>
#include <vector>

namespace mlask {
class Model {
  private:
    std::vector<std::unique_ptr<Layer>> layers_;

  public:
    Model() = default;
    Model(std::size_t size);
    void addLayer(std::unique_ptr<Layer> layer);
    void backprop(Eigen::Matrix<float_t, Eigen::Dynamic, 1> input,Eigen::Matrix<float_t, Eigen::Dynamic, 1> expected, err_function err);
    void fit(float_t leearning_rate);
    Eigen::Matrix<float_t, Eigen::Dynamic, 1> forward(Eigen::Matrix<float_t, Eigen::Dynamic, 1> input) const;

    const Layer* getLayer(std::size_t index)const{ return layers_[index].get(); }
    const Layer* operator[](std::size_t index)const { return getLayer(index); }
};

} // namespace mlask
