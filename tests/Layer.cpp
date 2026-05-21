#include "Layer.hpp"
#include "Model.hpp"
#include <Eigen/Core>
#include <assert.h>
#include <iostream>
#include <memory>

class TestLayer : public mlask::Layer {
  public:
    vectorOut_ forward(vectorIn_ input) const override { return input; }
    void fit() override { return; }
    vectorOut_ backward(vectorIn_ error) override { return error; }
};

class AddingLayer : public mlask::Layer {
  private:
    mlask::float_t adder_;

  public:
    AddingLayer(int adder) : adder_(adder) {}
    vectorOut_ forward(vectorIn_ input) const override {
        input.array() += adder_;
        return input;
    }
    void fit() override { return; }
    vectorOut_ backward(vectorIn_ error) override { return error; }
};

void test_test_layer(){
    std::cout<<"----Test Layer----\n";
    mlask::Model model;

    model.addLayer(std::make_unique<AddingLayer>(1));
    model.addLayer(std::make_unique<AddingLayer>(2));
    model.addLayer(std::make_unique<AddingLayer>(3));
    Eigen::Matrix<mlask::float_t, 4, 1> input;
    Eigen::Matrix<mlask::float_t, 4, 1> result;
    input << 1, 2, 3, 4;
    result = model.forward(input);
    input.array()+=6;
    assert(input == result);
    std::cout<<"----PASSED----\n";
}


void test_multi_layer(){
    std::cout<<"----Multi Layer----\n";
    mlask::Model model;

    model.addLayer(std::make_unique<AddingLayer>(1));
    model.addLayer(std::make_unique<AddingLayer>(2));
    model.addLayer(std::make_unique<TestLayer>());
    Eigen::Matrix<mlask::float_t, 4, 1> input;
    Eigen::Matrix<mlask::float_t, 4, 1> result;
    input << 1, 2, 3, 4;
    result = model.forward(input);
    input.array()+=3;
    assert(input == result);
    std::cout<<"----PASSED----\n";
}
void test_addition_layer(){
    std::cout<<"----Addition Layer----\n";
    mlask::Model model;

    model.addLayer(std::make_unique<TestLayer>());
    model.addLayer(std::make_unique<TestLayer>());
    model.addLayer(std::make_unique<TestLayer>());
    Eigen::Matrix<mlask::float_t, 4, 1> input;
    Eigen::Matrix<mlask::float_t, 4, 1> result;
    input << 1, 2, 3, 4;
    result = model.forward(input);
    assert(input == result);
    std::cout<<"----PASSED----\n";
}

int main() {
    std::cout << "----Testing Layer----\n";
    test_test_layer();
    test_addition_layer();
    test_multi_layer();
}
