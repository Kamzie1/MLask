#include "FullyConnectedLayer.hpp"
#include "Model.hpp"
#include <Eigen/Core>
#include <cassert>
#include <iostream>
#include <memory>
#include <ostream>
using namespace mlask;

void test_fit(){
    std::cout<<"Printing randomly fitted matrix 1x1: \n";
    FullyConnectedLayer<1,1> matrix1_1;
    std::cout<<matrix1_1.str()<<std::endl;

    std::cout<<"Printing randomly fitted matrix 4x1: \n";
    FullyConnectedLayer<4,1> matrix4_1;
    std::cout<<matrix4_1.str()<<std::endl;

    std::cout<<"Printing randomly fitted matrix 5x5: \n";
    FullyConnectedLayer<5,5> matrix5_5;
    std::cout<<matrix5_5.str()<<std::endl;
}

void test_forward(){
    std::cout<<"Testing forward function in FullyConnectedLayer\n";
    FullyConnectedLayer<1,1> matrix1_1;
    Eigen::Matrix<float_t, 1, 1> input;
    input << 1;
    std::cout<<"---------\n";
    std::cout<<matrix1_1.str()<<std::endl;
    std::cout<< matrix1_1.forward(input) << std::endl;
    std::cout<<"---------\n";
    FullyConnectedLayer<5,5> matrix5_5;
    Eigen::Matrix<float_t, 5, 1> input2;
    input.setConstant(1);
    std::cout<<"---------\n";
    std::cout<<matrix5_5.str()<<std::endl;
    std::cout<< matrix5_5.forward(input2) << std::endl;
    std::cout<<"---------\n";
}

void test_with_model(){
    std::cout<<"----Test Model with Fully Connected Layer----\n";
    mlask::Model model(1, 2);

    FullyConnectedLayer<1,1> layer;
    model.addLayer(std::make_unique<FullyConnectedLayer<1,1>>(layer));
    Eigen::Matrix<float_t, 1, 1> input;
    input << 1;
    std::cout<<"---------\n";
    std::cout<< model.forward(input) << std::endl;
    std::cout<<"---------\n";
    FullyConnectedLayer<1,2> layer2;
    model.addLayer(std::make_unique<FullyConnectedLayer<1,2>>(layer2));

    std::cout<<"---------\n";
    std::cout<< model.forward(input) << std::endl;
    std::cout<<"---------\n";
}


int main(){
    test_fit();
    test_forward();
    test_with_model();
}
