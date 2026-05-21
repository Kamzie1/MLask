#include "FullyConnectedLayer.hpp"
#include "Layer.hpp"
#include "Model.hpp"
#include <Eigen/Core>
#include <cstdlib>
#include <memory>
#include <random>
#include <iostream>
using namespace mlask;

float w = 0.3;
float b = 0;

#define SIZE 100
#define EPOCHS 10000

vectorOut_ err(vectorOut_ result, vectorOut_ expected){
    return result - expected;
}

int main(){
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.5,0.5);
    Model model(EPOCHS, true);
    model.addFullyConnectedLayer<1, 1>();

    std::vector<float> Y;
    Y.resize(SIZE + 1);
    for(int x = 0;x<=SIZE;x++){
        Y[x] = w * x + b + distribution(generator);
    }

    float_t learning_rate = 0.001;
    for(std::size_t epochs=0; epochs < EPOCHS; epochs++){
        for(int x = 0;x<=SIZE;x++){
            Eigen::Matrix<float_t, 1, 1> input;
            input << (x-(SIZE/2.f))/(SIZE/2.f);
            Eigen::Matrix<float_t, 1, 1> expected;
            expected << Y[x];
            model.backprop(input, expected, err);
        }
        model.fit(learning_rate);
    }
    model[0]->print(std::cout) << std::endl;
}
