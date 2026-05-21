#pragma once

namespace mlask{

#define LIST_OF_ACTIVATION_FUNCTIONS\
    REGISTER(Relu)

enum class InternalActivationFunction{
    #define REGISTER(X) X,
        LIST_OF_ACTIVATION_FUNCTIONS
    #undef REGISTER
};

}
