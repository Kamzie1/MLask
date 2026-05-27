#pragma once

namespace mlask{

///@brief List of activation functions built in MLask.
#define LIST_OF_ACTIVATION_FUNCTIONS\
    REGISTER(Relu)

///@brief Enum of activation functions built in MLask.
enum class InternalActivationFunction{
    #define REGISTER(X) X,
        LIST_OF_ACTIVATION_FUNCTIONS
    #undef REGISTER
};

}
