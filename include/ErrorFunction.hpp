#pragma once
#include "types.hpp"
#include <concepts>
#include <utility>

namespace mlask{

/** @brief Concept representing an error function that contains a functor inside it 
 * @detials a class that needs to have an operator () with two float_t args, and which returns float_t(error)
*/
template<typename err>
concept ErrorFunction = requires(err e, float_t result, float_t expected){
    { std::as_const(e)(result, expected) } -> std::same_as<float_t>;
};

}
