#include <Eigen/Core>
#include <stdfloat>
#include <cstddef>
#include <cstdint>

namespace mlask {
    /**
     * All types used in this project are coded here,
     * you can always override them in any file
     */

    using float_t = float;
    using size_t = std::size_t;
    using int_t = std::int32_t;
    using uint_t = std::uint32_t;
    using vectorIn = Eigen::Matrix<float_t, Eigen::Dynamic, 1>;
    using vectorOut = Eigen::Matrix<float_t, Eigen::Dynamic, 1>;
    using err_function = float_t (*)(float_t, float_t);
    using actfunc = std::function<float_t(float_t)>;
    using labeling_function = int_t (*)(float_t); //a function that assigns neural network result a label
    using matrixIn = std::vector<std::vector<float_t>>;
    using matrixOut = std::vector<std::vector<float_t>>;
}
