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
    using vectorIn_ = Eigen::Matrix<float_t, Eigen::Dynamic, 1>;
    using vectorOut_ = Eigen::Matrix<float_t, Eigen::Dynamic, 1>;
    using err_function = vectorOut_ (*)(vectorOut_, vectorOut_);
    using actfunc = std::function<float_t(float_t)>;
}
