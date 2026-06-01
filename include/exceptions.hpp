#pragma once
#include <stdexcept>

namespace mlask{
/** @brief Errors associated with Neural Network architecture e.g. wrong connection beetwen layers. */
class ArchitectureError : public std::logic_error{
    std::size_t layerId_;
public:
    /** @param msg message describing an error
    * @param layerId id of an layer the error happend
    * @param layer the string representation of a layer that failed
    */
    ArchitectureError(const std::string& msg, std::size_t layerId):
        std::logic_error(msg), layerId_(layerId){} 
    const char* what() const noexcept override { return std::logic_error::what(); }
    std::size_t layerId() { return layerId_; }
};

/** @brief Errors associated with ONNX export */
class ExportError : public ArchitectureError{
public:
    /** @param msg message describing an error
    * @param layerId id of an layer the error happend
    * @param layer the string representation of a layer that failed
    */
    ExportError(const std::string& msg, std::size_t layerId) : ArchitectureError(msg, layerId) {}
};
}
