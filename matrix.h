//
// Created by arvie on 09/11/25.
//

#ifndef GRAPHGEN_MATRIX_H
#define GRAPHGEN_MATRIX_H

#include <vector>
#include <iostream>
#include <optional>

using namespace std;

template<typename T>
class Matrix {
    /*
     * All the tokenizations can either be 1D or 2D, so this is a simple wrapper around a 1D vector
     * so that we can do multi-dim indexing.
     */
public:
    Matrix(){} // Default constructor since we dont know shape until resize generally
    Matrix(size_t size) : rows_(size), cols_(1), data_(size) {}  // Constructor for 1D array
    Matrix(size_t rows, size_t cols, optional<T> init = nullopt) : rows_(rows), cols_(cols), data_(rows * cols) {
        if (init.has_value()) {
            std::fill(data_.begin(), data_.end(), init.value());
        }
    }  // Constructor for 2D matrix

    void resize(size_t size, optional<T> init = nullopt) {
        rows_ = size;
        cols_ = 1;
        data_.resize(size);
        if (init.has_value()) {
            std::fill(data_.begin(), data_.end(), init.value());
        }
    }
    void resize(size_t rows, size_t cols, optional<T> init = nullopt) {
        rows_ = rows;
        cols_ = cols;
        data_.resize(rows * cols);
        if (init.has_value()) {
            std::fill(data_.begin(), data_.end(), init.value());
        }
    }

    vector<int> shape() const {
        return {static_cast<int>(rows_), static_cast<int>(cols_)};
    }

    T& operator()(size_t i) {
        if (cols_ != 1) { // Handle error: accessing 2D as 1D
            throw std::runtime_error("Attempted 1D access on 2D matrix");
        }
        return data_[i];
    }
    const T& operator()(size_t i) const {
        if (cols_ != 1) { // Handle error: accessing 2D as 1D
            throw std::runtime_error("Attempted 1D access on 2D matrix");
        }
        return data_[i];
    }

    T& operator()(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }
    const T& operator()(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    bool is1D() const { return cols_ == 1; }
    bool is2D() const { return cols_ > 1; }

private:
    size_t rows_, cols_;
    std::vector<T> data_; // Single underlying 1D data store
};
#endif //GRAPHGEN_MATRIX_H
