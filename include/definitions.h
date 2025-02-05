/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_DEFINITIONS_H
#define SPLINTER_DEFINITIONS_H

#ifndef SPLINTER_API
#ifdef _MSC_VER
#define SPLINTER_API __declspec(dllexport)
#else
#define SPLINTER_API
#endif
#endif

#ifndef NDEBUG
#include <iomanip>
#include <iostream>
#endif // NDEBUG

#include <exception>
#include <stdexcept>
#include <vector>
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#include <Eigen/Dense>
#include <Eigen/Sparse>
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace SPLINTER {

// Eigen vectors
typedef Eigen::VectorXd DenseVector;
typedef Eigen::SparseVector<double> SparseVector;

// Eigen matrices
typedef Eigen::MatrixXd DenseMatrix;
typedef Eigen::SparseMatrix<double>
    SparseMatrix; // declares a column-major sparse matrix type of double

class Exception : public std::exception {
private:
    std::string __what;

public:
    Exception(std::string const& what) : __what(what) {}

    char const* what() const throw() { return this->__what.c_str(); }
};

} // namespace SPLINTER

#endif // SPLINTER_DEFINITIONS_H
