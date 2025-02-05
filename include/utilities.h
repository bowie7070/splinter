/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_UTILITIES_H
#define SPLINTER_UTILITIES_H

#include <definitions.h>
#include <stdlib.h> // std::abs etc
#include <vector>

namespace SPLINTER {

// Compare two numbers
template <typename T>
bool assertNear(T x, T y, double tolAbs = 1e-8, double tolRel = 1e-8) {
    double dx   = std::abs(x - y);
    double xAbs = 0.5 * (std::abs(x) + std::abs(y));
    double err  = std::max(tolAbs, tolRel * xAbs);
    return dx < err;
}

std::vector<double> denseVectorToVector(DenseVector const& denseVec);

DenseVector vectorToDenseVector(std::vector<double> const& vec);

std::vector<std::vector<double>>
denseMatrixToVectorVector(DenseMatrix const& mat);

DenseMatrix
vectorVectorToDenseMatrix(std::vector<std::vector<double>> const& vec);

std::vector<double> linspace(double start, double stop, unsigned int num);

} // namespace SPLINTER

#endif // SPLINTER_UTILITIES_H
