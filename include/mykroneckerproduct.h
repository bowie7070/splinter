/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_MYKRONECKERPRODUCT_H
#define SPLINTER_MYKRONECKERPRODUCT_H

#include "definitions.h"

namespace SPLINTER {

SparseMatrix myKroneckerProduct(SparseMatrix const& A, SparseMatrix const& B);

// Apply Kronecker product on several vectors or matrices
SparseVector kroneckerProductVectors(std::vector<SparseVector> const& vectors);
DenseVector kroneckerProductVectors(std::vector<DenseVector> const& vectors);
SparseMatrix
kroneckerProductMatrices(std::vector<SparseMatrix> const& matrices);

} // namespace SPLINTER

#endif // SPLINTER_MYKRONECKERPRODUCT_H
