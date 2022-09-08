/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_BSPLINEBASIS_H
#define SPLINTER_BSPLINEBASIS_H

#include "bsplinebasis1d.h"
#include "definitions.h"
#include "mykroneckerproduct.h"
#include <unsupported/Eigen/KroneckerProduct>

namespace SPLINTER {

class BSplineBasis {
public:
    BSplineBasis(
        std::vector<std::vector<double>> const& knotVectors,
        std::vector<unsigned int> basisDegrees);

    // Evaluation
    template <class x_type, class callable>
    auto eval(x_type const& x, callable tail) const {
        if constexpr (std::is_floating_point_v<x_type>) {
            return tail(bases[0].eval(x));
        } else {
            assert(!bases.empty());

            SparseVector product = bases[0].eval(x[0]);

            for (int i = 1, I = getNumVariables(); i < I; ++i) {
                product = kroneckerProduct(product, bases[i].eval(x[i])).eval();
            }

            return tail(std::move(product));
        }
    }

    DenseMatrix evalBasisJacobianOld(DenseVector& x) const; // Depricated
    SparseMatrix evalBasisJacobian(DenseVector& x) const;
    SparseMatrix evalBasisJacobian2(
        DenseVector& x) const; // A bit slower than evaBasisJacobianOld()
    SparseMatrix evalBasisHessian(DenseVector& x) const;

    // Knot vector manipulation
    SparseMatrix refineKnots();
    SparseMatrix refineKnotsLocally(DenseVector x);
    SparseMatrix decomposeToBezierForm();
    SparseMatrix
    insertKnots(double tau, unsigned int dim, unsigned int multiplicity = 1);

    // Getters
    unsigned int getNumVariables() const { return bases.size(); }

    BSplineBasis1D getSingleBasis(int dim);
    std::vector<std::vector<double>> getKnotVectors() const;
    std::vector<double> getKnotVector(int dim) const;

    std::vector<unsigned int> getBasisDegrees() const;
    unsigned int getBasisDegree(unsigned int dim) const;
    unsigned int getNumBasisFunctions() const;
    unsigned int getNumBasisFunctions(unsigned int dim) const;
    std::vector<unsigned int> getNumBasisFunctionsTarget() const;

    double getKnotValue(int dim, int index) const;
    unsigned int getKnotMultiplicity(unsigned int dim, double tau) const;
    unsigned int getLargestKnotInterval(unsigned int dim) const;

    int supportedPrInterval() const;

    bool insideSupport(DenseVector& x) const;
    std::vector<double> getSupportLowerBound() const;
    std::vector<double> getSupportUpperBound() const;

    // Support related
    SparseMatrix
    reduceSupport(std::vector<double>& lb, std::vector<double>& ub);

    std::vector<unsigned int> getNumBasisFunctionsPerVariable() const {
        std::vector<unsigned int> ret;
        for (unsigned int i = 0; i < getNumVariables(); i++)
            ret.push_back(getNumBasisFunctions(i));
        return ret;
    }

private:
    std::vector<BSplineBasis1D> bases;

    friend bool operator==(BSplineBasis const& lhs, BSplineBasis const& rhs);
};

} // namespace SPLINTER

#endif // SPLINTER_BSPLINEBASIS_H
