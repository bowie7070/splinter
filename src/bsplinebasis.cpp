/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "bsplinebasis.h"
#include <iostream>

namespace SPLINTER {

SparseMatrix BSplineBasis::evalBasisJacobian2(DenseVector& x) const {
    auto const numVariables = getNumVariables();

    // Jacobian basis matrix
    SparseMatrix J(getNumBasisFunctions(), numVariables);

    // Evaluate B-spline basis functions before looping
    std::vector<SparseVector> funcValues(numVariables);
    std::vector<SparseVector> gradValues(numVariables);

    for (unsigned int i = 0; i < numVariables; ++i) {
        funcValues[i] = bases[i].eval(x(i));
        gradValues[i] = bases[i].evalFirstDerivative(x(i));
    }

    // Calculate partial derivatives
    for (unsigned int i = 0; i < numVariables; i++) {
        std::vector<SparseVector> values(numVariables);

        for (unsigned int j = 0; j < numVariables; j++) {
            if (j == i)
                values[j] = gradValues[j]; // Differentiated basis
            else
                values[j] = funcValues[j]; // Normal basis
        }

        SparseVector Ji = kroneckerProductVectors(values);

        // Fill out column
        for (SparseVector::InnerIterator it(Ji); it; ++it)
            J.insert(it.row(), i) = it.value();
    }

    return J;
}

SparseMatrix BSplineBasis::evalBasisHessian(DenseVector& x) const {
    auto const numVariables = getNumVariables();

    // Hessian basis matrix
    /* Hij = B1 x ... x DBi x ... x DBj x ... x Bn
     * (Hii = B1 x ... x DDBi x ... x Bn)
     * Where B are basis functions evaluated at x,
     * DB are the derivative of the basis functions,
     * and x is the kronecker product.
     * Hij is in R^(numBasisFunctions x 1)
     * so that basis hessian H is in R^(numBasisFunctions*numInputs x numInputs)
     * The real B-spline Hessian is calculated as (c^T x 1^(numInputs x 1))*H
     */
    SparseMatrix H(getNumBasisFunctions() * numVariables, numVariables);
    //H.setZero(numBasisFunctions()*numInputs, numInputs);

    // Calculate partial derivatives
    // Utilizing that Hessian is symmetric
    // Filling out lower left triangular
    for (unsigned int i = 0; i < numVariables; i++) // row
    {
        for (unsigned int j = 0; j <= i; j++) // col
        {
            // One column in basis jacobian
            SparseMatrix Hi(1, 1);
            Hi.insert(0, 0) = 1;

            for (unsigned int k = 0; k < numVariables; k++) {
                SparseMatrix temp = Hi;
                SparseMatrix Bk;
                if (i == j && k == i) {
                    // Diagonal element
                    Bk = bases[k].evalDerivative(x(k), 2);
                } else if (k == i || k == j) {
                    Bk = bases[k].evalDerivative(x(k), 1);
                } else {
                    Bk = bases[k].eval(x(k));
                }
                Hi = kroneckerProduct(temp, Bk);
            }

            // Fill out column
            for (int k = 0; k < Hi.outerSize(); ++k)
                for (SparseMatrix::InnerIterator it(Hi, k); it; ++it) {
                    if (it.value() != 0) {
                        int row = i * getNumBasisFunctions() + it.row();
                        int col = j;
                        H.insert(row, col) = it.value();
                    }
                }
        }
    }

    H.makeCompressed();

    return H;
}

SparseMatrix BSplineBasis::insertKnots(
    double tau, unsigned int dim, unsigned int multiplicity) {
    auto const numVariables = getNumVariables();

    SparseMatrix A(1, 1);
    //    A.resize(1,1);
    A.insert(0, 0) = 1;

    // Calculate multivariate knot insertion matrix
    for (unsigned int i = 0; i < numVariables; i++) {
        SparseMatrix temp = A;
        SparseMatrix Ai;

        if (i == dim) {
            // Build knot insertion matrix
            Ai = bases[i].insertKnots(tau, multiplicity);
        } else {
            // No insertion - identity matrix
            int m = bases[i].getNumBasisFunctions();
            Ai.resize(m, m);
            Ai.setIdentity();
        }

        //A = kroneckerProduct(temp, Ai);
        A = myKroneckerProduct(temp, Ai);
    }

    A.makeCompressed();

    return A;
}

SparseMatrix BSplineBasis::refineKnots() {
    auto const numVariables = getNumVariables();

    SparseMatrix A(1, 1);
    A.insert(0, 0) = 1;

    for (unsigned int i = 0; i < numVariables; i++) {
        SparseMatrix temp = A;
        SparseMatrix Ai   = bases[i].refineKnots();

        //A = kroneckerProduct(temp, Ai);
        A = myKroneckerProduct(temp, Ai);
    }

    A.makeCompressed();

    return A;
}

SparseMatrix BSplineBasis::refineKnotsLocally(DenseVector x) {
    auto const numVariables = getNumVariables();

    SparseMatrix A(1, 1);
    A.insert(0, 0) = 1;

    for (unsigned int i = 0; i < numVariables; i++) {
        SparseMatrix temp = A;
        SparseMatrix Ai   = bases[i].refineKnotsLocally(x(i));

        //A = kroneckerProduct(temp, Ai);
        A = myKroneckerProduct(temp, Ai);
    }

    A.makeCompressed();

    return A;
}

SparseMatrix BSplineBasis::decomposeToBezierForm() {
    auto const numVariables = getNumVariables();

    SparseMatrix A(1, 1);
    A.insert(0, 0) = 1;

    for (unsigned int i = 0; i < numVariables; i++) {
        SparseMatrix temp = A;
        SparseMatrix Ai   = bases[i].decomposeToBezierForm();

        //A = kroneckerProduct(temp, Ai);
        A = myKroneckerProduct(temp, Ai);
    }

    A.makeCompressed();

    return A;
}

SparseMatrix
BSplineBasis::reduceSupport(std::vector<double>& lb, std::vector<double>& ub) {
    auto const numVariables = getNumVariables();

    if (lb.size() != ub.size() || lb.size() != numVariables)
        throw Exception(
            "BSplineBasis::reduceSupport: Incompatible dimension of domain bounds.");

    SparseMatrix A(1, 1);
    A.insert(0, 0) = 1;

    for (unsigned int i = 0; i < numVariables; i++) {
        SparseMatrix temp = A;
        SparseMatrix Ai;

        Ai = bases[i].reduceSupport(lb[i], ub[i]);

        //A = kroneckerProduct(temp, Ai);
        A = myKroneckerProduct(temp, Ai);
    }

    A.makeCompressed();

    return A;
}

std::vector<unsigned int> BSplineBasis::getBasisDegrees() const {
    std::vector<unsigned int> degrees;
    for (auto const& basis : bases)
        degrees.push_back(basis.getBasisDegree());
    return degrees;
}

unsigned int BSplineBasis::getBasisDegree(unsigned int dim) const {
    return bases[dim].getBasisDegree();
}

unsigned int BSplineBasis::getNumBasisFunctions(unsigned int dim) const {
    return bases[dim].getNumBasisFunctions();
}

unsigned int BSplineBasis::getNumBasisFunctions() const {
    auto const numVariables = getNumVariables();

    unsigned int prod = 1;
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        prod *= bases[dim].getNumBasisFunctions();
    }
    return prod;
}

BSplineBasis1D BSplineBasis::getSingleBasis(int dim) {
    return bases[dim];
}

std::vector<double> BSplineBasis::getKnotVector(int dim) const {
    return bases[dim].getKnotVector();
}

std::vector<std::vector<double>> BSplineBasis::getKnotVectors() const {
    auto const numVariables = getNumVariables();

    std::vector<std::vector<double>> knots;
    for (unsigned int i = 0; i < numVariables; i++)
        knots.push_back(bases[i].getKnotVector());
    return knots;
}

unsigned int
BSplineBasis::getKnotMultiplicity(unsigned int dim, double tau) const {
    return bases[dim].knotMultiplicity(tau);
}

double BSplineBasis::getKnotValue(int dim, int index) const {
    return bases[dim].getKnotValue(index);
}

unsigned int BSplineBasis::getLargestKnotInterval(unsigned int dim) const {
    return bases[dim].indexLongestInterval();
}

std::vector<unsigned int> BSplineBasis::getNumBasisFunctionsTarget() const {
    auto const numVariables = getNumVariables();

    std::vector<unsigned int> ret;
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        ret.push_back(bases[dim].getNumBasisFunctionsTarget());
    }
    return ret;
}

int BSplineBasis::supportedPrInterval() const {
    auto const numVariables = getNumVariables();

    int ret = 1;
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        ret *= (bases[dim].getBasisDegree() + 1);
    }
    return ret;
}

bool BSplineBasis::insideSupport(DenseVector& x) const {
    auto const numVariables = getNumVariables();

    for (unsigned int dim = 0; dim < numVariables; dim++) {
        if (!bases[dim].insideSupport(x(dim))) {
            return false;
        }
    }
    return true;
}

std::vector<double> BSplineBasis::getSupportLowerBound() const {
    auto const numVariables = getNumVariables();

    std::vector<double> lb;
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        std::vector<double> knots = bases[dim].getKnotVector();
        lb.push_back(knots.front());
    }
    return lb;
}

std::vector<double> BSplineBasis::getSupportUpperBound() const {
    auto const numVariables = getNumVariables();

    std::vector<double> ub;
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        std::vector<double> knots = bases[dim].getKnotVector();
        ub.push_back(knots.back());
    }
    return ub;
}

} // namespace SPLINTER
