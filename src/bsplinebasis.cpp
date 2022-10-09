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

BSplineBasis::BSplineBasis(
    std::vector<std::vector<double>> const& knotVectors,
    std::vector<unsigned int> basisDegrees) :
    numVariables(knotVectors.size()) {
    if (knotVectors.size() != basisDegrees.size())
        throw Exception(
            "BSplineBasis::BSplineBasis: Incompatible sizes. Number of knot vectors is not equal to size of degree vector.");

    // Set univariate bases
    bases.clear();
    for (unsigned int i = 0; i < numVariables; i++) {
        bases.push_back(BSplineBasis1D(knotVectors[i], basisDegrees[i]));

        // Adjust target number of basis functions used in e.g. refinement
        if (numVariables > 2) {
            // One extra knot is allowed
            bases[i].setNumBasisFunctionsTarget(
                (basisDegrees[i] + 1) + 1); // Minimum degree+1
        }
    }
}

// Old implementation of Jacobian
DenseMatrix BSplineBasis::evalBasisJacobianOld(DenseVector& x) const {
    // Jacobian basis matrix
    DenseMatrix J;
    J.setZero(getNumBasisFunctions(), numVariables);

    // Calculate partial derivatives
    for (unsigned int i = 0; i < numVariables; i++) {
        // One column in basis jacobian
        DenseVector bi;
        bi.setOnes(1);
        for (unsigned int j = 0; j < numVariables; j++) {
            DenseVector temp = bi;
            DenseVector xi;
            if (j == i) {
                // Differentiated basis
                xi = bases[j].evalFirstDerivative(x(j));
            } else {
                // Normal basis
                xi = bases[j].eval(x(j));
            }

            bi = kroneckerProduct(temp, xi);
        }

        // Fill out column
        J.block(0, i, bi.rows(), 1) = bi.block(0, 0, bi.rows(), 1);
    }

    return J;
}

// NOTE: does not pass tests
SparseMatrix BSplineBasis::evalBasisJacobian(DenseVector& x) const {
    // Jacobian basis matrix
    SparseMatrix J(getNumBasisFunctions(), numVariables);
    //J.setZero(numBasisFunctions(), numInputs);

    // Calculate partial derivatives
    for (unsigned int i = 0; i < numVariables; ++i) {
        // One column in basis jacobian
        std::vector<SparseVector> values(numVariables);

        for (unsigned int j = 0; j < numVariables; ++j) {
            if (j == i) {
                // Differentiated basis
                values[j] = bases[j].evalDerivative(x(j), 1);
            } else {
                // Normal basis
                values[j] = bases[j].eval(x(j));
            }
        }

        SparseVector Ji = kroneckerProductVectors(values);

        // Fill out column
        assert(Ji.outerSize() == 1);
        for (SparseVector::InnerIterator it(Ji); it; ++it) {
            if (it.value() != 0)
                J.insert(it.row(), i) = it.value();
        }
        //J.block(0,i,Ji.rows(),1) = bi.block(0,0,Ji.rows(),1);
    }

    J.makeCompressed();

    return J;
}

SparseMatrix BSplineBasis::evalBasisJacobian2(DenseVector& x) const {
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
    std::vector<unsigned int> ret;
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        ret.push_back(bases[dim].getNumBasisFunctionsTarget());
    }
    return ret;
}

int BSplineBasis::supportedPrInterval() const {
    int ret = 1;
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        ret *= (bases[dim].getBasisDegree() + 1);
    }
    return ret;
}

bool BSplineBasis::insideSupport(DenseVector& x) const {
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        if (!bases[dim].insideSupport(x(dim))) {
            return false;
        }
    }
    return true;
}

std::vector<double> BSplineBasis::getSupportLowerBound() const {
    std::vector<double> lb;
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        std::vector<double> knots = bases[dim].getKnotVector();
        lb.push_back(knots.front());
    }
    return lb;
}

std::vector<double> BSplineBasis::getSupportUpperBound() const {
    std::vector<double> ub;
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        std::vector<double> knots = bases[dim].getKnotVector();
        ub.push_back(knots.back());
    }
    return ub;
}

} // namespace SPLINTER
