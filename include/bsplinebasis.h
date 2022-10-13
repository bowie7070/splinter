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
#include <variant>

namespace SPLINTER {

struct basis1d_eval_uncached {
    template <class B>
    SparseVector operator()(B const& basis, int, double x) const {
        return basis.eval(x);
    }
};

template <class cache_type = std::map<std::tuple<int, double>, SparseVector>>
struct basis1d_eval_cached {
    mutable cache_type cache;

    template <class B>
    SparseVector const& operator()(B const& basis, int i, double x) const {
        auto pos = cache.find({i, x});
        if (pos == cache.end()) {
            std::tie(pos, std::ignore) = cache.insert({{i, x}, basis.eval(x)});
        }
        return pos->second;
    }
};

template <unsigned _degree>
class BSplineBasis {
public:
    static constexpr unsigned degree = _degree;

    using bases_type = std::vector<BSplineBasis1D<degree>>;

    BSplineBasis(std::vector<std::vector<double>> const& knotVectors) {
        unsigned int numVariables = knotVectors.size();

        // Set univariate bases
        for (unsigned int i = 0; i < numVariables; i++) {
            bases.emplace_back(knotVectors[i]);

            // Adjust target number of basis functions used in e.g. refinement
            if (numVariables > 2) {
                // One extra knot is allowed
                bases[i].setNumBasisFunctionsTarget(
                    (degree + 1) + 1); // Minimum degree+1
            }
        }
    }

    // Evaluation
    template <class x_type, class eval_fn, class callable>
    auto eval(x_type const& x, eval_fn& eval, callable tail) const {
        if constexpr (std::is_floating_point_v<x_type>) {
            return tail(eval(bases[0], 0, x));
        } else {
            assert(!bases.empty());

            SparseMatrix product = eval(bases[0], 0, x[0]);
            SparseMatrix product_prev;

            for (int i = 1, I = getNumVariables(); i < I; ++i) {
                product.swap(product_prev);
                product =
                    kroneckerProduct(product_prev, eval(bases[i], i, x[i]));
            }

            return tail(std::move(product));
        }
    }

    DenseMatrix evalBasisJacobianOld(DenseVector& x) const {
        auto const numVariables = getNumVariables();
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
    SparseMatrix evalBasisJacobian(DenseVector& x) const {
        auto const numVariables = getNumVariables();

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

    std::vector<double> const& getKnotVector(int dim) const {
        return bases[dim].getKnotVector();
    }

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
    bases_type bases;

    friend bool operator==(BSplineBasis const& lhs, BSplineBasis const& rhs);
};

template <unsigned d>
SparseMatrix BSplineBasis<d>::evalBasisJacobian2(DenseVector& x) const {
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

template <unsigned d>
SparseMatrix BSplineBasis<d>::evalBasisHessian(DenseVector& x) const {
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

template <unsigned d>
SparseMatrix BSplineBasis<d>::insertKnots(
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

template <unsigned d>
SparseMatrix BSplineBasis<d>::refineKnots() {
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

template <unsigned d>
SparseMatrix BSplineBasis<d>::refineKnotsLocally(DenseVector x) {
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

template <unsigned d>
SparseMatrix BSplineBasis<d>::decomposeToBezierForm() {
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

template <unsigned d>
SparseMatrix BSplineBasis<d>::reduceSupport(
    std::vector<double>& lb, std::vector<double>& ub) {
    auto const numVariables = getNumVariables();

    if (lb.size() != ub.size() || lb.size() != numVariables)
        throw Exception(
            "BSplineBasis<d>::reduceSupport: Incompatible dimension of domain bounds.");

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

template <unsigned d>
unsigned int BSplineBasis<d>::getNumBasisFunctions(unsigned int dim) const {
    return bases[dim].getNumBasisFunctions();
}

template <unsigned d>
unsigned int BSplineBasis<d>::getNumBasisFunctions() const {
    auto const numVariables = getNumVariables();

    unsigned int prod = 1;
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        prod *= bases[dim].getNumBasisFunctions();
    }
    return prod;
}

template <unsigned d>
unsigned int
BSplineBasis<d>::getKnotMultiplicity(unsigned int dim, double tau) const {
    return bases[dim].knotMultiplicity(tau);
}

template <unsigned d>
double BSplineBasis<d>::getKnotValue(int dim, int index) const {
    return bases[dim].getKnotValue(index);
}

template <unsigned d>
unsigned int BSplineBasis<d>::getLargestKnotInterval(unsigned int dim) const {
    return bases[dim].indexLongestInterval();
}

template <unsigned d>
std::vector<unsigned int> BSplineBasis<d>::getNumBasisFunctionsTarget() const {
    auto const numVariables = getNumVariables();

    std::vector<unsigned int> ret;

    for (unsigned int dim = 0; dim < numVariables; dim++) {
        ret.push_back(bases[dim].getNumBasisFunctionsTarget());
    }

    return ret;
}

template <unsigned d>
int BSplineBasis<d>::supportedPrInterval() const {
    auto const numVariables = getNumVariables();

    int ret = 1;

    for (unsigned int dim = 0; dim < numVariables; dim++) {
        ret *= (bases[dim].getBasisDegree() + 1);
    }

    return ret;
}

template <unsigned d>
bool BSplineBasis<d>::insideSupport(DenseVector& x) const {
    auto const numVariables = getNumVariables();

    for (unsigned int dim = 0; dim < numVariables; dim++) {
        if (!bases[dim].insideSupport(x(dim))) {
            return false;
        }
    }
    return true;
}

template <unsigned d>
std::vector<double> BSplineBasis<d>::getSupportLowerBound() const {
    auto const numVariables = getNumVariables();

    std::vector<double> lb;
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        lb.push_back(bases[dim].knot_front());
    }

    return lb;
}

template <unsigned d>
std::vector<double> BSplineBasis<d>::getSupportUpperBound() const {
    auto const numVariables = getNumVariables();

    std::vector<double> ub;
    for (unsigned int dim = 0; dim < numVariables; dim++) {
        ub.push_back(bases[dim].knot_back());
    }
    return ub;
}

} // namespace SPLINTER

#endif // SPLINTER_BSPLINEBASIS_H
