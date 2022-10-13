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

template <unsigned _degree, unsigned _variables>
class BSplineBasis {
public:
    static constexpr unsigned degree    = _degree;
    static constexpr unsigned variables = _variables;

    using bases_type = std::vector<BSplineBasis1D<degree>>;

    BSplineBasis(std::vector<std::vector<double>> const& knotVectors) {
        assert(variables == knotVectors.size());

        // Set univariate bases
        for (unsigned int i = 0; i < variables; i++) {
            bases.emplace_back(knotVectors[i]);

            // Adjust target number of basis functions used in e.g. refinement
            if constexpr (variables > 2) {
                // One extra knot is allowed
                bases[i].setNumBasisFunctionsTarget(
                    (degree + 1) + 1); // Minimum degree+1
            }
        }
    }

    // eval workspace memory
    mutable SparseMatrix product;
    mutable SparseMatrix product_prev;

    // Evaluation
    template <class x_type, class eval_fn, class callable>
    auto eval(x_type const& x, eval_fn& eval, callable tail) const {
        if constexpr (std::is_arithmetic_v<x_type>) {
            return tail(eval(bases[0], 0, x));

        } else if constexpr (variables == 1) {
            return tail(eval(bases[0], 0, x[0]));

        } else if constexpr (variables == 2) {
            product = kroneckerProduct(
                eval(bases[0], 0, x[0]),
                eval(bases[1], 1, x[1]));
            return tail(product);

        } else {
            assert(!bases.empty());

            product = kroneckerProduct(
                eval(bases[0], 0, x[0]),
                eval(bases[1], 1, x[1]));

            for (int i = 2; i < variables; ++i) {
                product.swap(product_prev);
                product =
                    kroneckerProduct(product_prev, eval(bases[i], i, x[i]));
            }

            return tail(product);
        }

        assert(false);
    }

    DenseMatrix evalBasisJacobianOld(DenseVector& x) const {
        // Jacobian basis matrix
        DenseMatrix J;
        J.setZero(getNumBasisFunctions(), variables);

        // Calculate partial derivatives
        for (unsigned int i = 0; i < variables; i++) {
            // One column in basis jacobian
            DenseVector bi;
            bi.setOnes(1);
            for (unsigned int j = 0; j < variables; j++) {
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

        // Jacobian basis matrix
        SparseMatrix J(getNumBasisFunctions(), variables);
        //J.setZero(numBasisFunctions(), numInputs);

        // Calculate partial derivatives
        for (unsigned int i = 0; i < variables; ++i) {
            // One column in basis jacobian
            std::vector<SparseVector> values(variables);

            for (unsigned int j = 0; j < variables; ++j) {
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
        for (unsigned int i = 0; i < variables; i++)
            ret.push_back(getNumBasisFunctions(i));
        return ret;
    }

private:
    bases_type bases;
};

template <unsigned d, unsigned v>
SparseMatrix BSplineBasis<d, v>::evalBasisJacobian2(DenseVector& x) const {
    // Jacobian basis matrix
    SparseMatrix J(getNumBasisFunctions(), variables);

    // Evaluate B-spline basis functions before looping
    std::vector<SparseVector> funcValues(variables);
    std::vector<SparseVector> gradValues(variables);

    for (unsigned int i = 0; i < variables; ++i) {
        funcValues[i] = bases[i].eval(x(i));
        gradValues[i] = bases[i].evalFirstDerivative(x(i));
    }

    // Calculate partial derivatives
    for (unsigned int i = 0; i < variables; i++) {
        std::vector<SparseVector> values(variables);

        for (unsigned int j = 0; j < variables; j++) {
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

template <unsigned d, unsigned v>
SparseMatrix BSplineBasis<d, v>::evalBasisHessian(DenseVector& x) const {

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
    SparseMatrix H(getNumBasisFunctions() * variables, variables);
    //H.setZero(numBasisFunctions()*numInputs, numInputs);

    // Calculate partial derivatives
    // Utilizing that Hessian is symmetric
    // Filling out lower left triangular
    for (unsigned int i = 0; i < variables; i++) // row
    {
        for (unsigned int j = 0; j <= i; j++) // col
        {
            // One column in basis jacobian
            SparseMatrix Hi(1, 1);
            Hi.insert(0, 0) = 1;

            for (unsigned int k = 0; k < variables; k++) {
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

template <unsigned d, unsigned v>
SparseMatrix BSplineBasis<d, v>::insertKnots(
    double tau, unsigned int dim, unsigned int multiplicity) {

    SparseMatrix A(1, 1);
    //    A.resize(1,1);
    A.insert(0, 0) = 1;

    // Calculate multivariate knot insertion matrix
    for (unsigned int i = 0; i < variables; i++) {
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

template <unsigned d, unsigned v>
SparseMatrix BSplineBasis<d, v>::refineKnots() {

    SparseMatrix A(1, 1);
    A.insert(0, 0) = 1;

    for (unsigned int i = 0; i < variables; i++) {
        SparseMatrix temp = A;
        SparseMatrix Ai   = bases[i].refineKnots();

        //A = kroneckerProduct(temp, Ai);
        A = myKroneckerProduct(temp, Ai);
    }

    A.makeCompressed();

    return A;
}

template <unsigned d, unsigned v>
SparseMatrix BSplineBasis<d, v>::refineKnotsLocally(DenseVector x) {

    SparseMatrix A(1, 1);
    A.insert(0, 0) = 1;

    for (unsigned int i = 0; i < variables; i++) {
        SparseMatrix temp = A;
        SparseMatrix Ai   = bases[i].refineKnotsLocally(x(i));

        //A = kroneckerProduct(temp, Ai);
        A = myKroneckerProduct(temp, Ai);
    }

    A.makeCompressed();

    return A;
}

template <unsigned d, unsigned v>
SparseMatrix BSplineBasis<d, v>::decomposeToBezierForm() {

    SparseMatrix A(1, 1);
    A.insert(0, 0) = 1;

    for (unsigned int i = 0; i < variables; i++) {
        SparseMatrix temp = A;
        SparseMatrix Ai   = bases[i].decomposeToBezierForm();

        //A = kroneckerProduct(temp, Ai);
        A = myKroneckerProduct(temp, Ai);
    }

    A.makeCompressed();

    return A;
}

template <unsigned d, unsigned v>
SparseMatrix BSplineBasis<d, v>::reduceSupport(
    std::vector<double>& lb, std::vector<double>& ub) {

    if (lb.size() != ub.size() || lb.size() != variables)
        throw Exception(
            "BSplineBasis<d,v>::reduceSupport: Incompatible dimension of domain bounds.");

    SparseMatrix A(1, 1);
    A.insert(0, 0) = 1;

    for (unsigned int i = 0; i < variables; i++) {
        SparseMatrix temp = A;
        SparseMatrix Ai;

        Ai = bases[i].reduceSupport(lb[i], ub[i]);

        //A = kroneckerProduct(temp, Ai);
        A = myKroneckerProduct(temp, Ai);
    }

    A.makeCompressed();

    return A;
}

template <unsigned d, unsigned v>
unsigned int BSplineBasis<d, v>::getNumBasisFunctions(unsigned int dim) const {
    return bases[dim].getNumBasisFunctions();
}

template <unsigned d, unsigned v>
unsigned int BSplineBasis<d, v>::getNumBasisFunctions() const {

    unsigned int prod = 1;
    for (unsigned int dim = 0; dim < variables; dim++) {
        prod *= bases[dim].getNumBasisFunctions();
    }
    return prod;
}

template <unsigned d, unsigned v>
unsigned int
BSplineBasis<d, v>::getKnotMultiplicity(unsigned int dim, double tau) const {
    return bases[dim].knotMultiplicity(tau);
}

template <unsigned d, unsigned v>
double BSplineBasis<d, v>::getKnotValue(int dim, int index) const {
    return bases[dim].getKnotValue(index);
}

template <unsigned d, unsigned v>
unsigned int
BSplineBasis<d, v>::getLargestKnotInterval(unsigned int dim) const {
    return bases[dim].indexLongestInterval();
}

template <unsigned d, unsigned v>
std::vector<unsigned int>
BSplineBasis<d, v>::getNumBasisFunctionsTarget() const {

    std::vector<unsigned int> ret;

    for (unsigned int dim = 0; dim < variables; dim++) {
        ret.push_back(bases[dim].getNumBasisFunctionsTarget());
    }

    return ret;
}

template <unsigned d, unsigned v>
int BSplineBasis<d, v>::supportedPrInterval() const {

    int ret = 1;

    for (unsigned int dim = 0; dim < variables; dim++) {
        ret *= (bases[dim].getBasisDegree() + 1);
    }

    return ret;
}

template <unsigned d, unsigned v>
bool BSplineBasis<d, v>::insideSupport(DenseVector& x) const {

    for (unsigned int dim = 0; dim < variables; dim++) {
        if (!bases[dim].insideSupport(x(dim))) {
            return false;
        }
    }
    return true;
}

template <unsigned d, unsigned v>
std::vector<double> BSplineBasis<d, v>::getSupportLowerBound() const {

    std::vector<double> lb;
    for (unsigned int dim = 0; dim < variables; dim++) {
        lb.push_back(bases[dim].knot_front());
    }

    return lb;
}

template <unsigned d, unsigned v>
std::vector<double> BSplineBasis<d, v>::getSupportUpperBound() const {

    std::vector<double> ub;
    for (unsigned int dim = 0; dim < variables; dim++) {
        ub.push_back(bases[dim].knot_back());
    }
    return ub;
}

} // namespace SPLINTER

#endif // SPLINTER_BSPLINEBASIS_H
