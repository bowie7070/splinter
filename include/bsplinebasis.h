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

struct basis1d_eval_uncached {
    SparseVector operator()(BSplineBasis1D const& basis, int, double x) const {
        return basis.eval(x);
    }
};

template <class cache_type = std::map<std::tuple<int, double>, SparseVector>>
struct basis1d_eval_cached {
    mutable cache_type cache;

    SparseVector const&
    operator()(BSplineBasis1D const& basis, int i, double x) const {
        auto pos = cache.find({i, x});
        if (pos == cache.end()) {
            std::tie(pos, std::ignore) = cache.insert({{i, x}, basis.eval(x)});
        }
        return pos->second;
    }
};

class BSplineBasis {
public:
    BSplineBasis(
        std::vector<std::vector<double>> const& knotVectors,
        unsigned int basisDegrees) {
        unsigned int numVariables = knotVectors.size();

        // Set univariate bases
        bases.clear();
        for (unsigned int i = 0; i < numVariables; i++) {
            bases.push_back(BSplineBasis1D(knotVectors[i], basisDegrees));

            // Adjust target number of basis functions used in e.g. refinement
            if (numVariables > 2) {
                // One extra knot is allowed
                bases[i].setNumBasisFunctionsTarget(
                    (basisDegrees + 1) + 1); // Minimum degree+1
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

            SparseVector product = eval(bases[0], 0, x[0]);

            for (int i = 1, I = getNumVariables(); i < I; ++i) {
                product =
                    kroneckerProduct(product, eval(bases[i], i, x[i])).eval();
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

    BSplineBasis1D getSingleBasis(int dim);
    std::vector<double> const& getKnotVector(int dim) const {
        return bases[dim].getKnotVector();
    }

    unsigned int getBasisDegree() const { return bases[0].getBasisDegree(); }

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
