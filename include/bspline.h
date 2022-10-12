/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_BSPLINE_H
#define SPLINTER_BSPLINE_H

#include "bsplinebasis.h"
#include "function.h"

namespace SPLINTER {

/**
 * Class that implements the multivariate tensor product B-spline
 */
class SPLINTER_API BSpline {
public:
    /**
     * Builder class for construction by regression
     * Implemented in BSplineBuilder.*
     */
    template <class data_table>
    class Builder;
    enum class Smoothing;
    enum class KnotSpacing;

    /**
     * Construct B-spline from knot vectors, coefficients, and basis degrees
     */
    BSpline(DenseVector coefficients, BSplineBasis);

    static double as_scalar(Eigen::Vector<double, 1> x) { return x[0]; }

    template <class x_type, class eval_fn>
    double eval(x_type const& x, eval_fn& cache) const {
        checkInput(x);

        return basis.eval(x, cache, [this](auto const& y) {
            return as_scalar(coefficients.transpose() * y);
        });
    }

    template <class x_type>
    double eval(x_type const& x) const {
        basis1d_eval_uncached cache;
        return eval(x, cache);
    }

    DenseMatrix evalJacobian(DenseVector x) const;
    DenseMatrix evalHessian(DenseVector x) const;

    template <class x_type>
    void checkInput(x_type const& x) const {
        assert(x.size() == getNumVariables());
    }

    template <class x_type>
    double operator()(x_type const& x) const {
        return eval(x);
    }

    template <class x_type, class eval_fn>
    double operator()(x_type const& x, eval_fn& cache) const {
        return eval(x, cache);
    }

    double operator()(double const x0) const { return eval(std::array{x0}); }

    /**
     * Getters
     */
    DenseVector getCoefficients() { return coefficients; }

    unsigned int getNumCoefficients() const { return coefficients.size(); }

    unsigned int getNumControlPoints() const { return coefficients.size(); }

    unsigned int getNumBasisFunctions() const {
        return basis.getNumBasisFunctions();
    }

    std::vector<double> getDomainUpperBound() const;
    std::vector<double> getDomainLowerBound() const;

    /**
     * Setters
     */

    // Linear transformation of control points (B-spline has affine invariance)
    void updateControlPoints(DenseMatrix const& A);

    // Reduce support of B-spline
    void reduceSupport(
        std::vector<double> lb,
        std::vector<double> ub,
        bool doRegularizeKnotVectors = true);

    // Perform global knot refinement
    void globalKnotRefinement(); // All knots in one shabang

    // Perform a local knot refinement at x
    void localKnotRefinement(DenseVector x);

    // Decompose B-spline to Bezier form
    void decomposeToBezierForm();

    // Insert a knot until desired knot multiplicity is obtained
    void
    insertKnots(double tau, unsigned int dim, unsigned int multiplicity = 1);

    std::string getDescription() const;

    unsigned int getNumVariables() const { return basis.getNumVariables(); }

    auto centralDifference(DenseVector const& x) const {
        return SPLINTER::centralDifference(*this, x);
    }

private:
    BSplineBasis basis;

    /*
     * The control point matrix is P = (knotaverages, coefficients) in R^(m x n),
     * where m = numBasisFunctions and n = numVariables + 1. Each row in P is a control point.
     */
    DenseVector coefficients;
    DenseMatrix knotaverages;

    SparseMatrix evalBasisJacobian(DenseVector x) const;

    // Domain reduction
    void
    regularizeKnotVectors(std::vector<double>& lb, std::vector<double>& ub);
    bool removeUnsupportedBasisFunctions(
        std::vector<double>& lb, std::vector<double>& ub);

    // Helper functions
    bool pointInDomain(DenseVector x) const;

    friend bool operator==(BSpline const& lhs, BSpline const& rhs);
};

} // namespace SPLINTER

#endif // SPLINTER_BSPLINE_H
