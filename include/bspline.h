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
#include "mykroneckerproduct.h"
#include "unsupported/Eigen/KroneckerProduct"
#include <iostream>
#include <linearsolvers.h>
#include <utilities.h>

namespace SPLINTER {

/**
 * Class that implements the multivariate tensor product B-spline
 */
template <unsigned _degree>
class SPLINTER_API BSpline {
public:
    static constexpr unsigned degree = _degree;
    /**
     * Construct B-spline from knot vectors, coefficients, and basis degrees
     */
    BSpline(DenseVector coefficients, BSplineBasis<degree> _basis) :
        basis(std::move(_basis)),
        coefficients(std::move(coefficients)) {}

    template <class x_type>
    static double as_scalar(x_type const& x) {
        assert(x.rows() == 1);
        assert(x.cols() == 1);
        return x(0, 0);
    }

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
    BSplineBasis<degree> basis;

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

// Computes knot averages.
template <unsigned d>
DenseMatrix computeKnotAverages(BSplineBasis<d> const& basis) {
    // Calculate knot averages for each knot vector
    std::vector<DenseVector> mu_vectors;
    for (unsigned int i = 0; i < basis.getNumVariables(); i++) {
        auto const& knots = basis.getKnotVector(i);
        DenseVector mu    = DenseVector::Zero(basis.getNumBasisFunctions(i));

        for (unsigned int j = 0; j < basis.getNumBasisFunctions(i); j++) {
            double knotAvg = 0;
            for (unsigned int k = j + 1; k <= j + basis.getBasisDegree(); k++) {
                knotAvg += knots[k];
            }
            mu(j) = knotAvg / basis.getBasisDegree();
        }
        mu_vectors.push_back(mu);
    }

    // Calculate vectors of ones (with same length as corresponding knot average vector)
    std::vector<DenseVector> knotOnes;
    for (unsigned int i = 0; i < basis.getNumVariables(); i++)
        knotOnes.push_back(DenseVector::Ones(mu_vectors[i].rows()));

    // Fill knot average matrix one column at the time
    DenseMatrix knot_averages = DenseMatrix::Zero(
        basis.getNumBasisFunctions(),
        basis.getNumVariables());

    for (unsigned int i = 0; i < basis.getNumVariables(); i++) {
        DenseMatrix mu_ext(1, 1);
        mu_ext(0, 0) = 1;
        for (unsigned int j = 0; j < basis.getNumVariables(); j++) {
            DenseMatrix temp = mu_ext;
            if (i == j)
                mu_ext = Eigen::kroneckerProduct(temp, mu_vectors[j]);
            else
                mu_ext = Eigen::kroneckerProduct(temp, knotOnes[j]);
        }
        if (mu_ext.rows() != basis.getNumBasisFunctions())
            throw Exception(
                "BSpline<d>::computeKnotAverages: Incompatible size of knot average matrix.");
        knot_averages.block(0, i, basis.getNumBasisFunctions(), 1) = mu_ext;
    }

    return knot_averages;
}

/**
 * Returns the (1 x getNumVariables()) Jacobian evaluated at x
 */
template <unsigned d>
DenseMatrix BSpline<d>::evalJacobian(DenseVector x) const {
    checkInput(x);
    return coefficients.transpose() * evalBasisJacobian(x);
}

/*
 * Returns the Hessian evaluated at x.
 * The Hessian is an n x n matrix,
 * where n is the dimension of x.
 */
template <unsigned d>
DenseMatrix BSpline<d>::evalHessian(DenseVector x) const {
    checkInput(x);

#ifndef NDEBUG
    if (!pointInDomain(x))
        throw Exception(
            "BSpline<d>::evalHessian: Evaluation at point outside domain.");
#endif // NDEBUG

    DenseMatrix H;
    H.setZero(1, 1);
    DenseMatrix identity =
        DenseMatrix::Identity(getNumVariables(), getNumVariables());
    DenseMatrix caug = kroneckerProduct(identity, coefficients.transpose());
    DenseMatrix DB   = basis.evalBasisHessian(x);
    H                = caug * DB;

    // Fill in upper triangular of Hessian
    for (size_t i = 0; i < getNumVariables(); ++i)
        for (size_t j = i + 1; j < getNumVariables(); ++j)
            H(i, j) = H(j, i);

    return H;
}

template <unsigned d>
SparseMatrix BSpline<d>::evalBasisJacobian(DenseVector x) const {
#ifndef NDEBUG
    if (!pointInDomain(x))
        throw Exception(
            "BSpline<d>::evalBasisJacobian: Evaluation at point outside domain.");
#endif // NDEBUG

    //SparseMatrix Bi = basis.evalBasisJacobian(x);       // Sparse Jacobian implementation
    //SparseMatrix Bi = basis.evalBasisJacobian2(x);  // Sparse Jacobian implementation
    DenseMatrix Bi =
        basis.evalBasisJacobianOld(x); // Old Jacobian implementation

    return Bi.sparseView();
}

template <unsigned d>
std::vector<double> BSpline<d>::getDomainUpperBound() const {
    return basis.getSupportUpperBound();
}

template <unsigned d>
std::vector<double> BSpline<d>::getDomainLowerBound() const {
    return basis.getSupportLowerBound();
}

template <unsigned d>
void BSpline<d>::updateControlPoints(DenseMatrix const& A) {
    assert(A.cols() == coefficients.rows());
    coefficients = A * coefficients;
}

template <unsigned d>
bool BSpline<d>::pointInDomain(DenseVector x) const {
    return basis.insideSupport(x);
}

template <unsigned d>
void BSpline<d>::reduceSupport(
    std::vector<double> lb,
    std::vector<double> ub,
    bool doRegularizeKnotVectors) {
    if (lb.size() != getNumVariables() || ub.size() != getNumVariables())
        throw Exception(
            "BSpline<d>::reduceSupport: Inconsistent vector sizes!");

    std::vector<double> sl = basis.getSupportLowerBound();
    std::vector<double> su = basis.getSupportUpperBound();

    for (unsigned int dim = 0; dim < getNumVariables(); dim++) {
        // Check if new domain is empty
        if (ub[dim] <= lb[dim] || lb[dim] >= su[dim] || ub[dim] <= sl[dim])
            throw Exception(
                "BSpline<d>::reduceSupport: Cannot reduce B-spline domain to empty set!");

        // Check if new domain is a strict subset
        if (su[dim] < ub[dim] || sl[dim] > lb[dim])
            throw Exception(
                "BSpline<d>::reduceSupport: Cannot expand B-spline domain!");

        // Tightest possible
        sl[dim] = lb[dim];
        su[dim] = ub[dim];
    }

    if (doRegularizeKnotVectors) {
        regularizeKnotVectors(sl, su);
    }

    // Remove knots and control points that are unsupported with the new bounds
    if (!removeUnsupportedBasisFunctions(sl, su)) {
        throw Exception(
            "BSpline<d>::reduceSupport: Failed to remove unsupported basis functions!");
    }
}

template <unsigned d>
void BSpline<d>::globalKnotRefinement() {
    // Compute knot insertion matrix
    SparseMatrix A = basis.refineKnots();

    // Update control points
    updateControlPoints(A);
}

template <unsigned d>
void BSpline<d>::localKnotRefinement(DenseVector x) {
    // Compute knot insertion matrix
    SparseMatrix A = basis.refineKnotsLocally(x);

    // Update control points
    updateControlPoints(A);
}

template <unsigned d>
void BSpline<d>::decomposeToBezierForm() {
    // Compute knot insertion matrix
    SparseMatrix A = basis.decomposeToBezierForm();

    // Update control points
    updateControlPoints(A);
}

template <unsigned d>
void BSpline<d>::insertKnots(
    double tau, unsigned int dim, unsigned int multiplicity) {
    // Insert knots and compute knot insertion matrix
    SparseMatrix A = basis.insertKnots(tau, dim, multiplicity);

    // Update control points
    updateControlPoints(A);
}

template <unsigned d>
void BSpline<d>::regularizeKnotVectors(
    std::vector<double>& lb, std::vector<double>& ub) {
    // Add and remove controlpoints and knots to make the b-spline p-regular with support [lb, ub]
    if (!(lb.size() == getNumVariables() && ub.size() == getNumVariables()))
        throw Exception(
            "BSpline<d>::regularizeKnotVectors: Inconsistent vector sizes.");

    unsigned int multiplicityTarget = degree + 1;
    for (unsigned int dim = 0; dim < getNumVariables(); dim++) {

        // Inserting many knots at the time (to save number of B-spline coefficient calculations)
        // NOTE: This method generates knot insertion matrices with more nonzero elements than
        // the method that inserts one knot at the time. This causes the preallocation of
        // kronecker product matrices to become too small and the speed deteriorates drastically
        // in higher dimensions because reallocation is necessary. This can be prevented by
        // precomputing the number of nonzeros when preallocating memory (see myKroneckerProduct).
        int numKnotsLB =
            multiplicityTarget - basis.getKnotMultiplicity(dim, lb[dim]);
        if (numKnotsLB > 0) {
            insertKnots(lb[dim], dim, numKnotsLB);
        }

        int numKnotsUB =
            multiplicityTarget - basis.getKnotMultiplicity(dim, ub[dim]);
        if (numKnotsUB > 0) {
            insertKnots(ub[dim], dim, numKnotsUB);
        }
    }
}

template <unsigned d>
bool BSpline<d>::removeUnsupportedBasisFunctions(
    std::vector<double>& lb, std::vector<double>& ub) {
    if (lb.size() != getNumVariables() || ub.size() != getNumVariables())
        throw Exception(
            "BSpline<d>::removeUnsupportedBasisFunctions: Incompatible dimension of domain bounds.");

    SparseMatrix A = basis.reduceSupport(lb, ub);

    if (coefficients.size() != A.rows())
        return false;

    // Remove unsupported control points (basis functions)
    updateControlPoints(A.transpose());

    return true;
}

template <unsigned d>
std::string BSpline<d>::getDescription() const {
    return "BSpline of degree " + std::to_string(d);
}

} // namespace SPLINTER

#endif // SPLINTER_BSPLINE_H
