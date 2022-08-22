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

#include "function.h"
#include "bsplinebasis.h"

namespace SPLINTER
{

/**
 * Class that implements the multivariate tensor product B-spline
 */
class SPLINTER_API BSpline 
{
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
    BSpline(std::vector< std::vector<double> > knotVectors, std::vector<unsigned int> basisDegrees);
    BSpline(DenseVector coefficients, std::vector< std::vector<double> > knotVectors, std::vector<unsigned int> basisDegrees);

    template <class x_type>
    double eval(x_type const& x) const
    {
        checkInput(x);
        // NOTE: casting to DenseVector to allow accessing as res(0)
        DenseVector res = coefficients.transpose() * evalBasis(x);
        return res(0);
    }

    DenseMatrix evalJacobian(DenseVector x) const;
    DenseMatrix evalHessian(DenseVector x) const;

    template <class x_type>
    void checkInput(x_type const& x) const {
        assert(x.size() == numVariables);
    }

    template <class x_type>
    double operator()(x_type const& x) const { return eval(x); }

    double operator()(double const x0) const { 
        return eval(std::array{x0});
    }


    /**
     * Getters
     */
    DenseVector getCoefficients()
    {
        return coefficients;
    }

    unsigned int getNumCoefficients() const
    {
        return coefficients.size();
    }

    unsigned int getNumControlPoints() const
    {
        return coefficients.size();
    }

    std::vector<unsigned int> getNumBasisFunctionsPerVariable() const;

    unsigned int getNumBasisFunctions() const
    {
        return basis.getNumBasisFunctions();
    }

    DenseMatrix getControlPoints() const;
    std::vector< std::vector<double>> getKnotVectors() const;
    std::vector<unsigned int> getBasisDegrees() const;
    std::vector<double> getDomainUpperBound() const;
    std::vector<double> getDomainLowerBound() const;

    /**
     * Setters
     */
    void setCoefficients(const DenseVector &coefficients);
    void setControlPoints(const DenseMatrix &controlPoints);
    void checkControlPoints() const;

    // Linear transformation of control points (B-spline has affine invariance)
    void updateControlPoints(const DenseMatrix &A);

    // Reduce support of B-spline
    void reduceSupport(std::vector<double> lb, std::vector<double> ub, bool doRegularizeKnotVectors = true);

    // Perform global knot refinement
    void globalKnotRefinement(); // All knots in one shabang

    // Perform a local knot refinement at x
    void localKnotRefinement(DenseVector x);

    // Decompose B-spline to Bezier form
    void decomposeToBezierForm();

    // Insert a knot until desired knot multiplicity is obtained
    void insertKnots(double tau, unsigned int dim, unsigned int multiplicity = 1);

    std::string getDescription() const;

    unsigned int getNumVariables() const
    {
        return basis.getNumVariables();
    }

    auto centralDifference(DenseVector const& x) const
    {
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

    // Evaluation of B-spline basis functions
    template <class x_type>
    auto evalBasis(x_type const& x) const
    {
        return basis.eval(x);
    }

    SparseMatrix evalBasisJacobian(DenseVector x) const;

    // Domain reduction
    void regularizeKnotVectors(std::vector<double> &lb, std::vector<double> &ub);
    bool removeUnsupportedBasisFunctions(std::vector<double> &lb, std::vector<double> &ub);

    // Helper functions
    bool pointInDomain(DenseVector x) const;

    friend bool operator==(const BSpline &lhs, const BSpline &rhs);
};

} // namespace SPLINTER

#endif // SPLINTER_BSPLINE_H
