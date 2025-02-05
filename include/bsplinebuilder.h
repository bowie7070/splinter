/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_BSPLINEBUILDER_H
#define SPLINTER_BSPLINEBUILDER_H

#include "bspline.h"
#include <linearsolvers.h>

namespace SPLINTER {

// B-spline smoothing
enum class Smoothing {
    NONE,     // No smoothing
    IDENTITY, // Regularization term alpha*c'*I*c is added to OLS objective
    PSPLINE   // Smoothing term alpha*Delta(c,2) is added to OLS objective
};

// B-spline knot spacing
/*
 * To be added:
 * AS_SAMPLED_NOT_CLAMPED   // Place knots close to sample points. Without clamps.
 * EQUIDISTANT_NOT_CLAMPED  // Equidistant knots without clamps.
 */
enum class KnotSpacing {
    AS_SAMPLED, // Mimic spacing of sample points (moving average). With clamps (p+1 multiplicity of end knots).
    EQUIDISTANT, // Equidistant knots. With clamps (p+1 multiplicity of end knots).
    EXPERIMENTAL // Experimental knot spacing (for testing purposes).
};

std::vector<double>
knotVectorMovingAverage(std::vector<double> const& values, unsigned int degree);
std::vector<double> knotVectorEquidistant(
    std::vector<double> const& values,
    unsigned int degree,
    unsigned int numBasisFunctions);
std::vector<double> knotVectorBuckets(
    std::vector<double> const& values,
    unsigned int degree,
    unsigned int maxSegments = 10);

// Auxiliary
std::vector<double> extractUniqueSorted(std::vector<double> const& values);

// B-spline builder class
template <class data_table>
class SPLINTER_API Builder {
public:
    static constexpr unsigned variables = data_table::variables;

    Builder(data_table const& data) :
        _data(data),
        _numBasisFunctions(std::vector<unsigned int>(variables, 0)),
        _knotSpacing(KnotSpacing::AS_SAMPLED),
        _smoothing(Smoothing::NONE),
        _alpha(0.1) {}

    Builder& alpha(double alpha) {
        if (alpha < 0)
            throw Exception(
                "BSpline::Builder::alpha: alpha must be non-negative.");

        _alpha = alpha;
        return *this;
    }

    // Set build options

    Builder& numBasisFunctions(unsigned int numBasisFunctions) {
        _numBasisFunctions =
            std::vector<unsigned int>(variables, numBasisFunctions);
        return *this;
    }

    Builder& numBasisFunctions(std::vector<unsigned int> numBasisFunctions) {
        if (numBasisFunctions.size() != variables)
            throw Exception(
                "BSpline::Builder: Inconsistent length on numBasisFunctions vector.");
        _numBasisFunctions = numBasisFunctions;
        return *this;
    }

    Builder& knotSpacing(KnotSpacing knotSpacing) {
        _knotSpacing = knotSpacing;
        return *this;
    }

    Builder& smoothing(Smoothing smoothing) {
        _smoothing = smoothing;
        return *this;
    }

    // Build B-spline
    template <unsigned degree = 3>
    BSpline<degree, variables> build() const {
        BSplineBasis<degree, variables> basis{computeKnotVectors(degree)};
        auto coefficients = computeCoefficients(basis);

        return {std::move(coefficients), std::move(basis)};
    }

private:
    std::vector<unsigned int>
    getBSplineDegrees(unsigned int numVariables, unsigned int degree) {
        if (degree > 5)
            throw Exception(
                "BSpline::Builder: Only degrees in range [0, 5] are supported.");
        return std::vector<unsigned int>(numVariables, degree);
    }

    using matrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    template <class B>
    auto computeBasisFunctionMatrix(B const& basis) const {
        unsigned int numVariables = variables;
        unsigned int numSamples   = _data.getNumSamples();

        // TODO: Reserve nnz per row (degree+1)
        //int nnzPrCol = bspline.basis.supportedPrInterval();

        std::vector<Eigen::Triplet<double>> coefficients;

        basis1d_eval_uncached cache;

        int i = 0;
        for (auto const& sample : _data.csamples()) {
            using iterator = SparseVector::InnerIterator;
            basis.eval(
                sample.x,
                cache,
                [&coefficients, i](SparseVector const& row) {
                    for (iterator it(row); it; ++it) {
                        coefficients.emplace_back(i, it.index(), it.value());
                    }
                });

            ++i;
        }

        matrix A(numSamples, basis.getNumBasisFunctions());
        A.setFromTriplets(coefficients.begin(), coefficients.end());

        return A;
    }

    /*
    * Function for generating second order finite-difference matrix, which is used for penalizing the
    * (approximate) second derivative in control point calculation for P-splines.
    */
    template <class B>
    auto getSecondOrderFiniteDifferenceMatrix(B const& basis) const {
        static constexpr unsigned numVariables = B::variables;

        // Number of (total) basis functions - defines the number of columns in D
        unsigned int numCols = basis.getNumBasisFunctions();
        std::vector<unsigned int> numBasisFunctions =
            basis.getNumBasisFunctionsPerVariable();

        // Number of basis functions (and coefficients) in each variable
        std::vector<unsigned int> dims;
        for (unsigned int i = 0; i < numVariables; i++)
            dims.push_back(numBasisFunctions[i]);

        std::reverse(dims.begin(), dims.end());

        for (unsigned int i = 0; i < numVariables; ++i)
            if (numBasisFunctions[i] < 3)
                throw Exception(
                    "BSpline::Builder::getSecondOrderDifferenceMatrix: Need at least three coefficients/basis function per variable.");

        // Number of rows in D and in each block
        int numRows = 0;
        std::vector<int> numBlkRows;
        for (unsigned int i = 0; i < numVariables; i++) {
            int prod = 1;
            for (unsigned int j = 0; j < numVariables; j++) {
                if (i == j)
                    prod *= (dims[j] - 2);
                else
                    prod *= dims[j];
            }
            numRows += prod;
            numBlkRows.push_back(prod);
        }

        std::vector<Eigen::Triplet<double>> coefficients;
        coefficients.reserve(numCols * 2 * numVariables);

        int i = 0; // Row index

        // Loop though each dimension (each dimension has its own block)
        for (unsigned int d = 0; d < numVariables; d++) {
            // Calculate left and right products
            int leftProd      = 1;
            auto const insert = [&](int k) {
                coefficients.emplace_back(i, k, 1);
                k += leftProd;
                coefficients.emplace_back(i, k, -2);
                k += leftProd;
                coefficients.emplace_back(i, k, 1);
            };

            int rightProd = 1;
            for (unsigned int k = 0; k < d; k++) {
                leftProd *= dims[k];
            }
            for (unsigned int k = d + 1; k < numVariables; k++) {
                rightProd *= dims[k];
            }

            // Loop through subblocks on the block diagonal
            for (int j = 0; j < rightProd; j++) {
                // Start column of current subblock
                int blkBaseCol = j * leftProd * dims[d];
                // Block rows [I -2I I] of subblock
                for (unsigned int l = 0; l < (dims[d] - 2); l++) {
                    // Special case for first dimension
                    if (d == 0) {
                        int k = j * leftProd * dims[d] + l;
                        insert(k);
                        i++;
                    } else {
                        // Loop for identity matrix
                        for (int n = 0; n < leftProd; n++) {
                            int k = blkBaseCol + l * leftProd + n;
                            insert(k);
                            i++;
                        }
                    }
                }
            }
        }

        // Resize and initialize D
        matrix D(numRows, numCols);
        D.setFromTriplets(coefficients.begin(), coefficients.end());

        return D;
    }

    /*
    * Find coefficients of B-spline by solving:
    * min ||A*x - b||^2 + alpha*||R||^2,
    * where
    * A = mxn matrix of n basis functions evaluated at m sample points,
    * b = vector of m sample points y-values (or x-values when calculating knot averages),
    * x = B-spline coefficients (or knot averages),
    * R = Regularization matrix,
    * alpha = regularization parameter.
    */
    template <class basis_type>
    DenseVector computeCoefficients(basis_type const& basis) const {
        auto const B = computeBasisFunctionMatrix(basis);
        matrix A;
        DenseVector b = getSamplePointValues();

        if (_smoothing == Smoothing::IDENTITY) {
            /*
            * Computing B-spline coefficients with a regularization term
            * ||Ax-b||^2 + alpha*x^T*x
            *
            * NOTE: This corresponds to a Tikhonov regularization (or ridge regression) with the Identity matrix.
            * See: https://en.wikipedia.org/wiki/Tikhonov_regularization
            *
            * NOTE2: consider changing regularization factor to (alpha/numSample)
            */
            A = B.transpose() * B;
            b = B.transpose() * b;

            auto I = matrix(A.cols(), A.cols());
            I.setIdentity();
            A += _alpha * I;
        } else if (_smoothing == Smoothing::PSPLINE) {
            /*
            * The P-Spline is a smooting B-spline which relaxes the interpolation constraints on the control points to allow
            * smoother spline curves. It minimizes an objective which penalizes both deviation from sample points (to lower bias)
            * and the magnitude of second derivatives (to lower variance).
            *
            * Setup and solve equations Ax = b,
            * A = B'*W*B + l*D'*D
            * b = B'*W*y
            * x = control coefficients or knot averages.
            * B = basis functions at sample x-values,
            * W = weighting matrix for interpolating specific points
            * D = second-order finite difference matrix
            * l = penalizing parameter (increase for more smoothing)
            * y = sample y-values when calculating control coefficients,
            * y = sample x-values when calculating knot averages
            */

            // Assuming regular grid
            unsigned int numSamples = _data.getNumSamples();

            // Second order finite difference matrix
            auto const D = getSecondOrderFiniteDifferenceMatrix(basis);

            // Left-hand side matrix
            A = B.transpose() * B + _alpha * D.transpose() * D;

            // Compute right-hand side matrices
            b = B.transpose() * b;
        } else {
            A = B;
        }

        DenseVector x;

        int numEquations    = A.rows();
        int maxNumEquations = 100;
        bool solveAsDense   = (numEquations < maxNumEquations);

        if (!solveAsDense) {
            SparseLU<> s;
            //bool successfulSolve = (s.solve(A,Bx,Cx) && s.solve(A,By,Cy));

            solveAsDense = !s.solve(A, b, x);
        }

        if (solveAsDense) {
            DenseMatrix Ad = A.toDense();
            DenseQR<DenseVector> s;
            // DenseSVD<DenseVector> s;
            //bool successfulSolve = (s.solve(Ad,Bx,Cx) && s.solve(Ad,By,Cy));
            if (!s.solve(Ad, b, x)) {
                throw Exception(
                    "BSpline::Builder::computeBSplineCoefficients: Failed to solve for B-spline coefficients.");
            }
        }

        return x;
    }

    // Control point computations
    template <class S>
    DenseVector computeBSplineCoefficients(S const& bspline) const;
    DenseVector getSamplePointValues() const {
        DenseVector B(_data.getNumSamples());

        int i = 0;
        for (auto it = _data.cbegin(); it != _data.cend(); ++it, ++i)
            B(i) = it->y;

        return B;
    }

    // Computing knots
    std::vector<std::vector<double>> computeKnotVectors(unsigned degree) const {
        std::vector<std::vector<double>> grid = _data._getTableX();

        std::vector<std::vector<double>> knotVectors;

        for (unsigned int i = 0; i < variables; ++i) {
            // Compute knot vector
            knotVectors.push_back(
                computeKnotVector(grid[i], degree, _numBasisFunctions[i]));
        }

        return knotVectors;
    }

    std::vector<double> computeKnotVector(
        std::vector<double> const& values,
        unsigned int degree,
        unsigned int numBasisFunctions) const {
        switch (_knotSpacing) {
        case KnotSpacing::AS_SAMPLED:
            return knotVectorMovingAverage(values, degree);
        case KnotSpacing::EQUIDISTANT:
            return knotVectorEquidistant(values, degree, numBasisFunctions);
        case KnotSpacing::EXPERIMENTAL:
            return knotVectorBuckets(values, degree);
        default:
            return knotVectorMovingAverage(values, degree);
        }
    }

    // Member variables
    data_table _data;
    std::vector<unsigned int> _numBasisFunctions;
    KnotSpacing _knotSpacing;
    Smoothing _smoothing;
    double _alpha;
};

} // namespace SPLINTER

#endif // SPLINTER_BSPLINEBUILDER_H
