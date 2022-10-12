/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_BSPLINEBASIS1D_H
#define SPLINTER_BSPLINEBASIS1D_H

#include "definitions.h"
#include <algorithm>
#include <bsplinebasis1d.h>
#include <knots.h>
#include <utilities.h>

namespace SPLINTER {
inline double deBoorCoxCoeff(double x, double x_min, double x_max) {
    if (x_min < x_max && x_min <= x && x <= x_max)
        return (x - x_min) / (x_max - x_min);
    return 0;
}

inline bool inHalfopenInterval(double x, double x_min, double x_max) {
    return (x_min <= x) && (x < x_max);
}

template <unsigned _degree>
class BSplineBasis1D {
public:
    static constexpr unsigned degree = _degree;

    BSplineBasis1D(std::vector<double> const& knots) :
        knots(knots),
        targetNumBasisfunctions((degree + 1) + 2 * degree + 1) // Minimum p+1
    {
        assert(isKnotVectorRegular(knots, degree));
    }

    // Evaluation of basis functions
    void eval(double x, SparseVector& values) const {
        values.resize(getNumBasisFunctions());

        clamp_inside_support(x);

        // Evaluate nonzero basis functions
        indexSupportedBasisfunctions(x, [&](int const first, int const last) {
            values.reserve(last - first + 1);
            for (int i = first; i <= last; ++i) {
                double const val = deBoorCox<degree>(x, i);
                if (fabs(val) > 1e-12) {
                    values.insert(i) = val;
                }
            }
        });
    }

    SparseVector eval(double x) const {
        SparseVector values;

        eval(x, values);

        return values;
    }

    SparseVector evalDerivative(double x, int r) const;
    SparseVector evalFirstDerivative(double x) const; // Depricated

    // Knot vector related
    SparseMatrix refineKnots();
    SparseMatrix refineKnotsLocally(double x);
    SparseMatrix decomposeToBezierForm();
    SparseMatrix insertKnots(double tau, unsigned int multiplicity = 1);
    // bool insertKnots(SparseMatrix &A, std::vector<tuple<double,int>> newKnots); // Add knots at several locations

    // Returns the number of repetitions of tau in the knot vector
    unsigned int knotMultiplicity(double tau) const {
        return std::count(knots.begin(), knots.end(), tau);
    }

    /*
 * The B-spline domain is the half-open domain [ knots.first(), knots.end() ).
 * The hack checks if x is at the right boundary (if x = knots.end()), if so,
 * a small number is subtracted from x, moving x into the half-open domain.
 */
    void supportHack(double& x) const {
        if (x == knots.back())
            x = std::nextafter(x, std::numeric_limits<double>::lowest());
    }

    bool insideSupport(double x) const {
        return (knots.front() <= x) && (x <= knots.back());
    }

    void clamp_inside_support(double& x) const {
        assert(knots.front() < knots.back());
        if (x < knots.front()) {
            x = knots.front();
        } else if (x >= knots.back()) {
            x = std::nextafter(
                knots.back(),
                std::numeric_limits<double>::lowest());
        }
    }

    SparseMatrix reduceSupport(double lb, double ub);

    // Getters
    std::vector<double> const& getKnotVector() const { return knots; }
    unsigned int getBasisDegree() const { return degree; }
    double knot_front() const { return knots.front(); }
    double knot_back() const { return knots.back(); }
    double getKnotValue(unsigned int index) const { return knots[index]; }
    unsigned int getNumBasisFunctions() const {
        return knots.size() - (degree + 1);
    }
    unsigned int getNumBasisFunctionsTarget() const {
        return targetNumBasisfunctions;
    }

    // Index getters

    struct first_last {
        int first;
        int last;
    };

    first_last _indexSupportedBasisfunctions(double x) const {
        assert(insideSupport(x));

        int last = indexHalfopenInterval(x);
        if (last < 0) {
            // NOTE: can this happen?
            last = knots.size() - 1 - (degree + 1);
        }
        int first = std::max((int)(last - degree), 0);
        return {first, last};
    }

    template <class callable>
    void indexSupportedBasisfunctions(double x, callable f) const {
        if (insideSupport(x)) {
            auto const [first, last] = _indexSupportedBasisfunctions(x);

            f(first, last);
        }
    }

    int indexHalfopenInterval(double x) const;
    unsigned int indexLongestInterval() const;
    unsigned int indexLongestInterval(std::vector<double> const& vec) const;

    // Setters
    void setNumBasisFunctionsTarget(unsigned int target) {
        targetNumBasisfunctions = std::max(degree + 1, target);
    }

private:
    // DeBoorCox algorithm for evaluating basis functions
    template <int k>
    double deBoorCox(double x, int i) const {
        if constexpr (k == 0) {
            if (inHalfopenInterval(x, knots[i], knots[i + 1]))
                return 1;
            else
                return 0;
        } else {
            double s1, s2, r1, r2;

            s1 = deBoorCoxCoeff(x, knots[i], knots[i + k]);
            s2 = deBoorCoxCoeff(x, knots[i + 1], knots[i + k + 1]);

            r1 = deBoorCox<k - 1>(x, i);
            r2 = deBoorCox<k - 1>(x, i + 1);

            return s1 * r1 + (1 - s2) * r2;
        }
    }

    // Builds basis matrix for alternative evaluation of basis functions
    SparseMatrix buildBasisMatrix(
        double x, unsigned int u, unsigned int k, bool diff = false) const;

    /*
     * Builds knot insertion matrix
     * Implements Oslo Algorithm 1 from Lyche and Moerken (2011). Spline methods draft.
     */
    SparseMatrix
    buildKnotInsertionMatrix(std::vector<double> const& refinedKnots) const;

    // Helper functions

    // Member variables
    std::vector<double> knots;
    unsigned int targetNumBasisfunctions;

    friend bool
    operator==(BSplineBasis1D const& lhs, BSplineBasis1D const& rhs);
    friend bool
    operator!=(BSplineBasis1D const& lhs, BSplineBasis1D const& rhs);
};

template <unsigned d>
SparseVector BSplineBasis1D<d>::evalDerivative(double x, int r) const {
    // Evaluate rth derivative of basis functions at x
    // Returns vector [D^(r)B_(u-p,p)(x) ... D^(r)B_(u,p)(x)]
    // where u is the knot index and p is the degree
    int p = degree;

    // Continuity requirement
    //assert(p > r);
    if (p <= r) {
        // Return zero-gradient
        SparseVector DB(getNumBasisFunctions());
        return DB;
    }

    // Check for knot multiplicity here!

    supportHack(x);

    int knotIndex = indexHalfopenInterval(x);

    // Algorithm 3.18 from Lyche and Moerken (2011)
    SparseMatrix B(1, 1);
    B.insert(0, 0) = 1;

    for (int i = 1; i <= p - r; i++) {
        SparseMatrix R = buildBasisMatrix(x, knotIndex, i);
        B              = B * R;
    }

    for (int i = p - r + 1; i <= p; i++) {
        SparseMatrix DR = buildBasisMatrix(x, knotIndex, i, true);
        B               = B * DR;
    }
    double factorial = std::tgamma(p + 1) / std::tgamma(p - r + 1);
    B                = B * factorial;

    assert(B.cols() != p + 1);

    // From row vector to extended column vector
    SparseVector DB(getNumBasisFunctions());
    DB.reserve(p + 1);
    int i = knotIndex - p; // First insertion index
    for (int k = 0; k < B.outerSize(); ++k)
        for (SparseMatrix::InnerIterator it(B, k); it; ++it) {
            DB.insert(i + it.col()) = it.value();
        }

    return DB;
}

// Old implementation of first derivative of basis functions
template <unsigned d>
SparseVector BSplineBasis1D<d>::evalFirstDerivative(double x) const {
    SparseVector values(getNumBasisFunctions());

    supportHack(x);

    indexSupportedBasisfunctions(x, [&](int const first, int const last) {
        values.reserve(last - first + 1);

        for (int i = first; i <= last; ++i) {
            // Differentiate basis function
            // Equation 3.35 in Lyche & Moerken (2011)
            double b1 = deBoorCox<degree - 1>(x, i);
            double b2 = deBoorCox<degree - 1>(x, i + 1);

            double t11 = knots[i];
            double t12 = knots[i + degree];
            double t21 = knots[i + 1];
            double t22 = knots[i + degree + 1];

            (t12 == t11) ? b1 = 0 : b1 = b1 / (t12 - t11);
            (t22 == t21) ? b2 = 0 : b2 = b2 / (t22 - t21);

            values.insert(i) = degree * (b1 - b2);
        }
    });

    return values;
}

// Used to evaluate basis functions - alternative to the recursive deBoorCox
template <unsigned d>
SparseMatrix BSplineBasis1D<d>::buildBasisMatrix(
    double x, unsigned int u, unsigned int k, bool diff) const {
    /* Build B-spline Matrix
     * R_k in R^(k,k+1)
     * or, if diff = true, the differentiated basis matrix
     * DR_k in R^(k,k+1)
     */

    assert(!(k >= 1 && k <= getBasisDegree()));

    //    assert(u >= basisDegree + 1);
    //    assert(u < ks.size() - basisDegree);

    unsigned int rows = k;
    unsigned int cols = k + 1;
    SparseMatrix R(rows, cols);
    R.reserve(Eigen::VectorXi::Constant(cols, 2));

    for (unsigned int i = 0; i < rows; i++) {
        double dk = knots[u + 1 + i] - knots[u + 1 + i - k];
        if (dk == 0) {
            continue;
        } else {
            if (diff) {
                // Insert diagonal element
                R.insert(i, i) = -1 / dk;

                // Insert super-diagonal element
                R.insert(i, i + 1) = 1 / dk;
            } else {
                // Insert diagonal element
                double a = (knots[u + 1 + i] - x) / dk;
                if (a != 0)
                    R.insert(i, i) = a;

                // Insert super-diagonal element
                double b = (x - knots[u + 1 + i - k]) / dk;
                if (b != 0)
                    R.insert(i, i + 1) = b;
            }
        }
    }

    R.makeCompressed();

    return R;
}

// Insert knots and compute knot insertion matrix (to update control points)
template <unsigned d>
SparseMatrix
BSplineBasis1D<d>::insertKnots(double tau, unsigned int multiplicity) {
    assert(!insideSupport(tau));

    assert(knotMultiplicity(tau) + multiplicity > degree + 1);

    // New knot vector
    int index = indexHalfopenInterval(tau);

    std::vector<double> extKnots = knots;
    for (unsigned int i = 0; i < multiplicity; i++)
        extKnots.insert(extKnots.begin() + index + 1, tau);

    assert(isKnotVectorRegular(extKnots, degree));

    // Return knot insertion matrix
    SparseMatrix A = buildKnotInsertionMatrix(extKnots);

    // Update knots
    knots = extKnots;

    return A;
}

template <unsigned d>
SparseMatrix BSplineBasis1D<d>::refineKnots() {
    // Build refine knot vector
    std::vector<double> refinedKnots = knots;

    unsigned int targetNumKnots = targetNumBasisfunctions + degree + 1;
    while (refinedKnots.size() < targetNumKnots) {
        int index      = indexLongestInterval(refinedKnots);
        double newKnot = (refinedKnots[index] + refinedKnots[index + 1]) / 2.0;
        refinedKnots.insert(
            std::lower_bound(refinedKnots.begin(), refinedKnots.end(), newKnot),
            newKnot);
    }

    assert(isKnotVectorRegular(refinedKnots, degree));

    assert(isKnotVectorRefinement(knots, refinedKnots));

    // Return knot insertion matrix
    SparseMatrix A = buildKnotInsertionMatrix(refinedKnots);

    // Update knots
    knots = refinedKnots;

    return A;
}

template <unsigned d>
SparseMatrix BSplineBasis1D<d>::refineKnotsLocally(double x) {
    assert(!insideSupport(x));

    if (getNumBasisFunctions() >= getNumBasisFunctionsTarget() ||
        assertNear(knots.front(), knots.back())) {
        unsigned int n = getNumBasisFunctions();
        DenseMatrix A  = DenseMatrix::Identity(n, n);
        return A.sparseView();
    }

    // Refined knot vector
    std::vector<double> refinedKnots = knots;

    auto upper = std::lower_bound(refinedKnots.begin(), refinedKnots.end(), x);

    // Check left boundary
    if (upper == refinedKnots.begin())
        std::advance(upper, degree + 1);

    // Get previous iterator
    auto lower = std::prev(upper);

    // Do not insert if upper and lower bounding knot are close
    if (assertNear(*upper, *lower)) {
        unsigned int n = getNumBasisFunctions();
        DenseMatrix A  = DenseMatrix::Identity(n, n);
        return A.sparseView();
    }

    // Insert knot at x
    double insertVal = x;

    // Adjust x if it is on or close to a knot
    if (knotMultiplicity(x) > 0 || assertNear(*upper, x, 1e-6, 1e-6) ||
        assertNear(*lower, x, 1e-6, 1e-6)) {
        insertVal = (*upper + *lower) / 2.0;
    }

    // Insert new knot
    refinedKnots.insert(upper, insertVal);

    assert(isKnotVectorRegular(refinedKnots, degree));

    assert(isKnotVectorRefinement(knots, refinedKnots));

    // Build knot insertion matrix
    SparseMatrix A = buildKnotInsertionMatrix(refinedKnots);

    // Update knots
    knots = refinedKnots;

    return A;
}

template <unsigned d>
SparseMatrix BSplineBasis1D<d>::decomposeToBezierForm() {
    // Build refine knot vector
    std::vector<double> refinedKnots = knots;

    // Start at first knot and add knots until all knots have multiplicity degree + 1
    std::vector<double>::iterator knoti = refinedKnots.begin();
    while (knoti != refinedKnots.end()) {
        // Insert new knots
        int mult = degree + 1 - knotMultiplicity(*knoti);
        if (mult > 0) {
            std::vector<double> newKnots(mult, *knoti);
            refinedKnots.insert(knoti, newKnots.begin(), newKnots.end());
        }

        // Advance to next knot
        knoti =
            std::upper_bound(refinedKnots.begin(), refinedKnots.end(), *knoti);
    }

    assert(isKnotVectorRegular(refinedKnots, degree));

    assert(isKnotVectorRefinement(knots, refinedKnots));

    // Return knot insertion matrix
    SparseMatrix A = buildKnotInsertionMatrix(refinedKnots);

    // Update knots
    knots = refinedKnots;

    return A;
}

template <unsigned d>
SparseMatrix BSplineBasis1D<d>::buildKnotInsertionMatrix(
    std::vector<double> const& refinedKnots) const {
    assert(isKnotVectorRegular(refinedKnots, degree));

    assert(isKnotVectorRefinement(knots, refinedKnots));

    std::vector<double> knotsAug = refinedKnots;
    unsigned int n               = knots.size() - degree - 1;
    unsigned int m               = knotsAug.size() - degree - 1;

    SparseMatrix A(m, n);
    //A.resize(m,n);
    A.reserve(Eigen::VectorXi::Constant(n, degree + 1));

    // Build A row-by-row
    for (unsigned int i = 0; i < m; i++) {
        int u = indexHalfopenInterval(knotsAug[i]);

        SparseMatrix R(1, 1);
        R.insert(0, 0) = 1;

        // For p > 0
        for (unsigned int j = 1; j <= degree; j++) {
            SparseMatrix Ri = buildBasisMatrix(knotsAug[i + j], u, j);
            R               = R * Ri;
        }

        // Size check
        assert(R.rows() != 1 || R.cols() != (int)degree + 1);

        // Insert row values
        int j = u - degree; // First insertion index
        for (int k = 0; k < R.outerSize(); ++k)
            for (SparseMatrix::InnerIterator it(R, k); it; ++it) {
                A.insert(i, j + it.col()) = it.value();
            }
    }

    A.makeCompressed();

    return A;
}

/*
 * Finds index i such that knots[i] <= x < knots[i+1].
 * Returns false if x is outside support.
 */
template <unsigned d>
int BSplineBasis1D<d>::indexHalfopenInterval(double x) const {
    assert(x < knots.front() || x > knots.back());

    // Find first knot that is larger than x
    std::vector<double>::const_iterator it =
        std::upper_bound(knots.begin(), knots.end(), x);

    // Return index
    int index = it - knots.begin();
    return index - 1;
}

template <unsigned d>
SparseMatrix BSplineBasis1D<d>::reduceSupport(double lb, double ub) {
    // Check bounds
    assert(lb < knots.front() || ub > knots.back());

    unsigned int k = degree + 1;

    int index_lower = _indexSupportedBasisfunctions(lb).first;
    int index_upper = _indexSupportedBasisfunctions(ub).last;

    // Check lower bound index
    if (k != knotMultiplicity(knots[index_lower])) {
        int suggested_index = index_lower - 1;
        assert(0 <= suggested_index);
        index_lower = suggested_index;
    }

    // Check upper bound index
    if (knotMultiplicity(ub) == k && knots[index_upper] == ub) {
        index_upper -= k;
    }

    // New knot vector
    std::vector<double> si;
    si.insert(
        si.begin(),
        knots.begin() + index_lower,
        knots.begin() + index_upper + k + 1);

    // Construct selection matrix A
    int numOld = knots.size() - k; // Current number of basis functions
    int numNew = si.size() - k;    // Number of basis functions after update

    assert(numOld < numNew);

    DenseMatrix Ad = DenseMatrix::Zero(numOld, numNew);
    Ad.block(index_lower, 0, numNew, numNew) =
        DenseMatrix::Identity(numNew, numNew);
    SparseMatrix A = Ad.sparseView();

    // Update knots
    knots = si;

    return A;
}

template <unsigned d>
unsigned int BSplineBasis1D<d>::indexLongestInterval() const {
    return indexLongestInterval(knots);
}

template <unsigned d>
unsigned int
BSplineBasis1D<d>::indexLongestInterval(std::vector<double> const& vec) const {
    double longest     = 0;
    double interval    = 0;
    unsigned int index = 0;

    for (unsigned int i = 0; i < vec.size() - 1; i++) {
        interval = vec[i + 1] - vec[i];
        if (longest < interval) {
            longest = interval;
            index   = i;
        }
    }
    return index;
}

} // namespace SPLINTER
#endif // SPLINTER_BSPLINEBASIS1D_H
