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

namespace SPLINTER {

class BSplineBasis1D {
public:
    BSplineBasis1D(std::vector<double> const& knots, unsigned int degree);

    // Evaluation of basis functions
    SparseVector eval(double x) const;
    SparseVector evalDerivative(double x, int r) const;
    SparseVector evalFirstDerivative(double x) const; // Depricated

    // Knot vector related
    SparseMatrix refineKnots();
    SparseMatrix refineKnotsLocally(double x);
    SparseMatrix decomposeToBezierForm();
    SparseMatrix insertKnots(double tau, unsigned int multiplicity = 1);
    // bool insertKnots(SparseMatrix &A, std::vector<tuple<double,int>> newKnots); // Add knots at several locations
    unsigned int knotMultiplicity(double tau)
        const; // Returns the number of repetitions of tau in the knot vector

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
    std::vector<double> getKnotVector() const { return knots; }
    unsigned int getBasisDegree() const { return degree; }
    double getKnotValue(unsigned int index) const;
    unsigned int getNumBasisFunctions() const;
    unsigned int getNumBasisFunctionsTarget() const;

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
    double deBoorCox(double x, int i, int k) const;

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
    unsigned int degree;
    std::vector<double> knots;
    unsigned int targetNumBasisfunctions;

    friend bool
    operator==(BSplineBasis1D const& lhs, BSplineBasis1D const& rhs);
    friend bool
    operator!=(BSplineBasis1D const& lhs, BSplineBasis1D const& rhs);
};

} // namespace SPLINTER

#endif // SPLINTER_BSPLINEBASIS1D_H
