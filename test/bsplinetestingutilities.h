/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_BSPLINETESTINGUTILITIES_H
#define SPLINTER_BSPLINETESTINGUTILITIES_H

#include <bsplinebuilder.h>
#include <datatable.h>
#include <testingutilities.h>
#include <utilities.h>

namespace SPLINTER {

inline auto sampleTestFunction() {
    data_table_set_x<DenseVector> samples;

    // Sample function
    auto x0_vec = linspace(0, 2, 20);
    auto x1_vec = linspace(0, 2, 20);
    DenseVector x(2);
    double y;

    for (auto x0 : x0_vec) {
        for (auto x1 : x1_vec) {
            // Sample function at x
            x(0) = x0;
            x(1) = x1;
            y    = sixHumpCamelBack(x);

            // Store sample
            samples.addSample(x, y);
        }
    }

    return samples;
}

/*
 * Test knot insertion
 */
bool testKnotInsertion();

/*
 * Methods for B-spline domain reduction testing
 */
template <class S>
bool domainReductionTest(S& bs, S const& bs_orig) {
    if (bs.getNumVariables() != 2 || bs_orig.getNumVariables() != 2)
        return false;

    // Check for error
    if (!compareBSplines(bs, bs_orig))
        return false;

    auto lb = bs.getDomainLowerBound();
    auto ub = bs.getDomainUpperBound();

    bool flag          = false;
    unsigned int index = 0;
    for (; index < lb.size(); index++) {
        if (ub.at(index) - lb.at(index) > 1e-1) {
            flag = true;
            break;
        }
    }

    if (flag) {
        auto split = (ub.at(index) + lb.at(index)) / 2;

        auto lb2      = lb;
        auto ub2      = ub;
        ub2.at(index) = split;
        BSpline bs2(bs);
        bs2.reduceSupport(lb2, ub2);

        auto lb3      = lb;
        lb3.at(index) = split;
        auto ub3      = ub;
        BSpline bs3(bs);
        bs3.reduceSupport(lb3, ub3);

        return (
            domainReductionTest(bs2, bs_orig) &&
            domainReductionTest(bs3, bs_orig));
    }

    return true;
}

bool runRecursiveDomainReductionTest();

} // namespace SPLINTER

#endif //SPLINTER_BSPLINETESTINGUTILITIES_H
