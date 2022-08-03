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


namespace SPLINTER
{

inline auto sampleTestFunction()
{
    data_table_x<DenseVector> samples;

    // Sample function
    auto x0_vec = linspace(0,2,20);
    auto x1_vec = linspace(0,2,20);
    DenseVector x(2);
    double y;

    for (auto x0 : x0_vec)
    {
        for (auto x1 : x1_vec)
        {
            // Sample function at x
            x(0) = x0;
            x(1) = x1;
            y = sixHumpCamelBack(x);

            // Store sample
            samples.addSample(x,y);
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
bool domainReductionTest(BSpline &bs, const BSpline &bs_orig);
bool runRecursiveDomainReductionTest();

} // namespace SPLINTER

#endif //SPLINTER_BSPLINETESTINGUTILITIES_H
