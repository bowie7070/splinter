/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <bsplinebuilder.h>
#include <catch2/catch_all.hpp>
#include <testfunction.h>
#include <testingutilities.h>

using namespace SPLINTER;

#define COMMON_TAGS "[approximation][pspline]"
#define COMMON_TEXT " value approximation test with polynomials"

TEST_CASE("PSpline function" COMMON_TEXT, COMMON_TAGS "[function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.35;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<3>>(
                    Builder(table)
                        .smoothing(Smoothing::PSPLINE)
                        .alpha(0.03)
                        .build());
            },
            300,  // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE("PSpline jacobian" COMMON_TEXT, COMMON_TAGS "[jacobian]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 6e-6;
        double two_eps = 6e-6;
        double inf_eps = 6e-5;

        compareJacobianValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<3>>(
                    Builder(table)
                        .smoothing(Smoothing::PSPLINE)
                        .alpha(0.03)
                        .build());
            },
            300,  // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE("PSpline hessian" COMMON_TEXT, COMMON_TAGS "[hessian]") {
    for (auto testFunc : getPolynomialFunctions()) {
        checkHessianSymmetry(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<3>>(
                    Builder(table)
                        .smoothing(Smoothing::PSPLINE)
                        .alpha(0.03)
                        .build());
            },
            300,   // Number of points to sample at
            1337); // Number of points to test against
    }
}
