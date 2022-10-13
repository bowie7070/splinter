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
#include <testingutilities.h>

using namespace SPLINTER;

#define COMMON_TAGS "[approximation][bspline]"
#define COMMON_TEXT " value approximation test with polynomials"

// TESTING LINEAR B-SPLINES
TEST_CASE(
    "Linear BSpline function" COMMON_TEXT " densely sampled",
    COMMON_TAGS "[bsplinetype::linear][function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.1;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        // If the degree of the exact function is less than or equal to the degree
        // of the B-Spline we are using to approximate it, the B-Spline should approximate
        // the function exactly.
        if (testFunc->isConstDegree() && testFunc->getMaxDegree() <= 1.0) {
            one_eps = 1e-5;
            two_eps = 1e-5;
            inf_eps = 1e-5;
        }

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<1>>(Builder(table).build<1>());
            },
            5000, // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Linear BSpline function" COMMON_TEXT " sampled with medium density",
    COMMON_TAGS "[bsplinetype::linear][function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.25;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        // If the degree of the exact function is less than or equal to the degree
        // of the B-Spline we are using to approximate it, the B-Spline should approximate
        // the function exactly.
        if (testFunc->isConstDegree() && testFunc->getMaxDegree() <= 1.0) {
            one_eps = 1e-5;
            two_eps = 1e-5;
            inf_eps = 1e-5;
        }

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<1>>(Builder(table).build<1>());
            },
            500,  // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Linear BSpline function" COMMON_TEXT " sparsely sampled",
    COMMON_TAGS "[bsplinetype::linear][function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.6;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        // If the degree of the exact function is less than or equal to the degree
        // of the B-Spline we are using to approximate it, the B-Spline should approximate
        // the function exactly.
        if (testFunc->isConstDegree() && testFunc->getMaxDegree() <= 1.0) {
            one_eps = 1e-5;
            two_eps = 1e-5;
            inf_eps = 1e-5;
        }

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<1>>(Builder(table).build<1>());
            },
            50,   // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Linear BSpline jacobian" COMMON_TEXT,
    COMMON_TAGS "[bsplinetype::linear][jacobian]") {
    double one_eps = 5e-5;
    double two_eps = 5e-5;
    double inf_eps = 5e-5;

    for (auto testFunc : getPolynomialFunctions()) {
        compareJacobianValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<1>>(Builder(table).build<1>());
            },
            300,  // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Linear BSpline hessian" COMMON_TEXT,
    COMMON_TAGS "[bsplinetype::linear][hessian]") {
    for (auto testFunc : getPolynomialFunctions()) {
        checkHessianSymmetry(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<1>>(Builder(table).build<1>());
            },
            300,
            1337);
    }
}

// TESTING QUADRATIC B-SPLINES
TEST_CASE(
    "Quadratic BSpline function" COMMON_TEXT " densely sampled",
    COMMON_TAGS "[bsplinetype::quadratic][function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.1;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        // If the degree of the exact function is less than or equal to the degree
        // of the B-Spline we are using to approximate it, the B-Spline should approximate
        // the function exactly.
        if (testFunc->isConstDegree() && testFunc->getMaxDegree() <= 2.0) {
            one_eps = 1e-5;
            two_eps = 1e-5;
            inf_eps = 1e-5;
        }

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<2>>(Builder(table).build<2>());
            },
            5000, // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Quadratic BSpline function" COMMON_TEXT " sampled with normal density",
    COMMON_TAGS "[bsplinetype::quadratic][function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.1;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        // If the degree of the exact function is less than or equal to the degree
        // of the B-Spline we are using to approximate it, the B-Spline should approximate
        // the function exactly.
        if (testFunc->isConstDegree() && testFunc->getMaxDegree() <= 2.0) {
            one_eps = 1e-5;
            two_eps = 1e-5;
            inf_eps = 1e-5;
        }

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<2>>(Builder(table).build<2>());
            },
            500,  // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Quadratic BSpline function" COMMON_TEXT " sparsely sampled",
    COMMON_TAGS "[bsplinetype::quadratic][function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.7;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        // If the degree of the exact function is less than or equal to the degree
        // of the B-Spline we are using to approximate it, the B-Spline should approximate
        // the function exactly.
        if (testFunc->isConstDegree() && testFunc->getMaxDegree() <= 2.0) {
            one_eps = 1e-5;
            two_eps = 1e-5;
            inf_eps = 1e-5;
        }

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<2>>(Builder(table).build<2>());
            },
            50,   // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Quadratic BSpline jacobian" COMMON_TEXT,
    COMMON_TAGS "[bsplinetype::quadratic][jacobian]") {
    double one_eps = 6e-5;
    double two_eps = 6e-5;
    double inf_eps = 6e-5;

    for (auto testFunc : getPolynomialFunctions()) {
        compareJacobianValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<2>>(Builder(table).build<2>());
            },
            300,  // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Quadratic BSpline hessian" COMMON_TEXT,
    COMMON_TAGS "[bsplinetype::quadratic][hessian]") {
    for (auto testFunc : getPolynomialFunctions()) {
        checkHessianSymmetry(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<2>>(Builder(table).build<2>());
            },
            300,   // Number of points to sample at
            1337); // Number of points to test against
    }
}

// TESTING CUBIC B-SPLINES
TEST_CASE(
    "Cubic BSpline function" COMMON_TEXT " densely sampled",
    COMMON_TAGS "[bsplinetype::cubic][function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.1;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        // If the degree of the exact function is less than or equal to the degree
        // of the B-Spline we are using to approximate it, the B-Spline should approximate
        // the function exactly.
        if (testFunc->isConstDegree() && testFunc->getMaxDegree() <= 3.0) {
            one_eps = 1e-5;
            two_eps = 1e-5;
            inf_eps = 1e-5;
        }

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<3>>(Builder(table).build<3>());
            },
            5000, // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Cubic BSpline function" COMMON_TEXT " sampled with normal density",
    COMMON_TAGS "[bsplinetype::cubic][function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.1;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        // If the degree of the exact function is less than or equal to the degree
        // of the B-Spline we are using to approximate it, the B-Spline should approximate
        // the function exactly.
        if (testFunc->isConstDegree() && testFunc->getMaxDegree() <= 3.0) {
            one_eps = 1e-5;
            two_eps = 1e-5;
            inf_eps = 1e-5;
        }

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<3>>(Builder(table).build<3>());
            },
            500,  // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Cubic BSpline function" COMMON_TEXT " sparsely sampled",
    COMMON_TAGS "[bsplinetype::cubic][function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.2;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        // If the degree of the exact function is less than or equal to the degree
        // of the B-Spline we are using to approximate it, the B-Spline should approximate
        // the function exactly.
        if (testFunc->isConstDegree() && testFunc->getMaxDegree() <= 3.0) {
            one_eps = 1e-5;
            two_eps = 1e-5;
            inf_eps = 1e-5;
        }

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<3>>(Builder(table).build<3>());
            },
            80,   // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Cubic BSpline jacobian" COMMON_TEXT,
    COMMON_TAGS "[bsplinetype::cubic][jacobian]") {
    double one_eps = 6e-5;
    double two_eps = 6e-5;
    double inf_eps = 6e-4;

    for (auto testFunc : getPolynomialFunctions()) {
        compareJacobianValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<3>>(Builder(table).build<3>());
            },
            300,  // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Cubic BSpline hessian" COMMON_TEXT,
    COMMON_TAGS "[bsplinetype::cubic][hessian]") {
    for (auto testFunc : getPolynomialFunctions()) {
        checkHessianSymmetry(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<3>>(Builder(table).build<3>());
            },
            300,   // Number of points to sample at
            1337); // Number of points to test against
    }
}

// TESTING QUARTIC B-SPLINES
TEST_CASE(
    "Quartic BSpline function" COMMON_TEXT " densely sampled",
    COMMON_TAGS "[bsplinetype::quartic][function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.1;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        // If the degree of the exact function is less than or equal to the degree
        // of the B-Spline we are using to approximate it, the B-Spline should approximate
        // the function exactly.
        if (testFunc->isConstDegree() && testFunc->getMaxDegree() <= 4.0) {
            one_eps = 1e-5;
            two_eps = 1e-5;
            inf_eps = 1e-5;
        }

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<4>>(Builder(table).build<4>());
            },
            5000, // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Quartic BSpline function" COMMON_TEXT " sampled with normal density",
    COMMON_TAGS "[bsplinetype::quartic][function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.1;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        // If the degree of the exact function is less than or equal to the degree
        // of the B-Spline we are using to approximate it, the B-Spline should approximate
        // the function exactly.
        if (testFunc->isConstDegree() && testFunc->getMaxDegree() <= 4.0) {
            one_eps = 1e-5;
            two_eps = 1e-5;
            inf_eps = 1e-5;
        }

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<4>>(Builder(table).build<4>());
            },
            500,  // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Quartic BSpline function" COMMON_TEXT " sparsely sampled",
    COMMON_TAGS "[bsplinetype::quartic][function-value]") {
    for (auto testFunc : getPolynomialFunctions()) {
        double one_eps = 0.1;
        double two_eps = 0.1;
        double inf_eps = 0.1;

        // If the degree of the exact function is less than or equal to the degree
        // of the B-Spline we are using to approximate it, the B-Spline should approximate
        // the function exactly.
        if (testFunc->isConstDegree() && testFunc->getMaxDegree() <= 4.0) {
            one_eps = 1e-5;
            two_eps = 1e-5;
            inf_eps = 1e-5;
        }

        compareFunctionValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<4>>(Builder(table).build<4>());
            },
            200,  // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Quartic BSpline jacobian" COMMON_TEXT,
    COMMON_TAGS "[bsplinetype::quartic][jacobian]") {
    double one_eps = 7e-5;
    double two_eps = 7e-5;
    double inf_eps = 7e-5;

    for (auto testFunc : getPolynomialFunctions()) {
        compareJacobianValue(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<4>>(Builder(table).build<4>());
            },
            300,  // Number of points to sample at
            1337, // Number of points to test against
            one_eps,
            two_eps,
            inf_eps);
    }
}

TEST_CASE(
    "Quartic BSpline hessian" COMMON_TEXT,
    COMMON_TAGS "[bsplinetype::quartic][hessian]") {
    for (auto testFunc : getPolynomialFunctions()) {
        checkHessianSymmetry(
            testFunc,
            [](auto const& table) {
                return std::make_unique<BSpline<4>>(Builder(table).build<4>());
            },
            300,   // Number of points to sample at
            1337); // Number of points to test against
    }
}
