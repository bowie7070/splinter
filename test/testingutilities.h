/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_TESTINGUTILITIES_H
#define SPLINTER_TESTINGUTILITIES_H

#include "definitions.h"
#include <bspline.h>
#include <catch2/catch_all.hpp>
#include <datatable.h>
#include <function.h>
#include <iomanip>
#include <iostream>
#include <testfunction.h>

namespace SPLINTER {

double sixHumpCamelBack(DenseVector x);

double getOneNorm(DenseMatrix const& m);

double getTwoNorm(DenseMatrix const& m);

double getInfNorm(DenseMatrix const& m);

std::vector<double> denseToVec(DenseVector const& dense);

DenseVector vecToDense(std::vector<double> const& vec);

// points is a vector where each element is the number of points for that dim
std::vector<std::vector<double>> linspace(
    std::vector<double> start,
    std::vector<double> end,
    std::vector<unsigned int> points);

// points is the total number of points, not per dim
std::vector<std::vector<double>>
linspace(int dim, double start, double end, unsigned int points);

// Returns a default linspace of dim dim
std::vector<std::vector<double>> linspace(int dim);

std::vector<std::vector<double>> linspace(int dim, unsigned int pointsPerDim);

inline auto
sample(Function const& func, std::vector<std::vector<double>>& points) {
    data_table_set_x<std::vector<double>> table;

    for (auto& x : points) {
        table.addSample(x, func.eval(x));
    }

    return table;
}

inline auto
sample(Function const* func, std::vector<std::vector<double>>& points) {
    return sample(*func, points);
}

enum class TestType { All, FunctionValue, Jacobian, Hessian };

double getError(double exactVal, double approxVal);

bool equalsWithinRange(double a, double b, double margin = 0.0);

/*
 * Checks that the hessian is symmetric across the diagonal
 */
template <class callable>
bool isSymmetricHessian(callable const& approx, DenseVector const& x) {
    DenseMatrix hessian = approx.evalHessian(x);

    for (int row = 0; row < (int)hessian.rows(); ++row) {
        for (int col = 0; col < (int)hessian.cols(); ++col) {
            if (getError(hessian(row, col), hessian(col, row)) > 1e-9) {
                return false;
            }
        }
    }

    return true;
}

template <class T>
struct is_unique_ptr : std::false_type {};

template <class... Ts>
struct is_unique_ptr<std::unique_ptr<Ts...>> : std::true_type {};

template <class T>
constexpr bool is_unique_ptr_v = is_unique_ptr<T>::value;

template <class callable>
void compareFunctionValue(
    TestFunction* exact,
    callable approx_gen_func,
    size_t numSamplePoints,
    size_t numEvalPoints,
    double one_eps,
    double two_eps,
    double inf_eps) {
    auto dim = exact->getNumVariables();

    auto samplePoints =
        linspace(dim, -5, 5, std::pow(numSamplePoints, 1.0 / dim));
    auto evalPoints = linspace(dim, -5, 5, std::pow(numEvalPoints, 1.0 / dim));

    auto table = sample(exact, samplePoints);

    auto approx{approx_gen_func(table)};
    static_assert(is_unique_ptr_v<decltype(approx)>);

    INFO("Approximant: " << approx->getDescription());
    INFO("Function: " << exact->getFunctionStr());

    DenseVector errorVec(evalPoints.size());

    double maxError           = 0.0;
    DenseVector maxErrorPoint = vecToDense(evalPoints.at(0));

    int i = 0;
    for (auto& point : evalPoints) {
        DenseVector x = vecToDense(point);

        double exactValue  = exact->eval(x);
        double approxValue = approx->eval(x);
        double error       = getError(exactValue, approxValue);

        if (error > maxError) {
            maxError      = error;
            maxErrorPoint = x;
        }

        errorVec(i) = error;

        i++;
    }

    DenseVector norms(3);
    norms(0) = getOneNorm(errorVec);
    norms(1) = getTwoNorm(errorVec);
    norms(2) = getInfNorm(errorVec);

    INFO(
        std::setw(16) << std::left << "1-norm (\"avg\"):" << std::setw(16)
                      << std::right << norms(0) / evalPoints.size()
                      << " <= " << one_eps);
    INFO(
        std::setw(16) << std::left << "2-norm:" << std::setw(16) << std::right
                      << norms(1) << " <= " << two_eps);
    INFO(
        std::setw(16) << std::left << "Inf-norm:" << std::setw(16) << std::right
                      << norms(2) << " <= " << inf_eps);

    // Print out the point with the largest error
    std::string maxErrorPointStr("(");
    for (size_t i = 0; i < (size_t)maxErrorPoint.size(); ++i) {
        if (i != 0) {
            maxErrorPointStr.append(", ");
        }
        maxErrorPointStr.append(std::to_string(maxErrorPoint(i)));
    }
    maxErrorPointStr.append(")");
    INFO("");
    INFO(
        std::setw(16) << std::left << "Max error:" << std::setw(16)
                      << std::right << maxError);
    INFO(" at " << maxErrorPointStr);
    INFO(
        std::setw(16) << std::left << "Exact value:" << std::setw(16)
                      << std::right << exact->eval(maxErrorPoint));
    INFO(
        std::setw(16) << std::left << "Approx value:" << std::setw(16)
                      << std::right << approx->eval(maxErrorPoint));

    CHECK(norms(0) / evalPoints.size() <= one_eps);
    /*if(norms(0) / evalPoints.size() > one_eps*/
    /* || norms(1) > two_eps || norms(2) > inf_eps*/ /*) {
        CHECK(false);
    }*/
}

/*
 * Compares the function value of the Approximant generated by calling
 * approx_gen_func with all the exact functions in funcs.
 * Prints out the 1, 2 and inf norms for all functions if one of the
 * norms are larger than the corresponding epsilon.
 * Also prints out the point with the largest error.
 */
template <class callable>
void compareFunctionValue(
    std::vector<TestFunction*> funcs,
    callable approx_gen_func,
    size_t numSamplePoints,
    size_t numEvalPoints,
    double one_eps,
    double two_eps,
    double inf_eps) {
    for (auto& exact : funcs) {
        compareFunctionValue(
            exact,
            approx_gen_func,
            numSamplePoints,
            numEvalPoints,
            one_eps,
            two_eps,
            inf_eps);
    }
}

template <class callable>
void compareJacobianValue(
    TestFunction* exact,
    callable approx_gen_func,
    size_t numSamplePoints,
    size_t numEvalPoints,
    double one_eps,
    double two_eps,
    double inf_eps) {
    auto dim = exact->getNumVariables();

    auto samplePoints =
        linspace(dim, -5, 5, std::pow(numSamplePoints, 1.0 / dim));
    auto evalPoints =
        linspace(dim, -4.95, 4.95, std::pow(numEvalPoints, 1.0 / dim));

    auto table = sample(exact, samplePoints);

    auto approx{approx_gen_func(table)};
    static_assert(is_unique_ptr_v<decltype(approx)>);

    INFO("Approximant: " << approx->getDescription());
    INFO("Function: " << exact->getFunctionStr());

    DenseVector oneNormVec(evalPoints.size());
    DenseVector twoNormVec(evalPoints.size());
    DenseVector infNormVec(evalPoints.size());

    double maxOneNormError = 0.0;
    double maxTwoNormError = 0.0;
    double maxInfNormError = 0.0;

    DenseVector maxOneNormErrorPoint(dim);
    maxOneNormErrorPoint.fill(0.0);
    DenseVector maxTwoNormErrorPoint(dim);
    maxTwoNormErrorPoint.fill(0.0);
    DenseVector maxInfNormErrorPoint(dim);
    maxInfNormErrorPoint.fill(0.0);

    int i = 0;
    for (auto& point : evalPoints) {
        DenseVector x = vecToDense(point);

        // Compare the central difference to the approximated jacobian
        DenseMatrix exactValue  = approx->centralDifference(x);
        DenseMatrix approxValue = approx->evalJacobian(x);

        DenseVector error = DenseVector::Zero(exactValue.cols());
        for (size_t j = 0; j < (size_t)error.size(); ++j) {
            error(j) = getError(exactValue(j), approxValue(j));
        }

        oneNormVec(i) = getOneNorm(error) / error.size(); // "Average"
        twoNormVec(i) = getTwoNorm(error);
        infNormVec(i) = getInfNorm(error);

        if (oneNormVec(i) > maxOneNormError) {
            maxOneNormError      = oneNormVec(i);
            maxOneNormErrorPoint = x;
        }
        if (twoNormVec(i) > maxTwoNormError) {
            maxTwoNormError      = twoNormVec(i);
            maxTwoNormErrorPoint = x;
        }
        if (infNormVec(i) > maxInfNormError) {
            maxInfNormError      = infNormVec(i);
            maxInfNormErrorPoint = x;
        }

        i++;
    }

    DenseVector norms(3);
    norms(0) = getOneNorm(oneNormVec);
    norms(1) = getTwoNorm(twoNormVec);
    norms(2) = getInfNorm(infNormVec);

    INFO(
        std::setw(16) << std::left << "1-norm (\"avg\"):" << std::setw(16)
                      << std::right << norms(0) / evalPoints.size()
                      << " <= " << one_eps);
    INFO(
        std::setw(16) << std::left << "2-norm:" << std::setw(16) << std::right
                      << norms(1) << " <= " << two_eps);
    INFO(
        std::setw(16) << std::left << "Inf-norm:" << std::setw(16) << std::right
                      << norms(2) << " <= " << inf_eps);

    auto getDenseAsStrOneLine = [](DenseMatrix const& x) {
        std::string denseAsStrOneLine("(");
        for (size_t i = 0; i < (size_t)x.size(); ++i) {
            if (i != 0) {
                denseAsStrOneLine.append(", ");
            }
            denseAsStrOneLine.append(std::to_string(x(i)));
        }
        denseAsStrOneLine.append(")");
        return denseAsStrOneLine;
    };

    // Print out the points with the largest errors
    INFO("");
    INFO("Max errors:");
    INFO("");
    INFO(
        std::setw(16) << std::left << "1-norm:" << std::setw(32) << std::right
                      << maxOneNormError);
    INFO(" at " << getDenseAsStrOneLine(maxOneNormErrorPoint));
    INFO(
        std::setw(16) << std::left << "Approx value:" << std::setw(32)
                      << std::right
                      << getDenseAsStrOneLine(
                             approx->evalJacobian(maxOneNormErrorPoint)));
    INFO(
        std::setw(16) << std::left << "Central difference:" << std::setw(32)
                      << std::right
                      << getDenseAsStrOneLine(
                             approx->centralDifference(maxOneNormErrorPoint)));

    INFO("");
    INFO(
        std::setw(16) << std::left << "2-norm:" << std::setw(32) << std::right
                      << maxTwoNormError);
    INFO(" at " << getDenseAsStrOneLine(maxTwoNormErrorPoint));
    INFO(
        std::setw(16) << std::left << "Approx value:" << std::setw(32)
                      << std::right
                      << getDenseAsStrOneLine(
                             approx->evalJacobian(maxTwoNormErrorPoint)));
    INFO(
        std::setw(16) << std::left << "Central difference:" << std::setw(32)
                      << std::right
                      << getDenseAsStrOneLine(
                             approx->centralDifference(maxTwoNormErrorPoint)));

    INFO("");
    INFO(
        std::setw(16) << std::left << "Inf-norm:" << std::setw(32) << std::right
                      << maxInfNormError);
    INFO(" at " << getDenseAsStrOneLine(maxInfNormErrorPoint));
    INFO(
        std::setw(16) << std::left << "Approx value:" << std::setw(32)
                      << std::right
                      << getDenseAsStrOneLine(
                             approx->evalJacobian(maxInfNormErrorPoint)));
    INFO(
        std::setw(16) << std::left << "Central difference:" << std::setw(32)
                      << std::right
                      << getDenseAsStrOneLine(
                             approx->centralDifference(maxInfNormErrorPoint)));

    CHECK(norms(2) <= inf_eps);
    //CHECK(norms(0) / evalPoints.size() <= one_eps);
    /*if(norms(0) / evalPoints.size() > one_eps || norms(1) > two_eps || norms(2) > inf_eps) {
        CHECK(false);
    }*/
}

template <class callable>
void checkHessianSymmetry(
    TestFunction* exact,
    callable approx_gen_func,
    size_t numSamplePoints,
    size_t numEvalPoints) {
    auto dim = exact->getNumVariables();

    auto samplePoints =
        linspace(dim, -5, 5, std::pow(numSamplePoints, 1.0 / dim));
    auto evalPoints =
        linspace(dim, -4.95, 4.95, std::pow(numEvalPoints, 1.0 / dim));

    auto table = sample(exact, samplePoints);

    auto approx{approx_gen_func(table)};
    static_assert(is_unique_ptr_v<decltype(approx)>);

    INFO("Approximant: " << approx->getDescription());
    INFO("Function: " << exact->getFunctionStr());

    bool allSymmetric = true;

    DenseVector x(dim);
    for (auto& point : evalPoints) {
        x = vecToDense(point);

        if (!isSymmetricHessian(*approx, x)) {
            allSymmetric = false;
            break;
        }
    }

    std::string x_str;
    for (size_t i = 0; i < (size_t)x.size(); ++i) {
        if (i != 0) {
            x_str.append(", ");
        }
        x_str.append(std::to_string(x(i)));
    }
    INFO("Approximated hessian at " << x_str << ":");
    INFO(approx->evalHessian(x));
    CHECK(allSymmetric);
}

template <class exact_type, class approx_type>
bool compareFunctions(
    exact_type const& exact,
    approx_type const& approx,
    std::vector<std::vector<double>> const& points,
    double one_norm_epsilon,
    double two_norm_epsilon,
    double inf_norm_epsilon) {
    bool equal = true;

    REQUIRE(exact.getNumVariables() == approx.getNumVariables());

    DenseVector normOneValVec(points.size());
    DenseVector normTwoValVec(points.size());
    DenseVector normInfValVec(points.size());

    DenseVector normOneJacVec(points.size());
    DenseVector normTwoJacVec(points.size());
    DenseVector normInfJacVec(points.size());

    DenseVector normOneHesVec(points.size());
    DenseVector normTwoHesVec(points.size());
    DenseVector normInfHesVec(points.size());

    int i = 0;
    for (auto& point : points) {
        DenseVector x = vecToDense(point);

        //        INFO("Evaluation point: " << pretty_print(x));

        /*SECTION("Function approximates the value within tolerance")*/
        {
            DenseMatrix exactValue(1, 1);
            exactValue(0, 0) = exact.eval(x);
            DenseMatrix approxValue(1, 1);
            approxValue(0, 0) = approx.eval(x);
            DenseMatrix error = exactValue - approxValue;

            //            INFO("Exact value:");
            //            INFO(exactValue);
            //            INFO("Approximated value:");
            //            INFO(approxValue);
            //            INFO("Exact - approx:");
            //            INFO(error);

            normOneValVec(i) = getOneNorm(error);
            normTwoValVec(i) = getTwoNorm(error);
            normInfValVec(i) = getInfNorm(error);

            //            REQUIRE(oneNorm(error) <= one_norm_epsilon);
            //            REQUIRE(twoNorm(error) <= two_norm_epsilon);
            //            REQUIRE(maxNorm(error) <= inf_norm_epsilon);
        }

        /*SECTION("Function approximates the Jacobian within tolerance")*/
        {
            auto exactJacobian  = exact.evalJacobian(x);
            auto approxJacobian = approx.evalJacobian(x);
            auto errorJacobian  = exactJacobian - approxJacobian;

            normOneJacVec(i) = getOneNorm(errorJacobian);
            normTwoJacVec(i) = getTwoNorm(errorJacobian);
            normInfJacVec(i) = getInfNorm(errorJacobian);
            //            INFO("Exact Jacobian:");
            //            INFO(exactJacobian);
            //            INFO("Approximated Jacobian:");
            //            INFO(approxJacobian);
            //            INFO("Exact - Approx: ");
            //            INFO(errorJacobian);
            //
            //            REQUIRE(oneNorm(errorJacobian) <= one_norm_epsilon);
            //            REQUIRE(twoNorm(errorJacobian) <= two_norm_epsilon);
            //            REQUIRE(maxNorm(errorJacobian) <= inf_norm_epsilon);
        }

        /*SECTION("Function approximates the Hessian within tolerance")*/
        {
            auto exactHessian  = exact.evalHessian(x);
            auto approxHessian = approx.evalHessian(x);
            auto errorHessian  = exactHessian - approxHessian;

            normOneHesVec(i) = getOneNorm(errorHessian);
            normTwoHesVec(i) = getTwoNorm(errorHessian);
            normInfHesVec(i) = getInfNorm(errorHessian);

            //            INFO("x: ");
            //            INFO(x);
            //            INFO("Exact Hessian:");
            //            INFO(exactHessian);
            //            INFO("Approximated Hessian:");
            //            INFO(approxHessian);
            //            INFO("Exact - Approx: ");
            //            INFO(errorHessian);

            //            CHECK(getOneNorm(errorHessian) <= one_norm_epsilon);
            //            CHECK(getTwoNorm(errorHessian) <= two_norm_epsilon);
            //            CHECK(getInfNorm(errorHessian) <= inf_norm_epsilon);
        }

        i++;
    }

    DenseVector valNorms(3);
    valNorms(0) = getOneNorm(normOneValVec);
    valNorms(1) = getTwoNorm(normTwoValVec);
    valNorms(2) = getInfNorm(normInfValVec);

    DenseVector jacNorms(3);
    jacNorms(0) = getOneNorm(normOneJacVec);
    jacNorms(1) = getTwoNorm(normTwoJacVec);
    jacNorms(2) = getInfNorm(normInfJacVec);

    DenseVector hesNorms(3);
    hesNorms(0) = getOneNorm(normOneHesVec);
    hesNorms(1) = getTwoNorm(normTwoHesVec);
    hesNorms(2) = getInfNorm(normInfHesVec);

    if (valNorms(0) / points.size() > one_norm_epsilon) {
        INFO(
            "1-norm function value (\"avg\"): " << valNorms(0) / points.size());
        equal = false;
    }
    if (valNorms(1) > two_norm_epsilon) {
        INFO("2-norm function value: " << valNorms(1));
        equal = false;
    }
    if (valNorms(2) > inf_norm_epsilon) {
        INFO("inf-norm function value: " << valNorms(2));
        equal = false;
    }

    if (jacNorms(0) / points.size() > one_norm_epsilon) {
        INFO(
            "1-norm jacobian value (\"avg\"): " << jacNorms(0) / points.size());
        equal = false;
    }
    if (jacNorms(1) > two_norm_epsilon) {
        INFO("2-norm jacobian value: " << jacNorms(1));
        equal = false;
    }
    if (jacNorms(2) > inf_norm_epsilon) {
        INFO("inf-norm jacobian value: " << jacNorms(2));
        equal = false;
    }

    if (hesNorms(0) / points.size() > one_norm_epsilon) {
        INFO("1-norm hessian value (\"avg\"): " << hesNorms(0) / points.size());
        equal = false;
    }
    if (hesNorms(1) > two_norm_epsilon) {
        INFO("2-norm hessian value: " << hesNorms(1));
        equal = false;
    }
    if (hesNorms(2) > inf_norm_epsilon) {
        INFO("inf-norm hessian value: " << hesNorms(2));
        equal = false;
    }

    return equal;
}

template <class exact_type, class approx_type>
bool compareFunctions(
    exact_type const& exact,
    approx_type const& approx,
    std::vector<std::vector<double>> const& points) {
    // Max value of the norms of function/jacobian/hessian errors
    double const one_norm_epsilon = 0.1;
    double const two_norm_epsilon = 0.1;
    double const inf_norm_epsilon = 0.1;

    return compareFunctions(
        exact,
        approx,
        points,
        one_norm_epsilon,
        two_norm_epsilon,
        inf_norm_epsilon);
}

template <class S>
bool compareBSplines(S const& left, S const& right) {
    auto left_lb  = left.getDomainLowerBound();
    auto left_ub  = left.getDomainUpperBound();
    auto right_lb = right.getDomainLowerBound();
    auto right_ub = right.getDomainUpperBound();

    REQUIRE(left_lb.size() == left_ub.size());
    REQUIRE(left_ub.size() == right_lb.size());
    REQUIRE(right_lb.size() == right_ub.size());

    int dim = left_lb.size();

    auto points = linspace(dim);

    for (int i = 0; i < dim; i++) {
        REQUIRE(left_lb.at(i) == right_lb.at(i));
        REQUIRE(left_ub.at(i) == right_ub.at(i));
    }

    return compareFunctions(left, right, points);

    //    auto x0_vec = linspace(lb.at(0), ub.at(0), 10);
    //    auto x1_vec = linspace(lb.at(1), ub.at(1), 10);
    //
    //    DenseVector x(2);
    //    for (auto x0 : x0_vec)
    //    {
    //        for (auto x1 : x1_vec)
    //        {
    //            x(0) = x0;
    //            x(1) = x1;
    //
    //            double yb = bs.eval(x);
    //            double yb_orig = bs_orig.eval(x);
    //            if (std::abs(yb-yb_orig) > 1e-8)
    //            {
    //                cout << yb << endl;
    //                cout << yb_orig << endl;
    //                return false;
    //            }
    //        }
    //    }
    //
    //    return true;
}

/*
 * Computes the central difference at x. Returns a 1xN row-vector.
 */
DenseMatrix centralDifference(Function const& approx, DenseVector const& x);

// returns log(x) in base base
double log(double base, double x);

std::string pretty_print(DenseVector const& denseVec);

TestFunction* getTestFunction(int numVariables, int degree);
std::vector<TestFunction*> getTestFunctionsOfDegree(int degree);
std::vector<TestFunction*> getTestFunctionWithNumVariables(int numVariables);
std::vector<TestFunction*> getPolynomialFunctions();
std::vector<TestFunction*> getNastyTestFunctions();

/*
 * Returns 3x3 matrix,
 * first row: function value error norms
 * second row: jacobian value error norms
 * third row: hessian value error norms
 * first col: 1-norms
 * second col: 2-norms
 * third col: inf-norms
 */
DenseMatrix getErrorNorms(
    Function const* exact,
    Function const* approx,
    std::vector<std::vector<double>> const& points);

void checkNorms(
    DenseMatrix normValues,
    size_t numPoints,
    double one_eps,
    double two_eps,
    double inf_eps);
void checkNorm(
    DenseMatrix normValues,
    TestType type,
    size_t numPoints,
    double one_eps,
    double two_eps,
    double inf_eps);
void _checkNorm(
    DenseMatrix normValues,
    int row,
    size_t numPoints,
    double one_eps,
    double two_eps,
    double inf_eps);

template <class callable>
void testApproximation(
    std::vector<TestFunction*> funcs,
    callable approx_gen_func,
    TestType type,
    size_t numSamplePoints,
    size_t numEvalPoints,
    double one_eps,
    double two_eps,
    double inf_eps) {
    for (auto& exact : funcs) {

        auto dim = exact->getNumVariables();
        CHECK(dim > 0);
        if (dim > 0) {
            auto samplePoints =
                linspace(dim, -5, 5, std::pow(numSamplePoints, 1.0 / dim));
            auto evalPoints =
                linspace(dim, -4.95, 4.95, std::pow(numEvalPoints, 1.0 / dim));

            auto table = sample(exact, samplePoints);

            Function* approx = approx_gen_func(table);

            INFO("Function: " << exact->getFunctionStr());
            INFO("Approximant: " << approx->getDescription());

            DenseMatrix errorNorms = getErrorNorms(exact, approx, evalPoints);

            checkNorm(
                errorNorms,
                type,
                evalPoints.size(),
                one_eps,
                two_eps,
                inf_eps);

            delete approx;
        }
    }
}

} // namespace SPLINTER

#endif // SPLINTER_TESTINGUTILITIES_H
