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

#include <datatable.h>
#include <function.h>
#include "definitions.h"
#include <bspline.h>
#include <operatoroverloads.h>
#include <testfunction.h>
#include <iostream>
#include <iomanip>
#include <catch2/catch_all.hpp>




namespace SPLINTER
{

double sixHumpCamelBack(DenseVector x);

double getOneNorm(const DenseMatrix &m);

double getTwoNorm(const DenseMatrix &m);

double getInfNorm(const DenseMatrix &m);

std::vector<double> denseToVec(const DenseVector &dense);

DenseVector vecToDense(const std::vector<double> &vec);

// points is a vector where each element is the number of points for that dim
std::vector<std::vector<double>> linspace(std::vector<double> start, std::vector<double> end, std::vector<unsigned int> points);

// points is the total number of points, not per dim
std::vector<std::vector<double>> linspace(int dim, double start, double end, unsigned int points);

// Returns a default linspace of dim dim
std::vector<std::vector<double>> linspace(int dim);

std::vector<std::vector<double>> linspace(int dim, unsigned int pointsPerDim);

DataTable sample(const Function &func, std::vector<std::vector<double>> &points);
DataTable sample(const Function *func, std::vector<std::vector<double>> &points);

enum class TestType {
    All,
    FunctionValue,
    Jacobian,
    Hessian
};

double getError(double exactVal, double approxVal);

bool equalsWithinRange(double a, double b, double margin = 0.0);

/*
 * Checks that the hessian is symmetric across the diagonal
 */
template <class callable>
bool isSymmetricHessian(const callable &approx, const DenseVector &x)
{
    DenseMatrix hessian = approx.evalHessian(x);

    for(int row = 0; row < (int) hessian.rows(); ++row)
    {
        for(int col = 0; col < (int) hessian.cols(); ++col)
        {
            if(getError(hessian(row, col), hessian(col, row)) > 1e-9)
            {
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
void compareFunctionValue(TestFunction *exact,
                          callable approx_gen_func,
                          size_t numSamplePoints, size_t numEvalPoints,
                          double one_eps, double two_eps, double inf_eps)
{
    auto dim = exact->getNumVariables();

    auto samplePoints = linspace(dim, -5, 5, std::pow(numSamplePoints, 1.0 / dim));
    auto evalPoints = linspace(dim, -5, 5, std::pow(numEvalPoints, 1.0 / dim));

    DataTable table = sample(exact, samplePoints);

    auto approx{ approx_gen_func(table) };
    static_assert(is_unique_ptr_v<decltype(approx)>);


    INFO("Approximant: " << approx->getDescription());
    INFO("Function: " << exact->getFunctionStr());

    DenseVector errorVec(evalPoints.size());

    double maxError = 0.0;
    DenseVector maxErrorPoint = vecToDense(evalPoints.at(0));

    int i = 0;
    for (auto& point : evalPoints)
    {
        DenseVector x = vecToDense(point);

        double exactValue = exact->eval(x);
        double approxValue = approx->eval(x);
        double error = getError(exactValue, approxValue);

        if (error > maxError)
        {
            maxError = error;
            maxErrorPoint = x;
        }

        errorVec(i) = error;

        i++;
    }

    DenseVector norms(3);
    norms(0) = getOneNorm(errorVec);
    norms(1) = getTwoNorm(errorVec);
    norms(2) = getInfNorm(errorVec);

    INFO(std::setw(16) << std::left << "1-norm (\"avg\"):" << std::setw(16) << std::right << norms(0) / evalPoints.size() << " <= " << one_eps);
    INFO(std::setw(16) << std::left << "2-norm:" << std::setw(16) << std::right << norms(1) << " <= " << two_eps);
    INFO(std::setw(16) << std::left << "Inf-norm:" << std::setw(16) << std::right << norms(2) << " <= " << inf_eps);


    // Print out the point with the largest error
    std::string maxErrorPointStr("(");
    for (size_t i = 0; i < (size_t)maxErrorPoint.size(); ++i)
    {
        if (i != 0)
        {
            maxErrorPointStr.append(", ");
        }
        maxErrorPointStr.append(std::to_string(maxErrorPoint(i)));
    }
    maxErrorPointStr.append(")");
    INFO("");
    INFO(std::setw(16) << std::left << "Max error:" << std::setw(16) << std::right << maxError);
    INFO(" at " << maxErrorPointStr);
    INFO(std::setw(16) << std::left << "Exact value:" << std::setw(16) << std::right << exact->eval(maxErrorPoint));
    INFO(std::setw(16) << std::left << "Approx value:" << std::setw(16) << std::right << approx->eval(maxErrorPoint));

    CHECK(norms(0) / evalPoints.size() <= one_eps);
    /*if(norms(0) / evalPoints.size() > one_eps*//* || norms(1) > two_eps || norms(2) > inf_eps*//*) {
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
void compareFunctionValue(std::vector<TestFunction*> funcs,
    callable approx_gen_func,
    size_t numSamplePoints, size_t numEvalPoints,
    double one_eps, double two_eps, double inf_eps)
{
    for (auto& exact : funcs)
    {
        compareFunctionValue(exact, approx_gen_func, numSamplePoints, numEvalPoints, one_eps, two_eps, inf_eps);
    }
}

template <class callable>
void compareJacobianValue(TestFunction *exact,
                          callable approx_gen_func,
                          size_t numSamplePoints, size_t numEvalPoints,
                          double one_eps, double two_eps, double inf_eps)
{
    auto dim = exact->getNumVariables();

    auto samplePoints = linspace(dim, -5, 5, std::pow(numSamplePoints, 1.0 / dim));
    auto evalPoints = linspace(dim, -4.95, 4.95, std::pow(numEvalPoints, 1.0 / dim));

    DataTable table = sample(exact, samplePoints);

    auto approx{ approx_gen_func(table) };
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
    for (auto& point : evalPoints)
    {
        DenseVector x = vecToDense(point);

        // Compare the central difference to the approximated jacobian
        DenseMatrix exactValue = approx->centralDifference(x);
        DenseMatrix approxValue = approx->evalJacobian(x);

        DenseVector error = DenseVector::Zero(exactValue.cols());
        for (size_t j = 0; j < (size_t)error.size(); ++j)
        {
            error(j) = getError(exactValue(j), approxValue(j));
        }

        oneNormVec(i) = getOneNorm(error) / error.size(); // "Average"
        twoNormVec(i) = getTwoNorm(error);
        infNormVec(i) = getInfNorm(error);

        if (oneNormVec(i) > maxOneNormError)
        {
            maxOneNormError = oneNormVec(i);
            maxOneNormErrorPoint = x;
        }
        if (twoNormVec(i) > maxTwoNormError)
        {
            maxTwoNormError = twoNormVec(i);
            maxTwoNormErrorPoint = x;
        }
        if (infNormVec(i) > maxInfNormError)
        {
            maxInfNormError = infNormVec(i);
            maxInfNormErrorPoint = x;
        }

        i++;
    }

    DenseVector norms(3);
    norms(0) = getOneNorm(oneNormVec);
    norms(1) = getTwoNorm(twoNormVec);
    norms(2) = getInfNorm(infNormVec);

    INFO(std::setw(16) << std::left << "1-norm (\"avg\"):" << std::setw(16) << std::right << norms(0) / evalPoints.size() << " <= " << one_eps);
    INFO(std::setw(16) << std::left << "2-norm:" << std::setw(16) << std::right << norms(1) << " <= " << two_eps);
    INFO(std::setw(16) << std::left << "Inf-norm:" << std::setw(16) << std::right << norms(2) << " <= " << inf_eps);


    auto getDenseAsStrOneLine = [](const DenseMatrix& x) {
        std::string denseAsStrOneLine("(");
        for (size_t i = 0; i < (size_t)x.size(); ++i)
        {
            if (i != 0)
            {
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
    INFO(std::setw(16) << std::left << "1-norm:" << std::setw(32) << std::right << maxOneNormError);
    INFO(" at " << getDenseAsStrOneLine(maxOneNormErrorPoint));
    INFO(std::setw(16) << std::left << "Approx value:" << std::setw(32) << std::right << getDenseAsStrOneLine(approx->evalJacobian(maxOneNormErrorPoint)));
    INFO(std::setw(16) << std::left << "Central difference:" << std::setw(32) << std::right << getDenseAsStrOneLine(approx->centralDifference(maxOneNormErrorPoint)));

    INFO("");
    INFO(std::setw(16) << std::left << "2-norm:" << std::setw(32) << std::right << maxTwoNormError);
    INFO(" at " << getDenseAsStrOneLine(maxTwoNormErrorPoint));
    INFO(std::setw(16) << std::left << "Approx value:" << std::setw(32) << std::right << getDenseAsStrOneLine(approx->evalJacobian(maxTwoNormErrorPoint)));
    INFO(std::setw(16) << std::left << "Central difference:" << std::setw(32) << std::right << getDenseAsStrOneLine(approx->centralDifference(maxTwoNormErrorPoint)));

    INFO("");
    INFO(std::setw(16) << std::left << "Inf-norm:" << std::setw(32) << std::right << maxInfNormError);
    INFO(" at " << getDenseAsStrOneLine(maxInfNormErrorPoint));
    INFO(std::setw(16) << std::left << "Approx value:" << std::setw(32) << std::right << getDenseAsStrOneLine(approx->evalJacobian(maxInfNormErrorPoint)));
    INFO(std::setw(16) << std::left << "Central difference:" << std::setw(32) << std::right << getDenseAsStrOneLine(approx->centralDifference(maxInfNormErrorPoint)));

    CHECK(norms(2) <= inf_eps);
    //CHECK(norms(0) / evalPoints.size() <= one_eps);
    /*if(norms(0) / evalPoints.size() > one_eps || norms(1) > two_eps || norms(2) > inf_eps) {
        CHECK(false);
    }*/
}

template <class callable>
void checkHessianSymmetry(TestFunction *exact,
                          callable approx_gen_func,
                          size_t numSamplePoints, size_t numEvalPoints)
{
    auto dim = exact->getNumVariables();

    auto samplePoints = linspace(dim, -5, 5, std::pow(numSamplePoints, 1.0 / dim));
    auto evalPoints = linspace(dim, -4.95, 4.95, std::pow(numEvalPoints, 1.0 / dim));

    DataTable table = sample(exact, samplePoints);

    auto approx{ approx_gen_func(table) };
    static_assert(is_unique_ptr_v<decltype(approx)>);

    INFO("Approximant: " << approx->getDescription());
    INFO("Function: " << exact->getFunctionStr());

    bool allSymmetric = true;

    DenseVector x(dim);
    for (auto& point : evalPoints)
    {
        x = vecToDense(point);

        if (!isSymmetricHessian(*approx, x))
        {
            allSymmetric = false;
            break;
        }
    }

    std::string x_str;
    for (size_t i = 0; i < (size_t)x.size(); ++i)
    {
        if (i != 0)
        {
            x_str.append(", ");
        }
        x_str.append(std::to_string(x(i)));
    }
    INFO("Approximated hessian at " << x_str << ":");
    INFO(approx->evalHessian(x));
    CHECK(allSymmetric);
}

bool compareBSplines(const BSpline &left, const BSpline &right);

/*
 * Computes the central difference at x. Returns a 1xN row-vector.
 */
DenseMatrix centralDifference(const Function &approx, const DenseVector &x);

// returns log(x) in base base
double log(double base, double x);

std::string pretty_print(const DenseVector &denseVec);

TestFunction *getTestFunction(int numVariables, int degree);
std::vector<TestFunction *> getTestFunctionsOfDegree(int degree);
std::vector<TestFunction *> getTestFunctionWithNumVariables(int numVariables);
std::vector<TestFunction *> getPolynomialFunctions();
std::vector<TestFunction *> getNastyTestFunctions();

/*
 * Returns 3x3 matrix,
 * first row: function value error norms
 * second row: jacobian value error norms
 * third row: hessian value error norms
 * first col: 1-norms
 * second col: 2-norms
 * third col: inf-norms
 */
DenseMatrix getErrorNorms(const Function *exact, const Function *approx, const std::vector<std::vector<double>> &points);

void checkNorms(DenseMatrix normValues, size_t numPoints, double one_eps, double two_eps, double inf_eps);
void checkNorm(DenseMatrix normValues, TestType type, size_t numPoints, double one_eps, double two_eps, double inf_eps);
void _checkNorm(DenseMatrix normValues, int row, size_t numPoints, double one_eps, double two_eps, double inf_eps);

void testApproximation(std::vector<TestFunction *> funcs,
                       std::function<Function *(const DataTable &table)> approx_gen_func,
                       TestType type, size_t numSamplePoints, size_t numEvalPoints,
                       double one_eps, double two_eps, double inf_eps);

} // namespace SPLINTER

#endif // SPLINTER_TESTINGUTILITIES_H
