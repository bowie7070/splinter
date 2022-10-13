/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "testingutilities.h"
#include <utilities.h>
#include <testfunctions.h>
#include <bsplinebuilder.h>

using namespace std;

namespace SPLINTER
{

// See https://en.wikipedia.org/wiki/Relative_change_and_difference#Formulae
double getError(double exactVal, double approxVal)
{
    double maxAbsVal = std::max(std::abs(exactVal), std::abs(approxVal));

    // Both are ~0
    if(maxAbsVal < 1e-14)
    {
        return 0.0;
    }

    double absError = std::abs(exactVal - approxVal);

    return std::min(absError, absError / maxAbsVal);
}

// Checks if a is within margin of b
bool equalsWithinRange(double a, double b, double margin)
{
    return b - margin <= a && a <= b + margin;
}








double sixHumpCamelBack(DenseVector x)
{
    assert(x.rows() == 2);
    return (4 - 2.1*x(0)*x(0) + (1/3.)*x(0)*x(0)*x(0)*x(0))*x(0)*x(0) + x(0)*x(1) + (-4 + 4*x(1)*x(1))*x(1)*x(1);
}

double getOneNorm(const DenseMatrix &m)
{
    return m.lpNorm<1>();
    double norm = 0.0;
    for(int i = 0; i < m.rows(); i++) {
        for(int j = 0; j < m.cols(); j++) {
            norm += std::abs(m(i, j));
        }
    }

    return norm;
}

double getTwoNorm(const DenseMatrix &m)
{
    return m.lpNorm<2>();
    double norm = 0.0;
    for(int i = 0; i < m.rows(); i++) {
        for(int j = 0; j < m.cols(); j++) {
            norm += std::pow(m(i, j), 2);
        }
    }

    norm = std::sqrt(norm);

    return norm;
}

double getInfNorm(const DenseMatrix &m)
{
    return m.lpNorm<Eigen::Infinity>();
    double norm = 0.0;
    for(int i = 0; i < m.rows(); i++) {
        for(int j = 0; j < m.cols(); j++) {
            norm = std::max(std::abs(m(i, j)), norm);
        }
    }

    return norm;
}

double log(double base, double x)
{
    return std::log(x) / std::log(base);
}

// Enumerates all permutations
std::vector<std::vector<double>> linspace(std::vector<double> start, std::vector<double> end, std::vector<unsigned int> points)
{
    assert(start.size() == end.size() && end.size() == points.size());

    size_t nDims = start.size();
    size_t numPoints = 1;
    for(size_t i = 0; i < nDims; i++) {
        numPoints *= points.at(i);
    }

#ifndef NDEBUG
    if(numPoints > 10000) {
        std::cout << "Warning: Enumerating " << numPoints << " points." << std::endl;
    }
#endif // ifndef NDEBUG

    auto result = std::vector<std::vector<double>>(numPoints);

    auto dx = std::vector<double>(nDims);
    auto it = std::vector<unsigned int>(nDims);
    auto cur = std::vector<double>(nDims);

    for(unsigned int i = 0; i < nDims; i++) {
        cur.at(i) = start.at(i);
        it.at(i) = 0;
        dx.at(i) = (end.at(i) - start.at(i)) / (points.at(i) - 1);
    }

    size_t curDim = 0;
    size_t i = 0;
    // Add the start vector
    result.at(i++) = std::vector<double>(start);
    while(true) {
        curDim = 0;
        while(curDim < nDims && it.at(curDim)+1 >= points.at(curDim)) {
            it.at(curDim) = 0;
            cur.at(curDim) = start.at(curDim) + dx.at(curDim) * it.at(curDim);
            curDim++;
        }
        // If we're unable to find a number that can be increased,
        // it means were at the highest number
        // (for 4 digits in decimal that is 9999)
        if(curDim >= nDims) {
            break;
        }

        it.at(curDim)++;
        cur.at(curDim) = start.at(curDim) + dx.at(curDim) * it.at(curDim);

        result.at(i++) = std::vector<double>(cur);
    }

    assert(i == numPoints);

    return result;
}

std::vector<std::vector<double>> linspace(int dim, double start, double end, unsigned int points)
{
    auto startVec = std::vector<double>(dim);
    auto endVec = std::vector<double>(dim);
    auto pointsVec = std::vector<unsigned int>(dim);

    for(int i = 0; i < dim; i++) {
        startVec.at(i) = start;
        endVec.at(i) = end;
        pointsVec.at(i) = points;
    }

    return linspace(startVec, endVec, pointsVec);
}

std::vector<std::vector<double>> linspace(int dim)
{
    // Total number of points to evaluate
    // Will be distributed evenly per dimension
    const int totSamples = 10000;

    // Take the dim'th root to get number of samples per dimension
    unsigned int pointsPerDim = std::pow(totSamples, 1.0/dim);

    // Clamp so that pointsPerDim >= 2
    if(pointsPerDim < 2) {
        pointsPerDim = 2;
    }

    return linspace(dim, pointsPerDim);
}

std::vector<std::vector<double>> linspace(int dim, unsigned int pointsPerDim)
{
    auto start = std::vector<double>(dim);
    auto end = std::vector<double>(dim);
    auto numPoints = std::vector<unsigned int>(dim);


    for(int i = 0; i < dim; i++) {

        start.at(i) = -10.0;
        end.at(i) = 10.0;
        numPoints.at(i) = pointsPerDim;
    }

    return linspace(start, end, numPoints);
}


std::vector<double> denseToVec(const DenseVector &dense)
{
    auto vec = std::vector<double>(dense.size());
    for(int i = 0; i < (int) dense.size(); i++) {
        vec.at(i) = dense(i);
    }

    return vec;
}

DenseVector vecToDense(const std::vector<double> &vec)
{
    DenseVector dense(vec.size());
    for(int i = 0; i < (int) vec.size(); i++) {
        dense(i) = vec.at(i);
    }

    return dense;
}

std::string pretty_print(const DenseVector &denseVec)
{
    std::string str("[");
    for(int i = 0; i < denseVec.rows(); i++) {
        str += to_string(denseVec(i));
        if(i + 1 < denseVec.rows()) {
            str += "; ";
        }
    }
    str += "]";

    return str;
}


TestFunction *getTestFunction(int numVariables, int degree)
{
    return testFunctions.at(numVariables).at(degree);
}

std::vector<TestFunction *> getTestFunctionsOfDegree(int degree)
{
    auto testFuncs = std::vector<TestFunction *>();
    for(int i = 1; i < (int) testFunctions.size(); ++i) {
        if(degree < (int) testFunctions.at(i).size()) {
            testFuncs.push_back(testFunctions.at(i).at(degree));
        }
    }
    return testFuncs;
}

std::vector<TestFunction *> getTestFunctionWithNumVariables(int numVariables)
{
    return testFunctions.at(numVariables);
}

std::vector<TestFunction *> getPolynomialFunctions()
{
    auto testFuncs = std::vector<TestFunction *>();
    for(int i = 1; i < (int) testFunctions.size(); ++i) {
        for(int j = 0; j < (int) testFunctions.at(i).size(); ++j) {
            testFuncs.push_back(testFunctions.at(i).at(j));
        }
    }
    return testFuncs;
}

std::vector<TestFunction *> getNastyTestFunctions()
{
    return testFunctions.at(0);
}

DenseMatrix getErrorNorms(const Function *exact, const Function *approx, const std::vector<std::vector<double>> &points)
{
    assert(exact->getNumVariables() == approx->getNumVariables());

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
    for (auto &point : points) {
        DenseVector x = vecToDense(point);

        {
            DenseMatrix exactValue(1,1);
            exactValue(0,0) = exact->eval(x);
            DenseMatrix approxValue(1,1);
            approxValue(0,0) = approx->eval(x);
            DenseMatrix error = exactValue - approxValue;

            normOneValVec(i) = getOneNorm(error);
            normTwoValVec(i) = getTwoNorm(error);
            normInfValVec(i) = getInfNorm(error);
        }

        {
            auto exactJacobian = exact->evalJacobian(x);
            auto approxJacobian = approx->evalJacobian(x);
            auto errorJacobian = exactJacobian - approxJacobian;

            normOneJacVec(i) = getOneNorm(errorJacobian);
            normTwoJacVec(i) = getTwoNorm(errorJacobian);
            normInfJacVec(i) = getInfNorm(errorJacobian);
        }

        {
            auto exactHessian = exact->evalHessian(x);
            auto approxHessian = approx->evalHessian(x);
            auto errorHessian = exactHessian - approxHessian;

            normOneHesVec(i) = getOneNorm(errorHessian);
            normTwoHesVec(i) = getTwoNorm(errorHessian);
            normInfHesVec(i) = getInfNorm(errorHessian);
        }

        i++;
    }

    DenseMatrix errorNorms(3,3);
    errorNorms(0,0) = getOneNorm(normOneValVec);
    errorNorms(0,1) = getTwoNorm(normTwoValVec);
    errorNorms(0,2) = getInfNorm(normInfValVec);

    errorNorms(1,0) = getOneNorm(normOneJacVec);
    errorNorms(1,1) = getTwoNorm(normTwoJacVec);
    errorNorms(1,2) = getInfNorm(normInfJacVec);

    errorNorms(2,0) = getOneNorm(normOneHesVec);
    errorNorms(2,1) = getTwoNorm(normTwoHesVec);
    errorNorms(2,2) = getInfNorm(normInfHesVec);

    return errorNorms;
}

void checkNorms(DenseMatrix normValues, size_t numPoints, double one_eps, double two_eps, double inf_eps)
{
    checkNorm(normValues, TestType::All, numPoints, one_eps, two_eps, inf_eps);
}

void checkNorm(DenseMatrix normValues, TestType type, size_t numPoints, double one_eps, double two_eps, double inf_eps)
{
    if(type == TestType::All || type == TestType::FunctionValue) {
        INFO("Function value error:");
        _checkNorm(normValues, 0, numPoints, one_eps, two_eps, inf_eps);
    }
    if(type == TestType::All || type == TestType::Jacobian) {
        INFO("Jacobian value error:");
        _checkNorm(normValues, 1, numPoints, one_eps, two_eps, inf_eps);
    }
    if(type == TestType::All || type == TestType::Hessian) {
        INFO("Hessian value error:");
        _checkNorm(normValues, 2, numPoints, one_eps, two_eps, inf_eps);
    }
}

void _checkNorm(DenseMatrix normValues, int row, size_t numPoints, double one_eps, double two_eps, double inf_eps)
{
    bool withinThreshold =\
            normValues(row,0) / numPoints <= one_eps
         && normValues(row,1)             <= two_eps
         && normValues(row,2)             <= inf_eps;

    INFO(std::setw(16) << std::left << "1-norm (\"avg\"):" << std::setw(16) << std::right << normValues(row,0) / numPoints);
    INFO(std::setw(16) << std::left << "2-norm:"           << std::setw(16) << std::right << normValues(row,1));
    INFO(std::setw(16) << std::left << "inf-norm:"         << std::setw(16) << std::right << normValues(row,2));

    CHECK(withinThreshold);
}

DenseMatrix centralDifference(const Function &approx, const DenseVector &x)
{
    DenseMatrix dx(1, x.size());

    double h = 1e-6; // perturbation step size
    double hForward = 0.5*h;
    double hBackward = 0.5*h;

    for (unsigned int i = 0; i < approx.getNumVariables(); ++i)
    {
        DenseVector xForward(x);
//        if (xForward(i) + hForward > variables.at(i)->getUpperBound())
//        {
//            hForward = 0;
//        }
//        else
//        {
        xForward(i) = xForward(i) + hForward;
//        }

        DenseVector xBackward(x);
//        if (xBackward(i) - hBackward < variables.at(i)->getLowerBound())
//        {
//            hBackward = 0;
//        }
//        else
//        {
            xBackward(i) = xBackward(i) - hBackward;
//        }

        double yForward = approx.eval(xForward);
        double yBackward = approx.eval(xBackward);

        dx(i) = (yForward - yBackward)/(hBackward + hForward);
    }

    return dx;
}


} // namespace SPLINTER
