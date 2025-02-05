/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_FUNCTION_H
#define SPLINTER_FUNCTION_H

#include "definitions.h"

namespace SPLINTER {

template <class callable>
auto centralDifference(callable const& f, DenseVector const& x) {
    auto const I = x.size();

    DenseVector dx(I);

    double h         = 1e-6; // perturbation step size
    double hForward  = 0.5 * h;
    double hBackward = 0.5 * h;

    for (unsigned int i = 0; i < I; ++i) {
        DenseVector xForward(x);
        xForward(i) = xForward(i) + hForward;

        DenseVector xBackward(x);
        xBackward(i) = xBackward(i) - hBackward;

        // todo: use function operator
        double yForward  = f.eval(xForward);
        double yBackward = f.eval(xBackward);

        dx[i] = (yForward - yBackward) / (hBackward + hForward);
    }

    return dx;
}

/*
 * Interface for functions
 * All functions working with standard C++11 types are defined in terms of their Eigen counterparts.
 * Default implementations of jacobian and hessian evaluation is using central difference.
 */
class SPLINTER_API Function {
public:
    Function() : Function(1) {}

    Function(unsigned int numVariables) : numVariables(numVariables) {}

    virtual ~Function() {}

    /**
     * Returns the function value at x
     */
    virtual double eval(DenseVector x) const = 0;

    /**
     * Returns the function value at x
     */
    double eval(std::vector<double> const& x) const;

    /**
     * Returns the (1 x numVariables) Jacobian evaluated at x
     */
    virtual DenseMatrix evalJacobian(DenseVector x) const;

    /**
     * Returns the (1 x numVariables) Jacobian evaluated at x
     */
    std::vector<double> evalJacobian(std::vector<double> const& x) const;

    /**
     * Returns the (numVariables x numVariables) Hessian evaluated at x
     */
    virtual DenseMatrix evalHessian(DenseVector x) const;

    /**
     * Returns the (numVariables x numVariables) Hessian evaluated at x
     */
    std::vector<std::vector<double>>
    evalHessian(std::vector<double> const& x) const;

    /**
     * Get the dimension
     */
    inline unsigned int getNumVariables() const { return numVariables; }

    /**
     * Check input
     */
    void checkInput(DenseVector x) const {
        if (x.size() != numVariables)
            throw Exception(
                "Function::checkInput: Wrong dimension on evaluation point x.");
    }

    /**
     * Returns the central difference at x
     * Vector of numVariables length
     */
    std::vector<double> centralDifference(std::vector<double> const& x) const;
    DenseMatrix centralDifference(DenseVector x) const {
        return SPLINTER::centralDifference(*this, x);
    }

    std::vector<std::vector<double>>
    secondOrderCentralDifference(std::vector<double> const& x) const;
    DenseMatrix secondOrderCentralDifference(DenseVector x) const;

    /**
     * Description of function.
     */
    virtual std::string getDescription() const { return ""; }

protected:
    unsigned int numVariables; // Dimension of domain (size of x)
};

} // namespace SPLINTER

#endif // SPLINTER_FUNCTION_H
