/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_DATAPOINT_H
#define SPLINTER_DATAPOINT_H

#include "definitions.h"

namespace SPLINTER
{

/*
 * DataPoint is a class representing a data point (x, y),
 * where y is the value obtained by sampling at a point x.
 * Note that x is a vector and y is a scalar.
 */
class DataPoint
{
    using x_type = std::vector<double>;

    static x_type as_x_type(DenseVector const& x)
    {
        std::vector<double> newX;

        for (int i = 0; i < x.size(); i++)
        {
            newX.push_back(x(i));
        }

        return newX;
    }

public:
    DataPoint(double x, double y)
    :   x(1, x),
        y(y)
    {
    }

    DataPoint(std::vector<double> x, double y)
    :   x(x),
        y(y)
    {
    }

    DataPoint(DenseVector const& x, double y)
    :   x(as_x_type(x)),
        y(y)
    {
    }

    bool operator<(const DataPoint &rhs) const
    {
        if (this->getDimX() != rhs.getDimX())
            throw Exception("DataPoint::operator<: Cannot compare data points of different dimensions");

        return x < rhs.x;
    }

    std::vector<double> x;
    double y;

    unsigned int getDimX() const { return x.size(); }
};

/*
* Computes Euclidean distance ||x-y||
*/
inline double dist(const std::vector<double> x, const std::vector<double> y)
{
    if (x.size() != y.size())
        throw Exception("DataPoint::dist: Cannot measure distance between two points of different dimension");
    double sum = 0.0;
    for (unsigned int i=0; i<x.size(); i++)
        sum += (x.at(i)-y.at(i))*(x.at(i)-y.at(i));
    return std::sqrt(sum);
}

/*
* Computes Euclidean distance ||x-y||
*/
inline double dist(const DataPoint& x, const DataPoint& y)
{
    return dist(x.x, y.x);
}

inline bool dist_sort(const DataPoint x, const DataPoint y)
{
    std::vector<double> zeros(x.getDimX(), 0);
    DataPoint origin(zeros, 0.0);
    double x_dist = dist(x, origin);
    double y_dist = dist(y, origin);
    return (x_dist<y_dist);
}


} // namespace SPLINTER

#endif // SPLINTER_DATAPOINT_H
