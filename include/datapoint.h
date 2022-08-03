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

template <class x_type>
struct _data_point
{
	_data_point(double x, double y)
		: x{ x },
		y(y)
	{
	}

	_data_point(x_type x, double y)
		: x(std::move(x)),
		y(y)
	{
	}

	bool operator<(const _data_point& rhs) const
	{
		if (this->getDimX() != rhs.getDimX())
			throw Exception("DataPoint::operator<: Cannot compare data points of different dimensions");

		if constexpr (!std::is_same_v<x_type, DenseVector>) {
			return x < rhs.x;
		}
		else {
			return std::lexicographical_compare(x.begin(), x.end(), rhs.x.begin(), rhs.x.end());
		}
	}

	x_type x;
	double y;

	unsigned int getDimX() const { return x.size(); }
};

using DataPoint = _data_point<std::vector<double>>;

} // namespace SPLINTER

#endif // SPLINTER_DATAPOINT_H
