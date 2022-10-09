/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_KNOTS_H
#define SPLINTER_KNOTS_H

#include <vector>

namespace SPLINTER {

// Knot vector related
bool isKnotVectorRegular(std::vector<double> const& knots, unsigned int degree);
bool isKnotVectorClamped(std::vector<double> const& knots, unsigned int degree);
bool isKnotVectorRefinement(
    std::vector<double> const& knots, std::vector<double> const& refinedKnots);

} // namespace SPLINTER

#endif // SPLINTER_KNOTS_H
