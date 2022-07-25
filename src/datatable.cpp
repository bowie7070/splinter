/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "datatable.h"
#include <string>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include <initializer_list>

namespace SPLINTER
{


/*
 * Getters for iterators
 */



DataTable operator+(const DataTable &lhs, const DataTable &rhs)
{
    if(lhs.getNumVariables() != rhs.getNumVariables()) {
        throw Exception("operator+(DataTable, DataTable): trying to add two DataTable's of different dimensions!");
    }

    DataTable result;
    for(auto it = lhs.cbegin(); it != lhs.cend(); it++) {
        result.addSample(*it);
    }
    for(auto it = rhs.cbegin(); it != rhs.cend(); it++) {
        result.addSample(*it);
    }

    return result;
}

DataTable operator-(const DataTable &lhs, const DataTable &rhs)
{
    if(lhs.getNumVariables() != rhs.getNumVariables()) {
        throw Exception("operator-(DataTable, DataTable): trying to subtract two DataTable's of different dimensions!");
    }

    DataTable result;
    auto rhsSamples = rhs.getSamples();
    // Add all samples from lhs that are not in rhs
    for(auto it = lhs.cbegin(); it != lhs.cend(); it++) {
        if(rhsSamples.count(*it) == 0) {
            result.addSample(*it);
        }
    }

    return result;
}

} // namespace SPLINTER
