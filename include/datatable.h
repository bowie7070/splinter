/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_DATATABLE_H
#define SPLINTER_DATATABLE_H

#include <set>
#include "datapoint.h"

#include <ostream>

namespace SPLINTER
{

template <class T>
struct supports_duplicates;

template <class... Ts>
struct supports_duplicates<std::set<Ts...>> : std::false_type {};

template <class... Ts>
struct supports_duplicates<std::multiset<Ts...>> : std::true_type {};

template <class T>
constexpr bool supports_duplicates_v = supports_duplicates<T>::value;


/*
 * DataTable is a class for storing multidimensional data samples (x,y).
 * The samples are stored in a continuously sorted table.
 */

template <class samples_type = std::multiset<DataPoint>>
class SPLINTER_API _data_table
{
public:
    static constexpr bool allowDuplicates = supports_duplicates_v<samples_type>;    

    using data_point = typename samples_type::value_type;

    /*
     * Functions for adding a sample (x,y)
     */
    void addSample(const data_point&sample)
    {
        if (getNumSamples() == 0)
        {
            numVariables = sample.getDimX();
        }

        if(sample.getDimX() != numVariables) {
            throw Exception("Datatable::addSample: Dimension of new sample is inconsistent with previous samples!");
        }

        // Check if the sample has been added already
        if (samples.count(sample) > 0)
        {
            if (!allowDuplicates)
            {
                return;
            }
        }

        samples.insert(sample);
    }
    
    template <class x_type>
    void addSample(x_type const& x, double y) 
    {
        addSample(data_point(x, y));
    }

    auto const& csamples() const { return samples; }

    auto cbegin() const
    {
        return samples.cbegin();
    }

    auto cend() const
    {
        return samples.cend();
    }

    unsigned int getNumVariables() const {return numVariables;}
    unsigned int getNumSamples() const {return samples.size();}
    const samples_type& getSamples() const {return samples;}

    /*
    * Get table of samples x-values,
    * i.e. table[i][j] is the value of variable i at sample j
    */
    std::vector< std::vector<double> > _getTableX() const
    {
        // Initialize table
        std::vector<std::vector<double>> table;
        for (unsigned int i = 0; i < numVariables; i++)
        {
            std::vector<double> xi(getNumSamples(), 0.0);
            table.push_back(xi);
        }

        // Fill table with values
        int i = 0;
        for (auto &sample : samples)
        {
            for (unsigned int j = 0; j < numVariables; j++)
            {
                table[j][i] = sample.x[j];
            }
            i++;
        }

        return table;
    }

    // Get vector of y-values
    std::vector<double> getVectorY() const
    {
        std::vector<double> y;
        for (auto it = cbegin(); it != cend(); ++it)
        {
            y.push_back(it->y);
        }
        return y;
    }      

private:
    unsigned int numVariables = 0;

    samples_type samples;
};

template <class x_type>
using data_table_multiset_x = _data_table<std::multiset<_data_point<x_type>>>;

template <class x_type>
using data_table_set_x = _data_table<std::set<_data_point<x_type>>>;


using DataTable = _data_table<>;

} // namespace SPLINTER

#endif // SPLINTER_DATATABLE_H
