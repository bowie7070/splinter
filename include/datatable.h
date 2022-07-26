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

/*
 * DataTable is a class for storing multidimensional data samples (x,y).
 * The samples are stored in a continuously sorted table.
 */
class SPLINTER_API DataTable
{
public:
    DataTable(bool allowDuplicates = false, bool allowIncompleteGrid = false)
    : allowDuplicates(allowDuplicates),
      allowIncompleteGrid(allowIncompleteGrid),
      numDuplicates(0),
      numVariables(0)
    {
    }

    /*
     * Functions for adding a sample (x,y)
     */
    void addSample(const DataPoint &sample)
    {
        if (getNumSamples() == 0)
        {
            numVariables = sample.getDimX();
            initDataStructures();
        }

        if(sample.getDimX() != numVariables) {
            throw Exception("Datatable::addSample: Dimension of new sample is inconsistent with previous samples!");
        }

        // Check if the sample has been added already
        if (samples.count(sample) > 0)
        {
            if (!allowDuplicates)
            {
    #ifndef NDEBUG
                std::cout << "Discarding duplicate sample because allowDuplicates is false!" << std::endl;
                std::cout << "Initialise with DataTable(true) to set it to true." << std::endl;
    #endif // NDEBUG

                return;
            }

            numDuplicates++;
        }

        samples.insert(sample);

        recordGridPoint(sample);
    }
    
    template <class x_type>
    void addSample(x_type const& x, double y) 
    {
        addSample(DataPoint(x, y));
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
    const std::multiset<DataPoint>& getSamples() const {return samples;}

    std::vector<std::set<double>> getGrid() const { return grid; }

    /*
    * Get table of samples x-values,
    * i.e. table[i][j] is the value of variable i at sample j
    */
    std::vector< std::vector<double> > getTableX() const
    {
        gridCompleteGuard();

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
            std::vector<double> x = sample.x;

            for (unsigned int j = 0; j < numVariables; j++)
            {
                table[j][i] = x[j];
            }
            i++;
        }

        return table;
    }

    // Get vector of y-values
    std::vector<double> getVectorY() const
    {
        std::vector<double> y;
        for (std::multiset<DataPoint>::const_iterator it = cbegin(); it != cend(); ++it)
        {
            y.push_back(it->y);
        }
        return y;
    }
    
    bool isGridComplete() const
    {
        return samples.size() > 0 && samples.size() - numDuplicates == getNumSamplesRequired();
    }


private:
    bool allowDuplicates;
    bool allowIncompleteGrid;
    unsigned int numDuplicates;
    unsigned int numVariables;

    std::multiset<DataPoint> samples;
    std::vector< std::set<double> > grid;

    // Initialise grid to be a std::vector of xDim std::sets
    void initDataStructures()
    {
        for (unsigned int i = 0; i < getNumVariables(); i++)
        {
            grid.push_back(std::set<double>());
        }
    }

    unsigned int getNumSamplesRequired() const
    {
        unsigned long samplesRequired = 1;
        unsigned int i = 0;
        for (auto &variable : grid)
        {
            samplesRequired *= (unsigned long) variable.size();
            i++;
        }

        return (i > 0 ? samplesRequired : (unsigned long) 0);
    }


    void recordGridPoint(const DataPoint &sample)
    {
        for (unsigned int i = 0; i < getNumVariables(); i++)
        {
            grid[i].insert(sample.x[i]);
        }
    }

    // Used by functions that require the grid to be complete before they start their operation
    // This function prints a message and exits the program if the grid is not complete.
    void gridCompleteGuard() const
    {
        if (!(isGridComplete() || allowIncompleteGrid))
        {
            throw Exception("DataTable::gridCompleteGuard: The grid is not complete yet!");
        }
    }


    friend bool operator==(const DataTable &lhs, const DataTable &rhs);
};

} // namespace SPLINTER

#endif // SPLINTER_DATATABLE_H
