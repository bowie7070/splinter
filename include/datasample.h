#ifndef DATASAMPLE_H
#define DATASAMPLE_H

#include "include/generaldefinitions.h"

/* Class representing a data sample (x,y)
 * where y is the value obtained by sampling
 * at a point x. Both x and y may be scalar
 * or vectors.
*/
class DataSample
{
public:
    DataSample(double x, double y);
    DataSample(std::vector<double> x, double y);
    DataSample(std::vector<double> x, std::vector<double> y);
    DataSample(DenseVector x, double y);
    DataSample(DenseVector x, DenseVector y);

    bool operator<(const DataSample &rhs) const; // Returns false if the two are equal
    friend std::ostream &operator<<(std::ostream &outputStream, const DataSample &sample);

    std::vector<double> getX() const { return x; }
    std::vector<double> getY() const { return y; }
    unsigned int getDimX() const { return x.size(); }
    unsigned int getDimY() const { return y.size(); }

private:
    std::vector<double> x;
    std::vector<double> y;

    void setData(const std::vector<double> &x, const std::vector<double> &y);
};

#endif // DATASAMPLE_H
