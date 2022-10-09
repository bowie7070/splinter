/*
 * This file is part of the SPLINTER library.
 * Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SPLINTER_SERIALIZER_H
#define SPLINTER_SERIALIZER_H

#include "definitions.h"
#include <iostream>
#include <set>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

namespace SPLINTER {

class DataPoint;
class DataTable;
class BSpline;
class BSplineBasis;
class BSplineBasis1D;

/**
 * Class for serialization
 * NOTE: member variables must be serialized and deserialized in the same order.
 * NOTE2: Serialization / deserialization of SparseMatrix/SparseVector works by first converting to
 * DenseMatrix/DenseVector and vice versa.
 */
class Serializer {
public:
    Serializer();
    Serializer(std::string const& fileName);

    virtual ~Serializer() {}

    // Serialize obj into the internal stream
    template <class T>
    void serialize(T const& obj);

    template <class T>
    void deserialize(T& obj);

    template <class T>
    void deserialize(std::vector<T>& obj);

    template <class T>
    void deserialize(std::set<T>& obj);

    template <class T>
    void deserialize(std::multiset<T>& obj);

    void deserialize(DenseMatrix& obj);
    void deserialize(DenseVector& obj);
    void deserialize(SparseMatrix& obj);
    void deserialize(SparseVector& obj);

    void deserialize(DataPoint& obj);
    void deserialize(DataTable& obj);
    void deserialize(BSpline& obj);
    void deserialize(BSplineBasis& obj);
    void deserialize(BSplineBasis1D& obj);

    // Save the serialized stream to fileName
    void saveToFile(std::string const& fileName);

    // Load fileName into the internal stream
    void loadFromFile(std::string const& fileName);

    template <class T>
    static size_t get_size(T const& obj);

    template <class T>
    static size_t get_size(std::vector<T> const& obj);

    template <class T>
    static size_t get_size(std::set<T> const& obj);

    template <class T>
    static size_t get_size(std::multiset<T> const& obj);

    static size_t get_size(DenseMatrix const& obj);
    static size_t get_size(DenseVector const& obj);
    static size_t get_size(SparseMatrix const& obj);
    static size_t get_size(SparseVector const& obj);

    static size_t get_size(DataPoint const& obj);
    static size_t get_size(DataTable const& obj);
    static size_t get_size(BSpline const& obj);
    static size_t get_size(BSplineBasis const& obj);
    static size_t get_size(BSplineBasis1D const& obj);

protected:
    template <class T>
    void _serialize(T const& obj);

    template <class T>
    void _serialize(std::vector<T> const& obj);

    template <class T>
    void _serialize(std::set<T> const& obj);

    template <class T>
    void _serialize(std::multiset<T> const& obj);

    void _serialize(DenseMatrix const& obj);
    void _serialize(DenseVector const& obj);
    void _serialize(SparseMatrix const& obj);
    void _serialize(SparseVector const& obj);

    void _serialize(DataPoint const& obj);
    void _serialize(DataTable const& obj);
    void _serialize(BSpline const& obj);
    void _serialize(BSplineBasis const& obj);
    void _serialize(BSplineBasis1D const& obj);

    typedef std::vector<uint8_t> StreamType;
    StreamType stream;

    // Where we are when serializing
    StreamType::iterator write;

    // Where we are when deserializing
    StreamType::const_iterator read;
};

template <class T>
void Serializer::serialize(T const& obj) {
    // We can't set write to stream.end() here because the call
    // to stream.resize() below may invalidate iterators
    int writeIndex = stream.size();

    // Increase the size of the stream so it can hold the object
    stream.resize(stream.size() + get_size(obj));

    write = stream.begin() + writeIndex;

    _serialize(obj);
}

template <class T>
void Serializer::_serialize(T const& obj) {
    // This should really be used to avoid simply copying complex objects that are not trivially copyable
    // std::is_trivially_copyable is shipped with GCC 5
    //    static_assert(std::is_trivially_copyable<T>::value, "Missing Serializer::_serialize overload for T = "/* __PRETTY_FUNCTION__*/);

    // Get a uint8_t pointer to the object, so we can copy it into the stream
    auto objPtr = reinterpret_cast<uint8_t const*>(&obj);

    std::copy(objPtr, objPtr + sizeof(T), write);

    write += sizeof(T);
}

template <class T>
void Serializer::deserialize(T& obj) {
    // This should really be used to avoid simply copying complex objects that are not trivially copyable
    // std::is_trivially_copyable is shipped with GCC 5
    //    static_assert(std::is_trivially_copyable<T>::value, "Missing Serializer::deserialize overload for T = "/* __PRETTY_FUNCTION__*/);

    if (read + sizeof(T) > stream.cend()) {
        throw Exception("Serializer::deserialize: Stream is missing bytes!");
    }

    auto objPtr = reinterpret_cast<uint8_t*>(&obj);

    // Copy the data into val
    std::copy(read, read + sizeof(T), objPtr);

    read += sizeof(T);
}

template <class T>
size_t Serializer::get_size(T const& obj) {
    // This should really be used to avoid simply copying complex objects that are not trivially copyable
    // std::is_trivially_copyable is shipped with GCC 5
    //    static_assert(std::is_trivially_copyable<T>::value, "Missing Serializer::get_size overload for T = "/* __PRETTY_FUNCTION__*/);

    return sizeof(T);
}

/*
 * get_size specializations
 */
template <class T>
size_t Serializer::get_size(std::vector<T> const& obj) {
    size_t size = sizeof(size_t);
    for (auto& elem : obj) {
        size += get_size(elem);
    }

    return size;
}

template <class T>
size_t Serializer::get_size(std::set<T> const& obj) {
    size_t size = sizeof(size_t);
    for (auto& elem : obj) {
        size += get_size(elem);
    }

    return size;
}

template <class T>
size_t Serializer::get_size(std::multiset<T> const& obj) {
    size_t size = sizeof(size_t);
    for (auto& elem : obj) {
        size += get_size(elem);
    }

    return size;
}

/*
 * _serialize specializations
 */
template <class T>
void Serializer::_serialize(std::vector<T> const& obj) {
    _serialize(obj.size());
    for (auto& elem : obj) {
        _serialize(elem);
    }
}

template <class T>
void Serializer::_serialize(std::set<T> const& obj) {
    _serialize(obj.size());
    for (auto& elem : obj) {
        _serialize(elem);
    }
}

template <class T>
void Serializer::_serialize(std::multiset<T> const& obj) {
    _serialize(obj.size());
    for (auto& elem : obj) {
        _serialize(elem);
    }
}

/*
 * deserialize specializations
 */
template <class T>
void Serializer::deserialize(std::vector<T>& obj) {
    size_t size;
    deserialize(size);
    obj.resize(size);

    for (auto& elem : obj) {
        deserialize(elem);
    }
}

template <class T>
void Serializer::deserialize(std::set<T>& obj) {
    size_t size;
    deserialize(size);

    T elem;
    for (size_t i = 0; i < size; ++i) {
        deserialize(elem);
        obj.insert(elem);
    }
}

template <class T>
void Serializer::deserialize(std::multiset<T>& obj) {
    size_t size;
    deserialize(size);

    T elem;
    for (size_t i = 0; i < size; ++i) {
        deserialize(elem);
        obj.insert(elem);
    }
}

} // namespace SPLINTER

#endif // SPLINTER_SERIALIZER_H
