# This file is part of the SPLINTER library.
# Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.21)

project(SPLINTER CXX)

# Read the SPLINTER version from version file
file(STRINGS "version" VERSION)

set(CMAKE_CXX_STANDARD 17)


find_package(Catch2 CONFIG)
#find_package(Eigen3 CONFIG REQUIRED HINTS path-to-eigen)
find_package(Eigen3 CONFIG REQUIRED)


add_library(splinter)
target_sources(splinter PRIVATE
    include/bsplinebasis1d.h
    include/bsplinebasis.h
    include/bsplinebuilder.h
    include/bspline.h
    include/datapoint.h
    include/datatable.h
    include/definitions.h
    include/function.h
    include/knots.h
    include/linearsolvers.h
    include/mykroneckerproduct.h
    include/splinter.h
    include/utilities.h
    src/bsplinebasis1d.cpp
    src/bsplinebasis.cpp
    src/bsplinebuilder.cpp
    src/bspline.cpp
    src/datapoint.cpp
    src/datatable.cpp
    src/function.cpp
    src/knots.cpp
    src/mykroneckerproduct.cpp
    src/utilities.cpp
)

target_include_directories(splinter PUBLIC include)
target_link_libraries(splinter PUBLIC
    Eigen3::Eigen
)

option(SPLINTER_ENABLE_TESTING OFF "Enable testing")
if(SPLINTER_ENABLE_TESTING)
    # Testing executable
    add_executable(splinter-test)
    target_sources(splinter-test PRIVATE
        test/approximation/bspline.cpp
        test/approximation/pspline.cpp
        test/bsplinetestingutilities.cpp
        test/bsplinetestingutilities.h
        test/general/bspline.cpp
        test/general/datatable.cpp
        test/main.cpp
        test/testfunction.cpp
        test/testfunction.h
        test/testfunctions.cpp
        test/testfunctions.h
        test/testingutilities.cpp
        test/testingutilities.h
        test/unit/bsplinebasis1d.cpp
        test/unit/knots.cpp
    )

    include(Catch)

    target_include_directories(splinter-test PRIVATE
        test
    )
    target_link_libraries(splinter-test PRIVATE
        splinter
        Catch2::Catch2
    )

    catch_discover_tests(splinter-test)
endif()

# Install the shared library file
install(
    TARGETS splinter
    DESTINATION ${LIBRARY_INSTALL_DIRECTORY}
)
