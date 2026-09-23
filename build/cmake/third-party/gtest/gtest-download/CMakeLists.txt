# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(google-test-download NONE)

include(ExternalProject)
ExternalProject_Add(gtest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           release-1.11.0
  SOURCE_DIR        "/mnt/d/Unicamp/Mestrado/salsaarm/build/cmake/third-party/gtest/gtest-src"
  BINARY_DIR        "/mnt/d/Unicamp/Mestrado/salsaarm/build/cmake/third-party/gtest/gtest-build"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
  UPDATE_COMMAND    ""
)
