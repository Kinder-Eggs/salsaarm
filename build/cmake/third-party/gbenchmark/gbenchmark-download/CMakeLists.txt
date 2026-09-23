# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(google-benchmark-download NONE)

include(ExternalProject)
ExternalProject_Add(gbenchmark
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           f730846b0a3c0dc0699978846fb14ffb2fad0bdc # main branch on Oct 18, 2021
  SOURCE_DIR        "/mnt/d/Unicamp/Mestrado/salsaarm/build/cmake/third-party/gbenchmark/gbenchmark-src"
  BINARY_DIR        "/mnt/d/Unicamp/Mestrado/salsaarm/build/cmake/third-party/gbenchmark/gbenchmark-build"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
  UPDATE_COMMAND    ""
)
