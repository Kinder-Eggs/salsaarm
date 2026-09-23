# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(cpu-features-download NONE)

include(ExternalProject)
ExternalProject_Add(cpu_features
  GIT_REPOSITORY    https://github.com/google/cpu_features.git
  GIT_TAG           32b49eb5e7809052a28422cfde2f2745fbb0eb76 # master branch on Oct 20, 2021
  SOURCE_DIR        "/mnt/d/Unicamp/Mestrado/salsaarm/build/cmake/third-party/cpu-features/cpu-features-src"
  BINARY_DIR        "/mnt/d/Unicamp/Mestrado/salsaarm/build/cmake/third-party/cpu-features/cpu-features-build"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
  UPDATE_COMMAND    ""
)
