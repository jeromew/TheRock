# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install the CTS conformance CSV files alongside the test binaries so they
# are included in the artifact. The binaries land in
# ${CMAKE_INSTALL_BINDIR}/$<CONFIG> (i.e. share/opencl/opencl-cts/Release/);
# CMAKE_BUILD_TYPE=Release is fixed for this subproject so we use it directly.
file(
  GLOB _cts_csv_files
  "${CMAKE_CURRENT_SOURCE_DIR}/test_conformance/opencl_conformance_tests_*.csv"
)
install(
  FILES ${_cts_csv_files}
  DESTINATION "${CMAKE_INSTALL_BINDIR}/Release"
)
