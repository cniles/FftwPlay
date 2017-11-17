cmake_minimum_required(VERSION 3.0.0)
project(FftwPlay VERSION 0.0.0)

include(CTest)
enable_testing()

add_executable(FftwPlay main.cpp)
target_link_libraries(FftwPlay fftw3)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
