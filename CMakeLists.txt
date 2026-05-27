cmake_minimum_required(VERSION 3.18)

project(cs179 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CS179_BUILD_CUDA "Build CUDA kernels and vector_add in _native" ON)
option(CS179_BUILD_PYTHON "Build the pybind11 Python extension" ON)
option(CS179_BUILD_LOADER_TEST "Build C++ loader smoke test (cs179_loader_test)" OFF)
option(CS179_BUILD_REACHABILITY "Build directional reach envelope (requires Pinocchio)" ON)
option(CS179_BUILD_REACHABILITY_TEST "Build C++ reach envelope smoke test (cs179_reach_envelope_test)" OFF)
option(CS179_BUILD_EXAMPLES "Build example programs under examples/" OFF)

if(CS179_BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    add_subdirectory(cuda)
endif()

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

if(CS179_BUILD_PYTHON OR CS179_BUILD_REACHABILITY)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

add_library(cs179_core STATIC
    cpp/src/mmap_file.cpp
    cpp/src/npy.cpp
    cpp/src/rlds_loader.cpp
)
set_target_properties(cs179_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(cs179_core PUBLIC cpp/include)
target_link_libraries(cs179_core PUBLIC nlohmann_json::nlohmann_json)

set(CS179_HAVE_REACHABILITY 0)
if(CS179_BUILD_REACHABILITY)
    if(NOT CS179_CMEEL_PREFIX)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import pathlib, pinocchio; p=pathlib.Path(pinocchio.__file__).resolve();
for parent in p.parents:
    if parent.name == 'cmeel.prefix':
        print(parent)
        break
else:
    raise SystemExit(1)"
            OUTPUT_VARIABLE _cs179_cmeel_prefix
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _cs179_cmeel_status)
        if(_cs179_cmeel_status EQUAL 0)
            set(CS179_CMEEL_PREFIX "${_cs179_cmeel_prefix}")
        endif()
    endif()

    if(CS179_CMEEL_PREFIX)
        list(PREPEND CMAKE_PREFIX_PATH "${CS179_CMEEL_PREFIX}")
        find_package(Eigen3 CONFIG QUIET)
        find_package(Boost CONFIG QUIET)
        find_package(coal CONFIG QUIET)
        if(Eigen3_FOUND AND Boost_FOUND AND coal_FOUND AND EXISTS "${CS179_CMEEL_PREFIX}/lib/cmake/pinocchio/pinocchioTargets.cmake")
            include("${CS179_CMEEL_PREFIX}/lib/cmake/pinocchio/pinocchioTargets.cmake")
            target_sources(cs179_core PRIVATE cpp/src/reach_envelope.cpp)
            target_link_libraries(cs179_core PUBLIC pinocchio::pinocchio_default)
            target_compile_definitions(
                cs179_core
                PUBLIC PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION PINOCCHIO_WITH_COLLISION)
            find_package(OpenMP)
            if(OpenMP_CXX_FOUND)
                target_link_libraries(cs179_core PUBLIC OpenMP::OpenMP_CXX)
            endif()
            FetchContent_Declare(
                nlopt
                GIT_REPOSITORY https://github.com/stevengj/nlopt.git
                GIT_TAG v2.9.1
            )
            set(NLOPT_CXX ON CACHE BOOL "" FORCE)
            set(NLOPT_PYTHON OFF CACHE BOOL "" FORCE)
            set(NLOPT_TESTS OFF CACHE BOOL "" FORCE)
            set(NLOPT_EXAMPLES OFF CACHE BOOL "" FORCE)
            set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
            FetchContent_MakeAvailable(nlopt)

            target_sources(cs179_core PRIVATE cpp/src/retarget.cpp cpp/src/retarget_params.cpp)
            target_link_libraries(cs179_core PUBLIC nlopt)
            set(CS179_HAVE_REACHABILITY 1)
            message(STATUS "Reachability envelope + retarget: enabled (Pinocchio + NLopt + OpenMP)")
        else()
            message(STATUS "Reachability envelope: disabled (Pinocchio deps not found in ${CS179_CMEEL_PREFIX})")
        endif()
    else()
        message(STATUS "Reachability envelope: disabled (cmeel.prefix not found)")
    endif()
endif()

if(CS179_BUILD_LOADER_TEST)
    add_executable(cs179_loader_test cpp/tests/loader_test.cpp)
    target_link_libraries(cs179_loader_test PRIVATE cs179_core)
endif()

if(CS179_BUILD_REACHABILITY_TEST AND CS179_HAVE_REACHABILITY)
    add_executable(cs179_reach_envelope_test cpp/tests/reach_envelope_test.cpp)
    target_link_libraries(cs179_reach_envelope_test PRIVATE cs179_core)
endif()

if(CS179_BUILD_EXAMPLES AND CS179_HAVE_REACHABILITY)
    add_executable(reach_envelope_cpp examples/reach_envelope_cpp.cpp)
    target_link_libraries(reach_envelope_cpp PRIVATE cs179_core)
endif()

if(CS179_BUILD_PYTHON)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sys, pathlib; print(pathlib.Path(sys.base_prefix).resolve())"
        OUTPUT_VARIABLE _cs179_python_root
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(Python3_ROOT_DIR "${_cs179_python_root}")
    set(Python3_FIND_REGISTRY NEVER)
    set(Python3_FIND_FRAMEWORK NEVER)
    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 COMPONENTS Development.Module REQUIRED)

    set(PYBIND11_FINDPYTHON ON)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)

    set(_native_sources cpp/bindings.cpp cpp/bindings_rlds.cpp)
    if(CS179_BUILD_CUDA)
        list(APPEND _native_sources cpp/bindings_cuda.cpp)
    endif()
    if(CS179_HAVE_REACHABILITY)
        list(APPEND _native_sources cpp/bindings_reachability.cpp cpp/bindings_retarget.cpp)
    endif()

    pybind11_add_module(_native ${_native_sources})
    target_link_libraries(_native PRIVATE cs179_core Python3::Module)
    target_include_directories(_native PRIVATE cpp/include)

    if(CS179_CMEEL_PREFIX)
        set_target_properties(_native PROPERTIES BUILD_RPATH "${CS179_CMEEL_PREFIX}/lib")
    endif()

    if(CS179_HAVE_REACHABILITY)
        target_compile_definitions(_native PRIVATE CS179_HAVE_REACHABILITY=1)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
            OUTPUT_VARIABLE _cs179_numpy_include
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        target_include_directories(_native PRIVATE "${_cs179_numpy_include}")
        find_library(
            CS179_BOOST_PYTHON
            NAMES boost_python312 boost_python3 boost_python
            HINTS "${CS179_CMEEL_PREFIX}/lib")
        find_library(CS179_EIGENPY eigenpy HINTS "${CS179_CMEEL_PREFIX}/lib")
        if(CS179_BOOST_PYTHON)
            target_link_libraries(_native PRIVATE "${CS179_BOOST_PYTHON}")
        endif()
        if(CS179_EIGENPY)
            target_link_libraries(_native PRIVATE "${CS179_EIGENPY}")
        endif()
    endif()

    if(CS179_BUILD_CUDA)
        target_link_libraries(_native PRIVATE cs179_cuda CUDA::cudart)
        target_include_directories(_native PRIVATE cuda/include)
        target_compile_definitions(_native PRIVATE CS179_HAVE_CUDA=1)
    endif()

    install(
        TARGETS _native
        LIBRARY DESTINATION "${Python3_SITELIB}/cs179"
        RUNTIME DESTINATION "${Python3_SITELIB}/cs179"
    )
endif()
