list(APPEND CMAKE_MODULE_PATH "/home/smutch/.conda/envs/default/lib/python3.6/site-packages/skbuild/resources/cmake")
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

add_cython_target(cygpu_tests "pygpu_tests.pyx" CXX)
add_library(pygpu_tests MODULE ${cygpu_tests})

target_include_directories(pygpu_tests PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(pygpu_tests libgpu_tests)

python_extension_module(pygpu_tests)

# install(TARGETS pygpu_tests
    # LIBRARY DESTINATION lib)
