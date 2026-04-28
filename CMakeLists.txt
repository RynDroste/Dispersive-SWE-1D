cmake_minimum_required(VERSION 3.24)

# Picks GPU SM levels automatically when CMAKE_CUDA_ARCHITECTURES is unset.
project(DispersiveSWE1D LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Sensible default GPU arches (Pascal..Ada). Override via -DCMAKE_CUDA_ARCHITECTURES=86 etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 86 89)
endif()

# ---------------------------------------------------------------------------
# Dependencies (resolved via vcpkg manifest mode -- see vcpkg.json)
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)        # cudart, cufft
find_package(glfw3 CONFIG REQUIRED)
find_package(glad  CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

# ---------------------------------------------------------------------------
# Main viewer executable
# ---------------------------------------------------------------------------
add_executable(sim_viewer
    main.cpp
    Sim.cu
)
target_link_libraries(sim_viewer PRIVATE
    CUDA::cudart
    CUDA::cufft
    glfw
    glad::glad
    imgui::imgui
    OpenGL::GL
)
target_include_directories(sim_viewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Make sure relocatable device code is off (we have a single .cu file).
set_target_properties(sim_viewer PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    WIN32_EXECUTABLE FALSE   # console window kept for printf diagnostics
)

# ---------------------------------------------------------------------------
# 2D viewer executable -- builds only if Sim2Drender.cpp + Sim2D.cu exist.
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Sim2Drender.cpp" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Sim2D.cu")
    add_executable(sim2d_viewer
        Sim2Drender.cpp
        Sim2D.cu
    )
    target_link_libraries(sim2d_viewer PRIVATE
        CUDA::cudart
        CUDA::cufft
        glfw
        glad::glad
        imgui::imgui
        OpenGL::GL
    )
    target_include_directories(sim2d_viewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(sim2d_viewer PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        WIN32_EXECUTABLE FALSE
    )
    if(WIN32)
        add_custom_command(TARGET sim2d_viewer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:sim2d_viewer>
                $<TARGET_FILE_DIR:sim2d_viewer>
            COMMAND_EXPAND_LISTS)
    endif()
endif()

# ---------------------------------------------------------------------------
# Validation harness (Test.cu) -- builds only if Test.cu exists.
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Test.cu")
    add_executable(test_sim Test.cu Sim.cu)
    target_link_libraries(test_sim PRIVATE CUDA::cudart CUDA::cufft)
    target_include_directories(test_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(test_sim PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# ---------------------------------------------------------------------------
# Convenience: copy the runtime DLLs (GLFW etc.) next to the .exe on Windows.
# ---------------------------------------------------------------------------
if(WIN32)
    add_custom_command(TARGET sim_viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:sim_viewer>
            $<TARGET_FILE_DIR:sim_viewer>
        COMMAND_EXPAND_LISTS)
endif()
