cmake_minimum_required(VERSION 3.22)
project(openloong_orcagym)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)

# 查找 PyBind11 库
find_package(pybind11 CONFIG REQUIRED)

# 自动检测 CPU 核心数
find_package(Threads REQUIRED)
if(UNIX AND NOT APPLE)
    execute_process(
        COMMAND nproc
        OUTPUT_VARIABLE NUM_CORES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
elseif(APPLE)
    execute_process(
        COMMAND sysctl -n hw.ncpu
        OUTPUT_VARIABLE NUM_CORES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

message(STATUS "Detected CPU cores: ${NUM_CORES}")

# 将并行编译的线程数设置为核心数
set(ENV{CMAKE_BUILD_PARALLEL_LEVEL} ${NUM_CORES})


# 继续项目的其他配置


# 添加外部库的子模块
add_subdirectory(external/Openloong-dyn-control)

# 使用外部库中定义的头文件目录
set(dirEigen "external/Openloong-dyn-control/third_party/eigen3")
set(dirGlfw "external/Openloong-dyn-control/third_party/glfw")
set(dirPino "external/Openloong-dyn-control/third_party/pinocchio")
set(dirJson "external/Openloong-dyn-control/third_party/jsoncpp")
set(dirQuill "external/Openloong-dyn-control/third_party/quill")
set(dirQP "external/Openloong-dyn-control/third_party/qpOASES")

set(libURDF "external/Openloong-dyn-control/third_party/urdfdom")
set(incFmt "external/Openloong-dyn-control/third_party/quill/quill/bundled")
set(incBoost "external/Openloong-dyn-control/third_party/boost")
set(incMujoco "external/Openloong-dyn-control/third_party/mujoco")



# 将这些头文件目录传递给 openloong_dyn_ctrl
set(allDir ${dirEigen} ${dirGlfw} ${dirPino} ${dirJson} ${dirQuill} ${dirQP} )
set(allLib ${allDir} ${libURDF})
set(allInc ${dirEigen} ${dirGlfw} ${dirPino} ${dirJson} ${dirQuill} ${dirQP} ${incFmt} ${incBoost} ${incMujoco})

include_directories(${allInc})
include_directories("external/Openloong-dyn-control/algorithm")
include_directories("external/Openloong-dyn-control/common")
include_directories("external/Openloong-dyn-control/math")
include_directories("external/Openloong-dyn-control/sim_interface")

message(${CMAKE_SYSTEM_PROCESSOR})
if(${CMAKE_CXX_COMPILER} MATCHES "aarch64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
	list(APPEND allLib "external/Openloong-dyn-control/third_party/mujoco/lin_arm64")
else()
	list(APPEND allLib "external/Openloong-dyn-control/third_party/mujoco/lin_x64")
endif()
link_directories(${allLib})


# 强制所有静态库使用 -fPIC 编译，以支持动态链接
set_target_properties(core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 生成 Python 模块
pybind11_add_module(openloong_dyn_ctrl
    cpp/openloong_wbc.cpp
    cpp/orcagym_interface.cpp
    cpp/bindings.cpp
)

# 将生成的 .so 文件复制到上一级目录并确保目标目录存在
add_custom_command(TARGET openloong_dyn_ctrl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/libs  # 确保目录存在
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:openloong_dyn_ctrl> ${CMAKE_SOURCE_DIR}/libs/  # 复制 .so 文件
)

# 将头文件路径传递给 openloong_dyn_ctrl 目标
target_include_directories(openloong_dyn_ctrl PRIVATE ${EXTERNAL_INCLUDE_DIRS})

# 链接外部库 (假设 core 是 Openloong-dyn-control 生成的库)
target_link_libraries(openloong_dyn_ctrl PRIVATE core pthread)