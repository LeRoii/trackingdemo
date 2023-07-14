# CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# set(name "facerec")
# project(facerec_cpp_samples)

# #SET(OpenCV_DIR /path/to/your/opencv/installation)

# add_compile_options(-std=c++11)

# # packages
# http://opencv.org

# # probably you should loop through the sample files here
# add_executable(facerec_demo facerec_demo.cpp)
# target_link_libraries(facerec_demo opencv_core opencv_face opencv_imgproc opencv_highgui)


cmake_minimum_required(VERSION 2.4)
project(main)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

message("msg CMAKE_PREFIX_PATH:::${CMAKE_PREFIX_PATH}")
# set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/OpenCVConfig.cmake")
find_package( OpenCV  REQUIRED )
find_package( CUDA )

message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS：${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIB_DIR：${OpenCV_LIB_DIR}")
message(STATUS "OpenCV_LIBS：${OpenCV_LIBS}")

set(COMMONCLS /usr/src/jetson_multimedia_api/samples/common/classes)

set(COMMONCLSFILES
    ${COMMONCLS}/NvBuffer.cpp
    ${COMMONCLS}/NvElement.cpp
    ${COMMONCLS}/NvElementProfiler.cpp
    ${COMMONCLS}/NvEglRenderer.cpp
    ${COMMONCLS}/NvLogging.cpp
    ${COMMONCLS}/NvV4l2Element.cpp
    ${COMMONCLS}/NvV4l2ElementPlane.cpp
    ${COMMONCLS}/NvVideoEncoder.cpp
    ${COMMONCLS}/NvVideoDecoder.cpp
    ${COMMONCLS}/NvLogging.cpp)


include_directories(
    ${CUDA_INCLUDE_DIRS}
    ./include/
    /usr/src/jetson_multimedia_api/include
    /usr/src/jetson_multimedia_api/samples/common/algorithm/cuda/
    ${PROJECT_SOURCE_DIR}/include/idetector/
    ${PROJECT_SOURCE_DIR}/src/plugin/
 )

# add_compile_options(-std=c++11 )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -pthread -fopenmp")


set(CPP_FILES ${PROJECT_SOURCE_DIR}/src/main.cpp 
            ${PROJECT_SOURCE_DIR}/src/nvrender.cpp
            ${PROJECT_SOURCE_DIR}/src/fusion.cpp
            ${PROJECT_SOURCE_DIR}/src/tracker.cpp
             )

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/idetector/*.cpp
                     ${PROJECT_SOURCE_DIR}/src/idetector/*.cu 
                    ${PROJECT_SOURCE_DIR}/src/idetector.cpp)

message(STATUS "CPP_FILES${CPP_FILES}")

file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/src/plugin/*.cu)
add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

add_executable(main ${CPP_FILES} ${SRCS} ${COMMONCLSFILES})
target_link_libraries( main sdireader ${OpenCV_LIBRARIES} myplugins
-L/usr/lib/aarch64-linux-gnu/tegra/ -lnvbuf_utils  
-L/usr/lib/aarch64-linux-gnu/ -lv4l2 -lEGL -lGLESv2 -lX11 
${GLUT_LIBRARY}
${CUDA_LIBRARIES} ${CUDNN_LIBRARIES}
)
