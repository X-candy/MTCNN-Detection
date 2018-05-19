cmake_minimum_required(VERSION 3.8)
project(MTCNN)

set(CMAKE_CXX_STANDARD 11)



# ------------------------------------------------------------------------------
# OpenCV
# ------------------------------------------------------------------------------
set(OpenCV_DIR "/usr/local/Cellar/opencv/2.4.13.2/share/OpenCV")

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV Not Found!")
endif (NOT OpenCV_FOUND)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# ------------------------------------------------------------------------------
# Caffe
# ------------------------------------------------------------------------------
set(Caffe_DIR /Users/xcandy/WorkSpace/Caffe/caffe/buidl/Install/share/Caffe)
find_package(Caffe REQUIRED)
message(STATUS "Caffe library status:")
message(STATUS "    libraries: ${Caffe_LIBRARIES}")

# ------------------------------------------------------------------------------
# Add Version Control
# ------------------------------------------------------------------------------
SET(ROOT_DIR ${PROJECT_SOURCE_DIR})
# ------------------------------------------------------------------------------
# Add Version Control
# ------------------------------------------------------------------------------
SET(ROOT_DIR ${PROJECT_SOURCE_DIR})
IF(EXISTS "${ROOT_DIR}/.svn/")
    FIND_PACKAGE(Subversion)

    IF(SUBVERSION_FOUND)
        Subversion_WC_INFO(${ROOT_DIR} Project)
        SET(VERSION_REVISION ${Project_WC_REVISION})
    ENDIF(SUBVERSION_FOUND)
ENDIF(EXISTS "${ROOT_DIR}/.svn/")

CONFIGURE_FILE(
        "Version.h.in"
        "Version.h"
)

set(SOURCE_FILES main.cpp MTCNN.cpp MTCNN.h)
add_executable(MTCNN ${SOURCE_FILES} )
target_link_libraries(MTCNN ${OpenCV_LIBS} ${Caffe_LIBRARIES})