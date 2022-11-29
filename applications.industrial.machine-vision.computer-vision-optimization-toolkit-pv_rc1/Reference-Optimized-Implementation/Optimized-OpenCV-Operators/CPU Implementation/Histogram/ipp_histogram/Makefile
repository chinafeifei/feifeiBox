##
## Copyright 2014-2021 Intel Corporation.
##
## This software and the related documents are Intel copyrighted  materials,  and
## your use of  them is  governed by the  express license  under which  they were
## provided to you (License).  Unless the License provides otherwise, you may not
## use, modify, copy, publish, distribute,  disclose or transmit this software or
## the related documents without Intel's prior written permission.
##
## This software and the related documents  are provided as  is,  with no express
## or implied  warranties,  other  than those  that are  expressly stated  in the
## License.
##

ifeq ($(G_SYSTEM),)
	exit 0
endif

CFLAGS   ?= $(G_CFLAGS)
CXXFLAGS ?= $(G_CXXFLAGS)

#
# Set binary architecture
#
ifeq ($(G_ARCH),ia32)
	CXXFLAGS    += -m32
endif
ifeq ($(G_ARCH),intel64)
	CXXFLAGS    += -m64
endif

#
# Set build name and paths
#
PROJ_NAME := ipp_histogram
ifeq ($(OUT_NAME),)
	OUT_NAME := $(PROJ_NAME)
endif

CONF_LOC        := $(G_CONF)
BUILD_PATH      := $(G_BUILD_ROOT)/$(G_ARCH)/$(CONF_LOC)
OBJ_PATH        := $(BUILD_PATH)/obj/$(PROJ_NAME)
OUT_FULL        := $(BUILD_PATH)/$(OUT_NAME)
INCLUDES        := -I$(IPPROOT)/include -I../../common/include -I./include -I/usr/include/opencv4/

IPP_LIBS_PATH   := $(IPPROOT)/$(G_IPP_PATH_SUFFIX)
TBB_LIBS_PATH   := $(TBBROOT)/$(G_TBB_PATH_SUFFIX)
TBB_LIBS_PATH_2 := $(TBBROOT)/$(G_TBB_PATH_SUFFIX_2)

LIBS += -L$(BUILD_PATH) -lcommon $(IPP_LIBS_PATH)/libippi.a $(IPP_LIBS_PATH)/libipps.a $(IPP_LIBS_PATH)/libippvm.a $(IPP_LIBS_PATH)/libippcore.a -ldl -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
ifeq ($(G_USERENDERER),1)
	CXXFLAGS += -DENABLE_RENDERING
	LIBS     += -lGL -lX11
endif
LIBS += -lpthread
ifneq ($(G_SYSTEM),macOS)
LIBS += -lrt
endif

ifneq ($(G_SYSTEM),macOS)
	CXXFLAGS += -fopenmp
endif

OBJS := $(patsubst src/%.cpp,$(OBJ_PATH)/%.o,$(wildcard src/*.cpp))

#
# Set variable targets
#
BUILD_TARGET := build

CLEAN_TARGET :=

all: out_path $(BUILD_TARGET)

build: obj_path out

out_path:
	mkdir -p $(BUILD_PATH)

obj_path:
	mkdir -p $(OBJ_PATH)

$(OBJ_PATH)/%.o : src/%.cpp
	$(CXX) -c $(G_EXE_CFLAGS) $(CXXFLAGS) $(INCLUDES) -o $@ $<

out: $(OBJS)
	$(CXX) $(G_EXE_LDFLAGS) $(CXXFLAGS) -o $(OUT_FULL) $? $(LIBS)

clean: $(CLEAN_TARGET)
	$(RM) $(OUT_FULL)
	$(RM) $(OBJS)
