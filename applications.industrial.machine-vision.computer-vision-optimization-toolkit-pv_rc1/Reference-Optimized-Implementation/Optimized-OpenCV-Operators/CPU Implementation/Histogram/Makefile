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

#
# Base make configuaration
#
include ../common/Makefile_base.mk

#
# Define paths for build targets
#
MAKE_COMMON        := $(MAKE) -C ../common
MAKE_COMMON_TBB    := $(MAKE) -C ../common G_USETBB=0 G_CONF_TBB=$(G_CONF)
MAKE_FFT           := $(MAKE) -C ./ipp_fft
MAKE_RESIZE        := $(MAKE) -C ./ipp_resize_mt
MAKE_THREAD        := $(MAKE) -C ./ipp_thread
MAKE_MORPH         := $(MAKE) -C ./ipp_morphology
MAKE_HISTOGRAM	   := $(MAKE) -C ./ipp_histogram

all: examples

common:
	$(MAKE_COMMON)
	$(MAKE_COMMON_TBB)

examples: common
	$(MAKE_FFT)
	$(MAKE_RESIZE)
	$(MAKE_THREAD)
	$(MAKE_MORPH)
	$(MAKE_HISTOGRAM)

clean:
	$(MAKE_COMMON)       clean
	$(MAKE_COMMON_TBB)   clean
	$(MAKE_FFT)          clean
	$(MAKE_RESIZE)       clean
	$(MAKE_THREAD)       clean
	$(MAKE_MORPH)        clean
	$(MAKE_HISTOGRAM)    clean

