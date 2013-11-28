#g++ -v -m64 -std=c++0x -L /usr/lib/libOpenCL.so -I /opt/AMDAPP/include main.cpp -lOpenCL -lboost_filesystem

# CFLAGS for running
CPU_FLAGS = -Wall -std=c++0x -mtune=native -m64 -msse4.2 -O3

# CFLAGS for debugging
#CPU_FLAGS = -Wall -O0 -g

CXX = g++
RM = rm -rf
LIBS = -lOpenCL -lboost_filesystem

SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o)
DEPS := $(OBJS:.o=.d)


LDFLAGS = $(CPU_FLAGS) -L/usr/lib/libOpenCL.so 
CXXFLAGS = -I/opt/AMDAPP/include


TARGET = speckle_OpenCL
all:            $(TARGET)
$(TARGET):	main.o			\

	$(CXX) $(LDFLAGS) main.o 	\
		$(LIBS)			\
		-o $@


$(TARGET).o : $(TARGET).cpp
	$(CXX) $(CXXFLAGS) -c $(TARGET).cpp
