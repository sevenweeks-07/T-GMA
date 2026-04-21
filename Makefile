CXX = g++
CXXFLAGS = -std=c++11 -Wall -I/usr/local/cuda/include
LDFLAGS = -L/usr/lib/x86_64-linux-gnu/stubs -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcuda -pthread

TARGET = allocator_engine
OBJS = main.o ThermalAllocator.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

main.o: main.cpp ThermalAllocator.h
	$(CXX) $(CXXFLAGS) -c main.cpp

ThermalAllocator.o: ThermalAllocator.cpp ThermalAllocator.h
	$(CXX) $(CXXFLAGS) -c ThermalAllocator.cpp

clean:
	rm -f $(OBJS) $(TARGET)