#ifndef THERMAL_ALLOCATOR_H
#define THERMAL_ALLOCATOR_H

#include <cuda.h>
#include <iostream>
#include <vector>
#include <map>
#include <mutex>

struct PageNode {
    CUdeviceptr v_addr;
    CUmemGenericAllocationHandle physical_handle;
    bool is_free;
    PageNode* next;

    PageNode() : v_addr(0), physical_handle(0), is_free(true), next(nullptr) {}
};

class ThermalAllocator {
public:
    ThermalAllocator();
    ~ThermalAllocator();

    CUdeviceptr allocate();
    void free(CUdeviceptr addr);
    void defragment();
    void log_memory_state(int timestamp);

private:
    // Core CUDA handles
    CUdevice device;
    CUcontext context;
    CUdeviceptr base_v_addr;

    // Configuration
    const size_t total_vram = 1024 * 1024 * 1024; // 1GB
    const size_t page_size = 2 * 1024 * 1024;    // 2MB

    // Management
    PageNode* head;
    std::map<CUdeviceptr, PageNode*> active_allocations;
    std::mutex allocator_lock;
};

#endif