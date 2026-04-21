#ifndef THERMAL_ALLOCATOR_H
#define THERMAL_ALLOCATOR_H

#include <cuda.h>
#include <mutex>
#include <unordered_map>

// Our blueprint for a single 2MB physical box
struct PageNode {
    CUmemGenericAllocationHandle physical_handle;
    bool is_free;
    PageNode* next;
};

class ThermalAllocator {
private:
    PageNode* head;
    CUdeviceptr virtual_base;
    size_t page_size;
    size_t current_offset;
    
    std::mutex allocator_lock;
    std::unordered_map<CUdeviceptr, PageNode*> active_allocations;

public:
    ThermalAllocator();                    // Constructor
    CUdeviceptr allocate();                // Give memory to ML model
    void free(CUdeviceptr addr);           // Take memory back
    void defragment();       // Take memory back
};

#endif // THERMAL_ALLOCATOR_H