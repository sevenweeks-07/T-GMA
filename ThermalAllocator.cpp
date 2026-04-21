#include "ThermalAllocator.h"
#include <iostream>

ThermalAllocator::ThermalAllocator() {
    head = nullptr;
    page_size = 2 * 1024 * 1024; // 2MB Blocks
    current_offset = 0;

    // Boot up the CUDA Driver API
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, device);

    // Reserve 1GB of "Room Numbers"
    cuMemAddressReserve(&virtual_base, 1024ULL * 1024 * 1024, page_size, 0, 0);
}

CUdeviceptr ThermalAllocator::allocate() {
    std::lock_guard<std::mutex> lock(allocator_lock);

    // 1. First-Fit Search
    PageNode* current = head;
    PageNode* prev = nullptr;
    while (current != nullptr) {
        if (current->is_free) {
            current->is_free = false;
            
            CUdeviceptr mapped_addr = virtual_base + current_offset;
            cuMemMap(mapped_addr, page_size, 0, current->physical_handle, 0);
            
            CUmemAccessDesc accessDesc = {};
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = 0;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cuMemSetAccess(mapped_addr, page_size, &accessDesc, 1);
            
            if (prev == nullptr) head = current->next;
            else prev->next = current->next;
            
            current_offset += page_size;
            active_allocations[mapped_addr] = current;
            
            std::cout << "[Allocator] Reused recycled frame at offset: " << current_offset << "\n";
            return mapped_addr;
        }
        prev = current;
        current = current->next;
    }

    // 2. No free holes found? Create a new physical bed.
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    
    CUmemGenericAllocationHandle handle;
    cuMemCreate(&handle, page_size, &prop, 0);
    
    CUdeviceptr mapped_addr = virtual_base + current_offset;
    cuMemMap(mapped_addr, page_size, 0, handle, 0);
    
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = 0;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cuMemSetAccess(mapped_addr, page_size, &accessDesc, 1);
    
    PageNode* newNode = new PageNode{handle, false, nullptr};
    active_allocations[mapped_addr] = newNode;
    current_offset += page_size;
    
    std::cout << "[Allocator] Minted new 2MB physical frame.\n";
    return mapped_addr;
}

void ThermalAllocator::free(CUdeviceptr addr) {
    std::lock_guard<std::mutex> lock(allocator_lock);
    if (active_allocations.find(addr) == active_allocations.end()) return;
    
    PageNode* block = active_allocations[addr];
    cuMemUnmap(addr, page_size);
    
    block->is_free = true;
    block->next = head;
    head = block;
    active_allocations.erase(addr);
    
    std::cout << "[Allocator] Freed memory and pushed to front of list.\n";
}

// Inside ThermalAllocator::defragment()
void ThermalAllocator::defragment() {
    std::lock_guard<std::mutex> lock(allocator_lock);
    
    if (active_allocations.empty()) {
        std::cout << "[Defrag] No active tensors to migrate. Skipping...\n";
        return;
    }

    std::cout << " [CRITICAL] Thermal spike detected! Initiating Physical Migration...\n";

    // 1. Target the first active allocation to "move" it to a new physical frame
    auto it = active_allocations.begin();
    CUdeviceptr v_addr = it->first;
    PageNode* node = it->second;

    // 2. Mint a new "Cool" physical frame
    CUmemGenericAllocationHandle new_handle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    cuMemCreate(&new_handle, page_size, &prop, 0);

    // 3. Reserve a temporary "Staging" virtual address to perform the copy
    CUdeviceptr staging_addr;
    cuMemAddressReserve(&staging_addr, page_size, page_size, 0, 0);
    cuMemMap(staging_addr, page_size, 0, new_handle, 0);
    
    // Set access for the staging address
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = 0;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cuMemSetAccess(staging_addr, page_size, &accessDesc, 1);

    // 4. THE CORE ACT: Physically copy bytes from Old Physical Frame -> New Physical Frame
    // Even if the ML model is "paused," the data stays safe.
    cuMemcpyDtoD(staging_addr, v_addr, page_size);

    // 5. REMAP: Point the ML model's virtual address to the NEW physical frame
    cuMemUnmap(v_addr, page_size);
    cuMemMap(v_addr, page_size, 0, new_handle, 0);
    cuMemSetAccess(v_addr, page_size, &accessDesc, 1);

    // 6. CLEANUP: Destroy the old physical frame and the staging area
    cuMemUnmap(staging_addr, page_size);
    cuMemAddressFree(staging_addr, page_size);
    cuMemRelease(node->physical_handle); // Release the "hot" hardware memory
    
    node->physical_handle = new_handle; // Update our record

    std::cout << "[Defrag] Data Migration Successful. Virtual Address " << v_addr << " remapped to new VRAM frame.\n";
}