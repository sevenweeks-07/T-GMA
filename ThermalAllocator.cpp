#include "ThermalAllocator.h"
#include <fstream>

ThermalAllocator::ThermalAllocator() {
    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // 1. Reserve the 1GB Virtual "Hallway"
    cuMemAddressReserve(&base_v_addr, total_vram, 0, 0, 0);

    // 2. Build the Linked List (The Rooms)
    head = nullptr;
    int num_pages = total_vram / page_size;

    // We build this backwards so 'head' is Room 1, next is Room 2, etc.
    for (int i = num_pages - 1; i >= 0; i--) {
        PageNode* newNode = new PageNode();
        newNode->v_addr = base_v_addr + (i * page_size);
        newNode->is_free = true;
        newNode->next = head;
        head = newNode;
    }
}

CUdeviceptr ThermalAllocator::allocate() {
    std::lock_guard<std::mutex> lock(allocator_lock);

    // Find the first free "Room" in our static hallway
    PageNode* curr = head;
    while (curr) {
        if (curr->is_free) {
            // Found a room! Now put a physical "Bed" in it.
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = 0;

            cuMemCreate(&curr->physical_handle, page_size, &prop, 0);
            cuMemMap(curr->v_addr, page_size, 0, curr->physical_handle, 0);

            CUmemAccessDesc accessDesc = {};
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = 0;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cuMemSetAccess(curr->v_addr, page_size, &accessDesc, 1);

            curr->is_free = false;
            active_allocations[curr->v_addr] = curr;
            
            std::cout << "[Allocator] Minted new 2MB physical frame at Virtual Address " << curr->v_addr << "\n";
            return curr->v_addr;
        }
        curr = curr->next;
    }
    return 0; // Out of Memory
}

void ThermalAllocator::free(CUdeviceptr addr) {
    std::lock_guard<std::mutex> lock(allocator_lock);
    if (active_allocations.find(addr) == active_allocations.end()) return;
    
    PageNode* block = active_allocations[addr];
    
    // Unplug the physical bed from the virtual room
    cuMemUnmap(addr, page_size);
    cuMemRelease(block->physical_handle); // Return silicone to the OS
    
    // Mark the room as a hole, but DO NOT move it in the linked list
    block->is_free = true;
    block->physical_handle = 0;
    active_allocations.erase(addr);
    
    std::cout << "[Allocator] Freed memory at Virtual Address " << addr << " (Created Hole).\n";
}

void ThermalAllocator::defragment() {
    std::lock_guard<std::mutex> lock(allocator_lock);
    
    // 1. Scan the static hallway for the first 'Free' hole and the first 'Active' block AFTER it
    PageNode* hole = nullptr;
    PageNode* active_node = nullptr;
    PageNode* curr = head;

    while (curr) {
        if (curr->is_free && !hole) {
            hole = curr;
        } else if (!curr->is_free && hole) {
            active_node = curr;
            break; // We found a block to move into the hole
        }
        curr = curr->next;
    }

    if (!hole || !active_node) {
        std::cout << "[Defrag] VRAM is already optimized. No fragmentation detected.\n";
        return;
    }

    std::cout << "🚨 [COMPACTION] Moving data from " << active_node->v_addr 
              << " into hole at " << hole->v_addr << "...\n";

    // 2. Perform the Physical Copy (Device-to-Device)
    cuMemcpyDtoD(hole->v_addr, active_node->v_addr, page_size);

    // 3. Update the Hardware Handles
    hole->physical_handle = active_node->physical_handle;
    hole->is_free = false;
    
    active_node->is_free = true;
    active_node->physical_handle = 0; 

    // 4. Update the Tracking Map
    active_allocations.erase(active_node->v_addr);
    active_allocations[hole->v_addr] = hole;

    std::cout << "[Defrag] Compaction successful. Fragmentation reduced.\n";
}

void ThermalAllocator::log_memory_state(int timestamp) {
    std::lock_guard<std::mutex> lock(allocator_lock);
    
    size_t total_free = 0;
    size_t largest_free = 0;
    size_t current_contiguous = 0;
    int free_nodes = 0;
    int active_nodes = active_allocations.size();

    PageNode* curr = head;
    while(curr) {
        if(curr->is_free) {
            free_nodes++;
            total_free += page_size;
            current_contiguous += page_size;
            // Keep track of the biggest gap we've seen so far
            if (current_contiguous > largest_free) {
                largest_free = current_contiguous;
            }
        } else {
            // We hit a wall (active memory). Reset the contiguous counter.
            current_contiguous = 0; 
        }
        curr = curr->next;
    }

    float frag_score = (total_free == 0) ? 0.0f : 1.0f - ((float)largest_free / total_free);

    std::ofstream file;
    file.open("fragmentation_log.csv", std::ios_base::app);
    file << timestamp << "," << active_nodes << "," << free_nodes << "," << frag_score << "\n";
    file.close();
}

ThermalAllocator::~ThermalAllocator() {
    // 1. Delete all the nodes in our linked list
    PageNode* curr = head;
    while (curr != nullptr) {
        PageNode* nextNode = curr->next;
        delete curr;
        curr = nextNode;
    }
    
    // 2. Give the 1GB Virtual Hallway back to the OS
    cuMemAddressFree(base_v_addr, total_vram);
    
    // 3. Destroy the CUDA context
    cuCtxDestroy(context);
    std::cout << "[System] T-GMA successfully safely shut down.\n";
}