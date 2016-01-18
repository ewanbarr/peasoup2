#ifndef PEASOUP_POLICIES_CUH
#define PEASOUP_POLICIES_CUH

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <map>

namespace peasoup {
    namespace policy {
	
	class cached_allocator
	{
	public:
	    // just allocate bytes
	    typedef char value_type;
	    
	    cached_allocator() {}
	    ~cached_allocator();
	    char *allocate(std::ptrdiff_t num_bytes);
	    void deallocate(char *ptr, size_t n);

	private:
	    typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
	    typedef std::map<char *, std::ptrdiff_t>     allocated_blocks_type;
	    free_blocks_type      free_blocks;
	    allocated_blocks_type allocated_blocks;

	    void free_all();
	};
	
    } // namespace policy
} // namespace peasoup

#include "misc/detail/policies.inl"

#endif //PEASOUP_POLICIES_CUH
