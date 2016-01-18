#include "misc/policies.cuh"

namespace peasoup {
    namespace policy {

	cached_allocator::~cached_allocator()
	{
	    // free all allocations when cached_allocator goes out of scope
	    free_all();
	}

	char *cached_allocator::allocate(std::ptrdiff_t num_bytes)
	{
	    char *result = 0;
	    
	    // search the cache for a free block
	    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);
	    
	    if(free_block != free_blocks.end())
		{
		    // get the pointer
		    result = free_block->second;
		    
		    // erase from the free_blocks map
		    free_blocks.erase(free_block);
		}
	    else
		{
		    // no allocation of the right size exists
		    // create a new one with cuda::malloc
		    // throw if cuda::malloc can't satisfy the request
		    try
			{
			    // allocate memory and convert cuda::pointer to raw pointer
			    result = thrust::cuda::malloc<char>(num_bytes).get();
			}
		    catch(std::runtime_error &e)
			{
			    throw;
			}
		}

	    // insert the allocated pointer into the allocated_blocks map
	    allocated_blocks.insert(std::make_pair(result, num_bytes));
	    
	    return result;
	}
	
	void cached_allocator::deallocate(char *ptr, size_t n)
	{
	    // erase the allocated block from the allocated blocks map
	    allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
	    std::ptrdiff_t num_bytes = iter->second;
	    allocated_blocks.erase(iter);
	    
	    // insert the block into the free blocks map
	    free_blocks.insert(std::make_pair(num_bytes, ptr));
	}
	
	void cached_allocator::free_all()
	{
	    
	    // deallocate all outstanding blocks in both lists
	    for(free_blocks_type::iterator i = free_blocks.begin();
		i != free_blocks.end();
		++i)
		{
		    // transform the pointer to cuda::pointer before calling cuda::free
		    thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
		}
	    
	    for(allocated_blocks_type::iterator i = allocated_blocks.begin();
		i != allocated_blocks.end();
		++i)
		{
		    // transform the pointer to cuda::pointer before calling cuda::free
		    thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
		}
	}
	
    } // namespace policy
} // namespace peasoup


