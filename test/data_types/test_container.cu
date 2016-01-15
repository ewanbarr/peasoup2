#include <algorithm>
#include <cstdlib>

#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "data_types/container.cuh"

#define SIZE 999

namespace peasoup {
    namespace test{

	struct dummy_metadata {
	    int dummy_variable;
	};

	template <System A, System B>
	void test_case(size_t size)
	{
	    Container<A,float,dummy_metadata> x;
	    x.metadata.dummy_variable = std::rand();
	    x.data.resize(SIZE);
	    std::generate(x.data.begin(),x.data.end(),std::rand);
	    Container<B,float,dummy_metadata> y = x;
	    ASSERT_EQ(x.data.size(),y.data.size());
	    for (int ii=0;ii<SIZE;ii++)
		ASSERT_EQ(x.data[ii],y.data[ii]);
	    ASSERT_EQ(x.metadata.dummy_variable,y.metadata.dummy_variable);
	}

	TEST(ContainerTest, ImplicitHtoD)
	{ test_case<HOST,DEVICE>(SIZE); }  
	
	TEST(ContainerTest, ImplicitDtoH)
	{ test_case<DEVICE,HOST>(SIZE); }
	
	TEST(ContainerTest, ImplicitHtoH)
	{ test_case<HOST,HOST>(SIZE); }
	
	TEST(ContainerTest, ImplicitDtoD)
	{ test_case<DEVICE,DEVICE>(SIZE); }
	
    }
}
