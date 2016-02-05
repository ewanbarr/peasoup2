#ifndef PEASOUP_PRINTER_HPP
#define PEASOUP_PRINTER_HPP

#include <iostream>
#include <mutex>

namespace peasoup {
    namespace utils {
    
	inline std::ostream& print_one(std::ostream& os)
	{
	    return os;
	}
	
	template <class A0, class ...Args>
	inline std::ostream& print_one(std::ostream& os, const A0& a0, const Args& ...args)
	{
	    os << a0;
	    return print_one(os, args...);
	}
	
	template <class ...Args>
	inline std::ostream& print(std::ostream& os, const Args& ...args)
	{
	    return print_one(os, args...);
	}
	
	template <class ...Args>
	inline std::ostream& print(const Args& ...args)
	{
	    static std::mutex m;
	    std::lock_guard<std::mutex> _(m);
	    return print(std::cerr, args...);
	}

    } // utils
} // peasoup
	
#endif
