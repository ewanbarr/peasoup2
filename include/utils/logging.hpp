#ifndef PEASOUP_LOGGING_HPP
#define PEASOUP_LOGGING_HPP

#ifndef MIN_LOG_LEVEL
    #define MIN_LOG_LEVEL 0
#endif // MIN_LOG_LEVEL

#define LOG(logger,level,...) if (level>=MIN_LOG_LEVEL) logger.log(level,__PRETTY_FUNCTION__,__VA_ARGS__)

#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>
#include <ctime>
#include <map>
#include <memory>
#include <string>
#include "utils/printer.hpp"

namespace peasoup {
    namespace logging {
	
	enum LogLevel {
	    DEBUG,
	    INFO,
	    WARNING,
	    ERROR,
	    CRITICAL
	};
	
	static LogLevel default_log_level = WARNING;

	namespace internal {
	    
	    class Logger
	    {
	    private:
		std::ostream* _ostream;
		LogLevel level;
		char buf[128];

	    public:
		const std::string name;
		Logger(const std::string& name):name(name),_ostream(&std::cerr),level(default_log_level){}
		void set_level(LogLevel new_level){level = new_level;}
		void set_ostream(std::ostream* new_ostream){_ostream=new_ostream;}

		template <class ...Args>
		void log(const LogLevel level, const std::string& func, const Args& ...args)
		{
		    if (level>=this->level){
			std::time_t t = std::time(NULL);
			std::strftime(buf, 128, "%Y-%m-%d %H:%M:%S", std::gmtime(&t));
			utils::print(*_ostream,func,"\n",buf," -- Thread: ",std::this_thread::get_id(),
				     " -- Name: ",name," -- ",args...,"\n\n");
		    }
		}
	    };
	    
	    static std::map<std::string,std::shared_ptr<Logger> > loggers;
	}
	
	static internal::Logger& get_logger(const std::string& name)
	{
	    if ( internal::loggers.find(name) == internal::loggers.end() ){
		internal::loggers[name] = std::make_shared<internal::Logger>(name);
	    }
	    return *(internal::loggers[name]);
	}

	static void set_default_log_level_from_string(std::string level_str)
        {
	    std::transform(level_str.begin(), level_str.end(),level_str.begin(), ::toupper);
	    if (level_str == "CRITICAL") default_log_level = CRITICAL;
	    else if (level_str == "ERROR") default_log_level = ERROR;
	    else if (level_str == "WARNING") default_log_level = WARNING;
	    else if (level_str == "INFO") default_log_level = INFO;
	    else if (level_str == "DEBUG") default_log_level = DEBUG;
	    else {
		LOG(get_logger("default"),WARNING,"Invalid log level provided, defaulting to WARNING level\n",
		    "Valid levels are CRITICAL, ERROR, WARNING, INFO, DEBUG");
	    }
	}
    } //logging
} //peasoup

#endif //PEASOUP_LOGGING_HPP
