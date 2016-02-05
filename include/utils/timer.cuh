#ifndef PEASOUP_TIMER_CUH
#define PEASOUP_TIMER_CUH

namespace peasoup {
    namespace utils {
	class Timer
	{
	private:
	    float elapsed_time;
	    cudaEvent_t start_event;
	    cudaEvent_t stop_event;
	public:
	    Timer(){
		cudaEventCreate(&start_event);
		cudaEventCreate(&stop_event);
	    }
	    
	    ~Timer(){
		cudaEventDestroy(start_event);
		cudaEventDestroy(stop_event);
	    }

	    void start(){ cudaEventRecord(start_event, 0); }
	    void stop(){
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
	    }
	    float elapsed(){
		float elapsed_time;
		cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
		return elapsed_time;
	    }
	};
    } //utils
}//peasoup

#endif //PEASOUP_TIMER_CUH
