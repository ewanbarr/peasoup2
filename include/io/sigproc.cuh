#ifndef PEASOUP_SIGPROC_CUH
#define PEASOUP_SIGPROC_CUH

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "io/stream.cuh"
#include "utils/utils.cuh"

namespace peasoup {
    namespace io {
	namespace sigproc {
	    
	    void throw_sigproc_error(std::string handle, std::string msg);	    
	    
	    struct SigprocHeader 
	    {
		// Naming convention from sigproc header keys
		std::string source_name; /*!< Source name.*/
		std::string rawdatafile; /*!< Name of original data file.*/
		double az_start; /*!< Azimuth angle (deg).*/
		double za_start; /*!< Zenith angle (deg).*/
		double src_raj; /*!< Right ascension (hhmmss.ss format).*/
		double src_dej; /*!< Declination (ddmmss.ss format).*/
		double tstart; /*!< Modified Julian date of first sample.*/
		double tsamp; /*!< Sampling time (seconds).*/
		double period; 
		double fch1; /*!< Frequency of top channel (MHz).*/
		double foff; /*!< Channel bandwith (MHz).*/
		int    nchans; /*!< Number of frequency channels.*/
		int    telescope_id; /*!< Sigproc telescope ID.*/
		int    machine_id; /*!< Sigproc backend ID.*/
		int    data_type; /*!< Sigproc data type ID.*/ 
		int    ibeam; /*!< Beam number.*/
		int    nbeams; /*!< Number of beams.*/
		int    nbits; /*!< Number of bits per sample.*/
		int    barycentric; 
		int    pulsarcentric;
		int    nbins;  
		int    nsamples; /*!< Number of time samples.*/
		int    nifs; 
		int    npuls;
		double refdm; /*!< Reference DM of data.*/
		unsigned char signed_data; /*!< char or unsigned char.*/
		unsigned int size; /*!< Header size in bytes.*/
		SigprocHeader();
	    };
	    
	    enum sigproc_dtype {
		FILTERBANK,
		TIMESERIES,
		UNKNOWN
	    };

	    SigprocHeader read_header(IOStream* stream);
	    
	    sigproc_dtype get_data_type(IOStream* stream);
	    
	    template <typename DataType>
	    class SigprocReader
	    {
	    private:
		typedef std::map<std::string,std::string> map_type;
		IOStream* stream;
		DataType& data;
		SigprocHeader header;
		map_type header_map;
		void _get_metadata();
		
	    public:
		SigprocReader(DataType& data, IOStream* stream)
		    :data(data),stream(stream){}
		void read();
		map_type get_header_map(){return header_map;}
		
	    };
	    
	} // sigproc
    } // io
} // peasoup

#include "io/detail/sigproc.inl"

#endif // PEASOUP_SIGPROC_CUH
