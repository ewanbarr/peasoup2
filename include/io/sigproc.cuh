#ifndef PEASOUP_STREAM_CUH
#define PEASOUP_STREAM_CUH

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "io/stream.cuh"
#include "utils/utils.cuh"

namespace peasoup {
    namespace io {
	namespace exception {
	    
	    void throw_sigproc_error(SigprocReader& reader, std::string msg);	    
	    
	} //exception
	
	
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
	};
	
	template <typename DataType>
	class SigprocReader
	{
	private:
	    IOStream* stream;
	    DataType& data;
	    SigprocHeader header;
	    void _get_parameter();
	    void _get_metadata();

	public:
	    SigprocReader(DataType& data, IOStream* stream)
		:data(data),stream(stream){}
	    void read(DataType& data);
	    void get_handle(){return stream->handle;}
	};

    } // io
} // peasoup

#endif // PEASOUP_STREAM_CUH
