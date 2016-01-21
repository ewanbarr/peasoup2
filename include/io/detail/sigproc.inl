#include <sstream>
#include <stdexcept>
#include "data_types/timefrequency.cuh"
#include "data_types/timeseries.cuh"
#include "io/stream->cuh"
#include "io/sigproc.cuh"

namespace peasoup {
    namespace io {
	namespace exception {

	    void throw_sigproc_error(std::string handle, std::string msg){
		std::stringstream error_msg;
                error_msg << "SigprocReader failed with error: "
                          << msg << std::endl
                          << "Stream handle: " << handle << std::endl;
                throw std::runtime_error(error_msg.str());
            }
	    
	} //exception

	
	template <typename DataType>
	void SigprocReader<DataType>::_get_parameter(std::string str){
	    int len;
	    char c_str[80];
	    stream->read((char*)&len, sizeof(int));
	    if( len <= 0 || len >= 80 ){
		std::stringstream error_msg;
		error_msg << "Invalid length for header key: " << std::to_string(len);
		exception::throw_sigproc_error(stream->handle,error_msg);
	    }
	    stream->read(c_str, len*sizeof(char));
	    c_str[len] = '\0';
	    str = c_str;
	}
	
	template <typename DataType>
	void SigprocReader<DataType>::_read_header()
	{
	    
	    std::string s;
	    //Seek to start of stream where header should start
	    stream->seekg(0, std::ios::beg);
	    _get_parameter(s);
	    if(s != "HEADER_START" ) 
		exception::throw_sigproc_error(stream->handle,"Invalid Sigproc header: HEADER_START tag missing.");
		    
	    while( true ) {
		_get_parameter(s);
		
		if( s == "HEADER_END" ) break;
		else if( s == "source_name" ){
		    _get_parameter(s);
		    header.source_name = s;
		}
		else if( s == "rawdatafile"){
		    _get_parameter(s);
		    header.rawdatafile = s;
		}
		else if( s == "az_start" )      stream->read((char*)&header.az_start, sizeof(double));
		else if( s == "za_start" )      stream->read((char*)&header.za_start, sizeof(double));
		else if( s == "src_raj" )       stream->read((char*)&header.src_raj, sizeof(double));
		else if( s == "src_dej" )       stream->read((char*)&header.src_dej, sizeof(double));
		else if( s == "tstart" )        stream->read((char*)&header.tstart, sizeof(double));
		else if( s == "tsamp" )         stream->read((char*)&header.tsamp, sizeof(double));
		else if( s == "period" )        stream->read((char*)&header.period, sizeof(double));
		else if( s == "fch1" )          stream->read((char*)&header.fch1, sizeof(double));
		else if( s == "foff" )          stream->read((char*)&header.foff, sizeof(double));
		else if( s == "nchans" )        stream->read((char*)&header.nchans, sizeof(int));
		else if( s == "telescope_id" )  stream->read((char*)&header.telescope_id, sizeof(int));
		else if( s == "machine_id" )    stream->read((char*)&header.machine_id, sizeof(int));
		else if( s == "data_type" )     stream->read((char*)&header.data_type, sizeof(int));
		else if( s == "ibeam" )         stream->read((char*)&header.ibeam, sizeof(int));
		else if( s == "nbeams" )        stream->read((char*)&header.nbeams, sizeof(int));
		else if( s == "nbits" )         stream->read((char*)&header.nbits, sizeof(int));
		else if( s == "barycentric" )   stream->read((char*)&header.barycentric, sizeof(int));
		else if( s == "pulsarcentric" ) stream->read((char*)&header.pulsarcentric, sizeof(int));
		else if( s == "nbins" )         stream->read((char*)&header.nbins, sizeof(int));
		else if( s == "nsamples" )      stream->read((char*)&header.nsamples, sizeof(int));
		else if( s == "nifs" )          stream->read((char*)&header.nifs, sizeof(int));
		else if( s == "npuls" )         stream->read((char*)&header.npuls, sizeof(int));
		else if( s == "refdm" )         stream->read((char*)&header.refdm, sizeof(double));
		else if( s == "signed" )        stream->read((char*)&header.signed_data, sizeof(unsigned char));
		else { std::cerr << "Warning unknown parameter: " << s << std::endl; }
	    }
	    
	    data.metadata.original["rawdatafile"] = header.rawdatafile;
	    data.metadata.original["source_name"] = header.source_name;
	    data.metadata.original["az_start"] = std::to_string(header.az_start);
	    data.metadata.original["za_start"] = std::to_string(header.za_start);
	    data.metadata.original["src_raj"] = std::to_string(header.src_raj);
	    data.metadata.original["src_dej"] = std::to_string(header.src_dej);
	    data.metadata.original["tstart"] = std::to_string(header.tstart);
	    data.metadata.original["tsamp"] = std::to_string(header.tsamp);
	    data.metadata.original["period"] = std::to_string(header.period);
	    data.metadata.original["fch1"] = std::to_string(header.fch1);
	    data.metadata.original["foff"] = std::to_string(header.foff);
	    data.metadata.original["nchans"] = std::to_string(header.nchans);
	    data.metadata.original["telescope_id"] = std::to_string(header.telescope_id);
	    data.metadata.original["machine_id"] = std::to_string(header.machine_id);
	    data.metadata.original["data_type"] = std::to_string(header.data_type);
	    data.metadata.original["ibeam"] = std::to_string(header.ibeam);
	    data.metadata.original["nbeams"] = std::to_string(header.nbeams);
	    data.metadata.original["nbits"] = std::to_string(header.nbits);
	    data.metadata.original["barycentric"] = std::to_string(header.barycentric);
	    data.metadata.original["pulsarcentric"] = std::to_string(header.pulsarcentric);
	    data.metadata.original["nbins"] = std::to_string(header.nbins);
	    data.metadata.original["nsamples"] = std::to_string(header.nsamples);
	    data.metadata.original["nifs"] = std::to_string(header.nifs);
	    data.metadata.original["npuls"] = std::to_string(header.npuls);
	    data.metadata.original["refdm"] = std::to_string(header.refdm);
	    data.metadata.original["signed"] = std::to_string(header.signed);
	    
	    header.size = stream->tellg();
	    if( 0 == header.nsamples ) {
		// Compute the number of samples from the file size
		stream->seekg(0, std::ios::end);
		size_t total_size = stream->tellg();
		header.nsamples = (total_size-header.size) / header.nchans * 8 / header.nbits;
		// Seek back to the end of the header
		stream->seekg(header.size, std::ios::beg);
	    }
	}
	
	template <>
	void SigprocReader< type::TimeFrequencyBits<HOST> >::read(type::TimeFrequencyBits<HOST>& data)
	{
	    _read_header();
	    unsigned nbits = header.nbits;
	    if (!(nbits==1 || nbits==2 || nbits==4 || nbits==8))
		exception::throw_sigproc_error(stream->handle,"Invalid sample bitwidth.");

	    data.nbits = header.nbits;
	    data.metadata.tsamp = header.tsamp;
	    data.metadata.fch1 = header.fch1;
	    data.metadata.nchans = header.nchans;
	    data.metadata.foff = header.foff;
	    data.metadata.fch1 = header.fch1;
	    data.resize(nbits * header.nsamples * header.nchans / 8);
	    stream->read_vector< decltype(data.data) >(data.data);
	}

	template <>
	void SigprocReader< type::TimeSeries<HOST,float> >::read(type::TimeSeries<HOST,float>& data)
        {
            _read_header();
            if (nbits!=32)
		exception::throw_sigproc_error(stream->handle,"Invalid sample bitwidth (only 32 bit supported).");
	    data.metadata.tsamp = header.tsamp;
	    data.data.resize(header.nsamples);
	    stream->read_vector< decltype(data.data) >(data.data);
        }

    } // io
} // peasoup

#endif // PEASOUP_STREAM_CUH
