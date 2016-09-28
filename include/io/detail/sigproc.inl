#include <sstream>
#include <stdexcept>
#include <map>
#include "data_types/timefrequency.cuh"
#include "data_types/timeseries.cuh"
#include "io/stream.cuh"
#include "io/sigproc.cuh"

namespace peasoup {
    namespace io {
	namespace sigproc {

	    SigprocHeader::SigprocHeader(): 
		az_start(0.0), za_start(0.0), src_raj(0.0), src_dej(0.0),
		tstart(0.0), tsamp(0.0), period(0.0), fch1(0.0), foff(0.0),
		nchans(0), telescope_id(0), machine_id(0), data_type(0),
		ibeam(0), nbeams(0), nbits(0), barycentric(0),
		pulsarcentric(0), nbins(0), nsamples(0), nifs(0), npuls(0),
		refdm(0.0), signed_data(0), size(0) {}
	    
	    void throw_sigproc_error(std::string handle, std::string msg){
		std::stringstream error_msg;
                error_msg << "SigprocReader failed with error: "
                          << msg << std::endl
                          << "Stream handle: " << handle << std::endl;
                throw std::runtime_error(error_msg.str());
            }
	    
	    void _get_header_parameter(IOStream* stream, std::string& str){
		int len;
		char c_str[80];
		stream->read((char*)&len, sizeof(int));
		if( len < 0 || len >= 80 ){
		    std::stringstream error_msg;
		    error_msg << "Invalid length for header key: " << std::to_string(len);
		    throw_sigproc_error(stream->get_handle(),error_msg.str());
		}
		if (len>0)
		    stream->read(c_str, len*sizeof(char));
		c_str[len] = '\0';
		str = c_str;
	    }

	    sigproc_dtype get_data_type(IOStream* stream)
	    {
		SigprocHeader header = read_header(stream);
		stream->seekg(0,std::ios::beg);
		switch (header.data_type) {
		case 1:
		    return sigproc_dtype::FILTERBANK;
		case 2:
		    return sigproc_dtype::TIMESERIES;
		default:
		    return sigproc_dtype::UNKNOWN;
		}
	    }
	
	    SigprocHeader read_header(IOStream* stream)
	    {
		SigprocHeader header;
		std::string buffer;
		//Seek to start of stream where header should start
		stream->seekg(0, std::ios::beg);
		_get_header_parameter(stream,buffer);
		if(buffer != "HEADER_START" ) 
		    throw_sigproc_error(stream->get_handle(),"Invalid Sigproc header: HEADER_START tag missing.");
		
		while( true ) {
		    _get_header_parameter(stream,buffer);
		    
		    if( buffer == "HEADER_END" ) break;
		    else if( buffer == "source_name" ){
			_get_header_parameter(stream,buffer);
			header.source_name = buffer;
		    }
		    else if( buffer == "rawdatafile"){
			_get_header_parameter(stream,buffer);
			header.rawdatafile = buffer;
		    }
		    else if( buffer == "az_start" )      stream->read((char*)&header.az_start, sizeof(double));
		    else if( buffer == "za_start" )      stream->read((char*)&header.za_start, sizeof(double));
		    else if( buffer == "src_raj" )       stream->read((char*)&header.src_raj, sizeof(double));
		    else if( buffer == "src_dej" )       stream->read((char*)&header.src_dej, sizeof(double));
		    else if( buffer == "tstart" )        stream->read((char*)&header.tstart, sizeof(double));
		    else if( buffer == "tsamp" )         stream->read((char*)&header.tsamp, sizeof(double));
		    else if( buffer == "period" )        stream->read((char*)&header.period, sizeof(double));
		    else if( buffer == "fch1" )          stream->read((char*)&header.fch1, sizeof(double));
		    else if( buffer == "foff" )          stream->read((char*)&header.foff, sizeof(double));
		    else if( buffer == "nchans" )        stream->read((char*)&header.nchans, sizeof(int));
		    else if( buffer == "telescope_id" )  stream->read((char*)&header.telescope_id, sizeof(int));
		    else if( buffer == "machine_id" )    stream->read((char*)&header.machine_id, sizeof(int));
		    else if( buffer == "data_type" )     stream->read((char*)&header.data_type, sizeof(int));
		    else if( buffer == "ibeam" )         stream->read((char*)&header.ibeam, sizeof(int));
		    else if( buffer == "nbeams" )        stream->read((char*)&header.nbeams, sizeof(int));
		    else if( buffer == "nbits" )         stream->read((char*)&header.nbits, sizeof(int));
		    else if( buffer == "barycentric" )   stream->read((char*)&header.barycentric, sizeof(int));
		    else if( buffer == "pulsarcentric" ) stream->read((char*)&header.pulsarcentric, sizeof(int));
		    else if( buffer == "nbins" )         stream->read((char*)&header.nbins, sizeof(int));
		    else if( buffer == "nsamples" )      stream->read((char*)&header.nsamples, sizeof(int));
		    else if( buffer == "nifs" )          stream->read((char*)&header.nifs, sizeof(int));
		    else if( buffer == "npuls" )         stream->read((char*)&header.npuls, sizeof(int));
		    else if( buffer == "refdm" )         stream->read((char*)&header.refdm, sizeof(double));
		    else if( buffer == "signed" )        stream->read((char*)&header.signed_data, sizeof(unsigned char));
		    else { std::cerr << "Warning unknown parameter: " << buffer << std::endl; }
		}
		
		header.size = stream->tellg();
		// Compute the number of samples from the file size
		stream->seekg(0, std::ios::end);
		
		header.nsamples = (size_t)((stream->tellg()-header.size)* 8) / header.nchans / header.nbits;
		// Seek back to the end of the header
		stream->seekg(header.size, std::ios::beg);
		return header;
	    }

	
	    template <typename DataType>
	    void SigprocReader<DataType>::_get_metadata()
	    {
		header = read_header(stream);
		header_map["rawdatafile"] = header.rawdatafile;
		header_map["source_name"] = header.source_name;
		header_map["az_start"] = std::to_string(header.az_start);
		header_map["za_start"] = std::to_string(header.za_start);
		header_map["src_raj"] = std::to_string(header.src_raj);
		header_map["src_dej"] = std::to_string(header.src_dej);
		header_map["tstart"] = std::to_string(header.tstart);
		header_map["tsamp"] = std::to_string(header.tsamp);
		header_map["period"] = std::to_string(header.period);
		header_map["fch1"] = std::to_string(header.fch1);
		header_map["foff"] = std::to_string(header.foff);
		header_map["nchans"] = std::to_string(header.nchans);
		header_map["telescope_id"] = std::to_string(header.telescope_id);
		header_map["machine_id"] = std::to_string(header.machine_id);
		header_map["data_type"] = std::to_string(header.data_type);
		header_map["ibeam"] = std::to_string(header.ibeam);
		header_map["nbeams"] = std::to_string(header.nbeams);
		header_map["nbits"] = std::to_string(header.nbits);
		header_map["barycentric"] = std::to_string(header.barycentric);
		header_map["pulsarcentric"] = std::to_string(header.pulsarcentric);
		header_map["nbins"] = std::to_string(header.nbins);
		header_map["nsamples"] = std::to_string(header.nsamples);
		header_map["nifs"] = std::to_string(header.nifs);
		header_map["npuls"] = std::to_string(header.npuls);
		header_map["refdm"] = std::to_string(header.refdm);
		header_map["signed"] = std::to_string(header.signed_data);
	    }


	    template <>
	    void SigprocReader< type::TimeFrequencyBits<HOST> >::read()
	    {
		_get_metadata();
		unsigned nbits = header.nbits;
		if (!(nbits==1 || nbits==2 || nbits==4 || nbits==8))
		    throw_sigproc_error(stream->get_handle(),"Invalid sample bitwidth.");
		data.nbits = header.nbits;
		data.metadata.tsamp = header.tsamp;
		data.metadata.fch1 = header.fch1;
		data.metadata.nchans = header.nchans;
		data.metadata.foff = header.foff;
		data.data.resize(nbits * (size_t) header.nsamples * header.nchans / 8);
		stream->read_vector< decltype(data.data) >(data.data);
	    }

	    template <>
	    void SigprocReader< type::TimeSeries<HOST,float> >::read()
	    {
		_get_metadata();
		unsigned nbits = header.nbits;
		if (nbits!=32)
		    throw_sigproc_error(stream->get_handle(),"Invalid sample bitwidth (only 32 bit supported).");
		data.metadata.tsamp = header.tsamp;
		data.data.resize(header.nsamples);
		stream->read_vector< decltype(data.data) >(data.data);
	    }
	    
	} //sigproc
    } // io
} // peasoup

