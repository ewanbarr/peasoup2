#include "gtest/gtest.h"
#include "misc/system.cuh"
#include "io/stream.cuh"
#include "io/file.cuh"
#include "io/sigproc.cuh"
#include "data_types/timefrequency.cuh"
#include "data_types/timeseries.cuh"

using namespace peasoup;

void read_fil(io::IOStream* stream)
{
    typedef type::TimeFrequencyBits<HOST> data_type;
    data_type data(0);
    io::sigproc::SigprocReader< data_type > reader(data,stream);
    reader.read();
    ASSERT_FLOAT_EQ(data.metadata.tsamp,0.000320);
    ASSERT_FLOAT_EQ(data.metadata.fch1,1510.000000);
    ASSERT_FLOAT_EQ(data.metadata.foff,-1.090000);
    ASSERT_EQ(data.metadata.nchans,64);
    ASSERT_EQ(data.nbits,2);
    ASSERT_EQ(data.get_nsamps(),187520);
}

void read_tim(io::IOStream* stream)
{
    typedef type::TimeSeries<HOST,float> data_type;
    data_type data;
    io::sigproc::SigprocReader< data_type > reader(data,stream);
    reader.read();
    ASSERT_FLOAT_EQ(data.metadata.tsamp,0.000320);
    ASSERT_EQ(data.data.size(),187504);
}


TEST(SigprocIOTest, TestReadFilterbankFromFile)
{
    std::stringstream filename;
    filename << PEASOUP_DATA_DIR << "/sigproc/test.fil";
    io::IOStream* stream = new io::FileStream(filename.str());
    stream->prepare();
    read_fil(stream);
    delete stream;
}

TEST(SigprocIOTest, TestReadTimeseriesFromFile)
{
    std::stringstream filename;
    filename << PEASOUP_DATA_DIR << "/sigproc/test.tim";
    io::IOStream* stream = new io::FileStream(filename.str());
    stream->prepare();
    read_tim(stream);
    delete stream;
}
