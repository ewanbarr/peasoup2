#include "gtest/gtest.h"
#include "utils/logging.hpp"

using namespace peasoup;

TEST(LoggerTest, TestLevels)
{
    auto& logger = logging::get_logger("test");
    LOG(logger,logging::ERROR,"this is an error");
    LOG(logger,logging::WARNING,"this is a warning");
    auto& logger2 = logging::get_logger("other");
    logger2.set_level(logging::ERROR);
    LOG(logger2,logging::WARNING,"If this prints then the test has failed");
    LOG(logger2,logging::ERROR,"This should print correctly"," beep"," beep");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
