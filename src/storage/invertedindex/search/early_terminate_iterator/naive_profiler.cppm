module;
#include <chrono>
#include <iostream>

export module naive_profiler;

import stl;

namespace infinity {
export struct NaiveProfiler {
    NaiveProfiler(const String &name) : name_(name) { begin_ts_ = std::chrono::high_resolution_clock::now(); }

    ~NaiveProfiler() {
        auto end_ts = std::chrono::high_resolution_clock::now();
        using TimeDurationType = std::chrono::duration<float, std::milli>;
        TimeDurationType duration = end_ts - begin_ts_;
        std::cout << name_ << " duration: " << duration << std::endl;
    }

    void Foo() {}

    String name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> begin_ts_;
};
} // namespace infinity