#include "cs179/rlds_loader.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        const std::filesystem::path data_dir =
            (argc > 1) ? std::filesystem::path(argv[1]) : std::filesystem::path("data");

        cs179::RldsObservationLoader loader(data_dir);
        std::cout << "demos=" << loader.num_demos()
                  << " total_steps=" << loader.total_steps()
                  << " control_hz=" << loader.control_hz() << '\n';

        if (loader.num_demos() == 0) {
            std::cerr << "No demos in cache\n";
            return 1;
        }

        const auto demo = loader.get_demo(0);
        for (const auto& key : loader.observation_keys()) {
            const auto& shape = demo.shapes.at(key);
            std::cout << key << " shape:";
            for (const auto dim : shape) {
                std::cout << ' ' << dim;
            }
            std::cout << " elements=" << demo.fields.at(key).size() << '\n';
        }

        const auto views = loader.get_demo_views(0);
        std::cout << "demo0_views=" << (views.has_value() ? "yes" : "no") << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "loader_test failed: " << ex.what() << '\n';
        return 1;
    }
}
