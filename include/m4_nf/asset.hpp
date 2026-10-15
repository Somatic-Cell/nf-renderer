#pragma once
#include "types.hpp"
#include <string>
#include <vector>
namespace m4nf {
// Host-owned arrays. ModelView remains a plain host/device structure.
struct Asset {
    std::vector<float> weights,wavelengths,cosines,gs,scattering,extinction,angular_constants;
    std::uint32_t selected_step=0;
    float lambda_min=380,lambda_max=830;
    ModelView view()const;
};
Asset load_asset(const std::string& filename);
} // namespace m4nf
