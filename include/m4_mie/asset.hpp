#pragma once
// Host-only on-disk MIE1LUT v1 loader and strict transport/sampling validation.
#include "table.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace m4mie {
struct Asset {
    std::vector<float> wavelengths,theta_edges,masses,alias_probs,qext,qsca,g;
    std::vector<std::uint32_t> aliases;
    float radius=0.0f,temperature=0.0f,density=0.0f;
    View view() const noexcept {
        View v{};
        v.wavelengths_nm=wavelengths.data();
        v.theta_edges_rad=theta_edges.data();
        v.masses=masses.data();v.alias_prob=alias_probs.data();v.alias=aliases.data();
        v.qext=qext.data();v.qsca=qsca.data();v.g=g.data();
        v.nw=static_cast<std::uint32_t>(wavelengths.size());
        v.nbins=static_cast<std::uint32_t>(theta_edges.size()-1);
        v.radius_m=radius;
        return v;
    }
};
Asset load_asset(const std::string& path);
} // namespace m4mie
