#pragma once

#include <cstdint>
#include <type_traits>

namespace m1 {

inline constexpr std::uint32_t no_index = 0xffffffffu;

enum class FailureStage : std::uint32_t {
    Camera, Surface, SunNee, AreaNee, Bsdf, Miss, Film,
    RadianceAnyHit, ShadowAnyHit, Atmosphere, AtmosphereSunNee, Planet, Rain, RainSunNee
};

// One name per validation condition. Keep the values and their labels together
// so a GPU failure can be interpreted without guessing from an aggregate count.
#define M1_FAILURE_REASONS(X) \
    X(None, "none") \
    X(ReflectanceSpectrum, "rgb.r, rgb.g, rgb.b, wavelength_nm, result") \
    X(EmissionRgb, "rgb.r, rgb.g, rgb.b") \
    X(EmissionAmplitude, "rgb.r, rgb.g, rgb.b, amplitude") \
    X(EmissionSpectrum, "spectrum, amplitude, d65, intensity") \
    X(ShadowRay, "origin.x, origin.y, origin.z, direction.x, direction.y, direction.z, distance, direction_length2") \
    X(ShadowUnhandled, "shadow_invalid") \
    X(DirectBsdf, "wo.x, wo.y, wo.z, wi.x, wi.y, wi.z, wo_length2, wi_length2, f, bsdf_pdf, color, metallic, roughness") \
    X(DirectInputs, "radiance, light_pdf") \
    X(DirectContribution, "radiance, light_pdf, f, bsdf_pdf, cosine, mis_weight") \
    X(SurfaceOpacity, "texture_alpha, opacity_factor, cutoff, uv.x, uv.y") \
    X(MissingNormals, "has_normal") \
    X(InterpolatedNormal, "normal.x, normal.y, normal.z, object_normal.x, object_normal.y, object_normal.z, b0, b1, b2") \
    X(MissingTangentFrame, "has_tangent, has_normal") \
    X(NormalTexture, "normal_r, normal_g, uv.x, uv.y") \
    X(TangentHandedness, "tangent.w, w0, w1, w2, b0, b1, b2") \
    X(TangentFrame, "tangent.x, tangent.y, tangent.z, bitangent.x, bitangent.y, bitangent.z, normal.x, normal.y, normal.z, t_length2, b_length2, object_tangent.x, object_tangent.y, object_tangent.z") \
    X(TriangleTangentFrame, "uv_det_mantissa, uv_det_exponent, normal.x, normal.y, normal.z, uv0.x, uv0.y, uv1.x, uv1.y, uv2.x, uv2.y, edge1_max_abs, edge2_max_abs") \
    X(MappedNormal, "normal.x, normal.y, normal.z, map.x, map.y, map.z") \
    X(HitSbt, "geometry_type, mesh_count, material_count") \
    X(HitGeometry, "twice_area, cross.x, cross.y, cross.z") \
    X(MissingSurfaceUV, "has_diffuse_texture, has_rm_texture, has_emissive_texture") \
    X(EmitterMissing, "light_count") \
    X(EmitterGeometry, "connection_cosine, ray_cosine, area, distance2, light_count, connection.x, connection.y, connection.z, ray_direction.x, ray_direction.y, ray_direction.z, previous_position.x, previous_position.y, previous_position.z") \
    X(EmitterPdf, "light_pdf, connection_cosine, area, distance2, light_count") \
    X(EmitterContribution, "radiance, light_pdf, mis_weight") \
    X(MetallicParameters, "color, metallic, roughness, uv.x, uv.y") \
    X(UnsupportedMaterial, "material_type") \
    X(SunParameters, "radiance, light_pdf, angular_radius, sun_scale") \
    X(LightSelection, "selected, light_count") \
    X(LightType, "selected, light_type, type_index") \
    X(AreaLightParameters, "radiance, light_pdf, distance2, area, cosine, light_count, selected") \
    X(BsdfSample, "wo.x, wo.y, wo.z, branch, u1, u2, color, metallic, roughness_or_alpha, wi.x, wi.y, wi.z, bsdf_pdf, weight, eta_scale, eta_i_over_t") \
    X(Throughput, "bsdf_weight, eta_scale") \
    X(Roulette, "rr_beta") \
    X(Continuation, "position.x, position.y, position.z, direction.x, direction.y, direction.z") \
    X(CameraRay, "origin.x, origin.y, origin.z, raw_direction.x, raw_direction.y, raw_direction.z, lens_weight") \
    X(MissContribution, "radiance, light_pdf") \
    X(FilmConversion, "xyz.x, xyz.y, xyz.z, rgb.r, rgb.g, rgb.b, lens_weight, wavelength_pdf") \
    X(SampleMean, "mean.r, mean.g, mean.b, normal.x, normal.y, normal.z, albedo.r, albedo.g, albedo.b") \
    X(TemporalMean, "film.r, film.g, film.b, normal.x, normal.y, normal.z, albedo.r, albedo.g, albedo.b") \
    X(RadianceSbt, "geometry_type, mesh_count, material_count") \
    X(RadianceUV, "has_diffuse_texture") \
    X(RadianceOpacity, "texture_alpha, opacity_factor, cutoff, uv.x, uv.y") \
    X(ShadowSbt, "geometry_type, mesh_count, material_count") \
    X(ShadowUV, "has_diffuse_texture") \
    X(ShadowOpacity, "texture_alpha, opacity_factor, cutoff, uv.x, uv.y") \
    X(AtmosphereGeometry, "status, ray_t_max, origin.x, origin.y, origin.z, direction.x, direction.y, direction.z") \
    X(AtmosphereIntegral, "status, begin_m, end_m, optical_depth, error_estimate, density_evaluations, wavelength_nm") \
    X(AtmosphereInverse, "status, uniform_24bit, distance_m, target_tau, optical_depth, residual_tau, error_estimate, inverse_iterations, density_evaluations") \
    X(AtmospherePhase, "direction.x, direction.y, direction.z, phase_value, phase_pdf, u_mu, u_phi") \
    X(AtmospherePosition, "status, candidates, ray_t, precise.x, precise.y, precise.z, rounded.x, rounded.y, rounded.z") \
    X(AtmosphereDirect, "radiance, sun_pdf, phase_value, transmittance, mis_weight") \
    X(AtmosphereContinuation, "position.x, position.y, position.z, direction.x, direction.y, direction.z, phase_value, phase_pdf") \
    X(PlanetIntersection, "ray_t, position.x, position.y, position.z, normal.x, normal.y, normal.z, tracking_status, root_height_m, root_height_bound_m, root_t_hi, root_t_lo, radial_correction_m, root_position.x, root_position.y, root_position.z") \
    X(AtmosphereCoefficient, "wavelength_nm, beta550") \
    X(AtmosphereTracking, "status, ray_t, begin_t, end_t, candidates, null_collisions, beta_lambda, wavelength_nm") \
    X(AtmosphereRatio, "status, begin_t, end_t, transmittance, candidates, roulette_tests, wavelength_nm") \
    X(RainTracking, "tracking_status, nf_status, ray_t, rayleigh_candidates, rain_candidates, rain_nulls, wavelength_nm") \
    X(RainRatio, "tracking_status, nf_status, transmittance, rayleigh_candidates, rain_candidates, rain_roulette, wavelength_nm") \
    X(RainScatter, "nf_status, operation, wavelength_nm, proposal_cosine, incident_cosine, value, pdf, weight, log_physical, log_proposal, csca_in, cext_camera, u0, u1") \
    X(RainDirect, "radiance, sun_pdf, value, proposal_pdf, transmittance, mis_weight")

enum class FailureReason : std::uint32_t {
#define M1_ENUM_REASON(name, labels) name,
    M1_FAILURE_REASONS(M1_ENUM_REASON)
#undef M1_ENUM_REASON
};

inline const char* failure_reason_name(FailureReason reason) {
    switch(reason) {
#define M1_NAME_REASON(name, labels) case FailureReason::name: return #name;
        M1_FAILURE_REASONS(M1_NAME_REASON)
#undef M1_NAME_REASON
    }
    return "UnknownReason";
}

inline const char* failure_value_labels(FailureReason reason) {
    switch(reason) {
#define M1_LABEL_REASON(name, labels) case FailureReason::name: return labels;
        M1_FAILURE_REASONS(M1_LABEL_REASON)
#undef M1_LABEL_REASON
    }
    return "unknown";
}
#undef M1_FAILURE_REASONS

inline const char* failure_stage_name(FailureStage stage) {
    switch(stage) {
    case FailureStage::Camera: return "camera";
    case FailureStage::Surface: return "surface";
    case FailureStage::SunNee: return "sun_nee";
    case FailureStage::AreaNee: return "area_nee";
    case FailureStage::Bsdf: return "bsdf";
    case FailureStage::Miss: return "miss";
    case FailureStage::Film: return "film";
    case FailureStage::RadianceAnyHit: return "radiance_anyhit";
    case FailureStage::ShadowAnyHit: return "shadow_anyhit";
    case FailureStage::Atmosphere: return "atmosphere";
    case FailureStage::AtmosphereSunNee: return "atmosphere_sun_nee";
    case FailureStage::Planet: return "planet";
    case FailureStage::Rain: return "rain";
    case FailureStage::RainSunNee: return "rain_sun_nee";
    }
    return "unknown";
}

struct FailureContext {
    std::uint32_t mesh = no_index;
    std::uint32_t material = no_index;
    std::uint32_t instance = no_index;
    std::uint32_t primitive = no_index;
};

struct FailureValues { float v[16] = {}; };

struct FailureRecord {
    FailureReason reason = FailureReason::None;
    FailureStage stage = FailureStage::Camera;
    FailureContext context{};
    std::uint32_t pixel_x = 0, pixel_y = 0, sample = 0;
    std::uint32_t frame = 0, seed = 0;
    std::int32_t bounce = 0;
    float wavelength_nm = 0, throughput = 0, radiance = 0;
    float bsdf_pdf = 0, eta_scale = 0;
    // Path PRD state, except ShadowAnyHit records the actual shadow trace ray.
    float current_position[3] = {};
    float current_direction[3] = {};
    FailureValues values{};
};

struct FailureBuffer {
    // Device writers compete only for this word using atomicCAS. The winning
    // thread owns first; the host reads it after cudaStreamSynchronize.
    std::uint32_t claimed = 0;
    FailureRecord first{};
};

static_assert(std::is_standard_layout<FailureBuffer>::value,
              "The diagnostic is shared verbatim by host and CUDA");
static_assert(std::is_trivially_copyable<FailureBuffer>::value,
              "CUDABuffer uploads/downloads the diagnostic as bytes");
static_assert(sizeof(FailureBuffer) == 160, "Unexpected diagnostic ABI");

} // namespace m1
