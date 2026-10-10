#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include <stdio.h>

extern "C" __device__ float3 __direct_callable__brdf_principled_sample_rgb(const float3 wo, const IntersectedData_RGB& material, PRDRGB* prd)
{
    float3 wi;
    float3 wm;
    const float roughness = material.roughness;
    const float alpha = fmaxf(roughness * roughness, 1e-3f);
    const float metallic = material.metallic;
    const float weightDiffuse = fmaxf(1.0f - metallic, 0.f);
    const float weightSpecular = 1.0f;

    // MIS
    const float weightSum = weightDiffuse + weightSpecular;
    const float normalizedWeightDiffuse  = weightDiffuse / weightSum;
    const float normalizedWeightSpecular = weightSpecular / weightSum;

    float pdf;
    if(normalizedWeightDiffuse > prd->random()){
        // diffuse
        wi = randomCosineHemisphere(prd->random(), prd->random());
    } else {
        // specular
        wm = visibleNormalSampling(alpha, wo, prd->random(), prd->random());
        wi = normalize(-1.0f * wo + 2.0f * dot(wo, wm) * wm);
    }

    const float diffusePdf = getLambertPdf(wo, wi);
    const float specularPdf = getGGXPdf(alpha, wo, wi);

    pdf = normalizedWeightDiffuse * diffusePdf + normalizedWeightSpecular * specularPdf;
    // printf("Diffuse PDF %f, Specular PDF: %f \n", diffusePdf, specularPdf);
    

    prd->lastHitMaterialType = MATERIAL_TYPE_PRINCIPLED_BRDF;
    prd->pdf.bxdf = pdf;
    return wi;

}

extern "C" __device__ float3 __direct_callable__brdf_principled_eval_rgb(const float3 wi, const float3 wo, const IntersectedData_RGB& material, PRDRGB* prd)
{

    // NaN を発生させない
    if(wi.y < 1e-7f){
        return make_float3(0.0f);
    }
    
    const float roughness = material.roughness;
    const float alpha = fmaxf(roughness * roughness, 1e-3f);
    const float metallic = material.metallic;
    const float3 baseColor = material.baseColor;
    const float weightDiffuse = fmaxf(1.0f - metallic, 0.f);
    const float weightSpecular = 1.0f;

    // MIS
    const float weightSum = weightDiffuse + weightSpecular;
    const float normalizedWeightDiffuse  = weightDiffuse / weightSum;
    const float normalizedWeightSpecular = weightSpecular / weightSum;

    const float diffusePdf = getLambertPdf(wo, wi);
    const float specularPdf = getGGXPdf(alpha, wo, wi);

    float pdf = normalizedWeightDiffuse * diffusePdf + normalizedWeightSpecular * specularPdf;

    prd->pdf.bxdf = pdf;

    const float3 F0 = lerp(make_float3(0.04f), baseColor, metallic);

    const float3 diffuseColor = evalLambertBRDF_RGB(baseColor, wo, wi);
    const float3 specularColor = evalSpecularBRDF_RGB(alpha, F0, wo, wi);

    prd->lastHitMaterialType = MATERIAL_TYPE_PRINCIPLED_BRDF;
    float3 color = diffuseColor * (1.0f - metallic) + specularColor;
    return color;
}


extern "C" __device__ float3 __direct_callable__brdf_principled_sample_spectral(const float3 wo, const IntersectedData_Spectral& material, PRDSpectral* prd)
{
    float3 wi;
    float3 wm;
    const float roughness = material.roughness;
    const float alpha = fmaxf(roughness * roughness, 1e-3f);
    const float metallic = material.metallic;
    const float weightDiffuse = fmaxf(1.0f - metallic, 0.f);
    const float weightSpecular = 1.0f;

    // MIS
    const float weightSum = weightDiffuse + weightSpecular;
    const float normalizedWeightDiffuse  = weightDiffuse / weightSum;
    const float normalizedWeightSpecular = weightSpecular / weightSum;

    float pdf;
    if(normalizedWeightDiffuse > prd->random()){
        // diffuse
        wi = randomCosineHemisphere(prd->random(), prd->random());
    } else {
        // specular
        wm = visibleNormalSampling(alpha, wo, prd->random(), prd->random());
        wi = normalize(-1.0f * wo + 2.0f * dot(wo, wm) * wm);
    }

    const float diffusePdf = getLambertPdf(wo, wi);
    const float specularPdf = getGGXPdf(alpha, wo, wi);

    pdf = normalizedWeightDiffuse * diffusePdf + normalizedWeightSpecular * specularPdf;
    // printf("Diffuse PDF %f, Specular PDF: %f \n", diffusePdf, specularPdf);
    

    // prd->lastHitMaterialType = MATERIAL_TYPE_PRINCIPLED_BRDF;
    prd->pdf.bxdf = pdf;
    return wi;

}

extern "C" __device__ float __direct_callable__brdf_principled_eval_spectral(const float3 wi, const float3 wo, const IntersectedData_Spectral& material, PRDSpectral* prd)
{

    // NaN を発生させない
    if(wi.y < 1e-7f){
        return 0.0f;
    }
    
    const float roughness = material.roughness;
    const float alpha = fmaxf(roughness * roughness, 1e-3f);
    const float metallic = material.metallic;
    const float baseColor = material.baseColor;
    const float weightDiffuse = fmaxf(1.0f - metallic, 0.f);
    const float weightSpecular = 1.0f;

    // MIS
    const float weightSum = weightDiffuse + weightSpecular;
    const float normalizedWeightDiffuse  = weightDiffuse / weightSum;
    const float normalizedWeightSpecular = weightSpecular / weightSum;

    const float diffusePdf = getLambertPdf(wo, wi);
    const float specularPdf = getGGXPdf(alpha, wo, wi);

    float pdf = normalizedWeightDiffuse * diffusePdf + normalizedWeightSpecular * specularPdf;

    prd->pdf.bxdf = pdf;

    const float F0 = lerp(0.04f, baseColor, metallic);

    const float diffuseColor = evalLambertBRDF_Spectral(baseColor, wo, wi);
    const float specularColor = evalSpecularBRDF_Spectral(alpha, F0, wo, wi);

    // prd->lastHitMaterialType = MATERIAL_TYPE_PRINCIPLED_BRDF;
    float color = diffuseColor * (1.0f - metallic) + specularColor;
    return color;
}