#ifndef LAUNCH_PARAMS_H_
#define LAUNCH_PARAMS_H_

#include "optix8.hpp"
#include "../utils/my_math.hpp"
#include <cuda_runtime.h>


// #define PHASE_FUNCTION_TABULATED // 0: HG, 1: Tabulated, 2: Neural
// #define PHASE_FUNCTION_HG // 0: HG, 1: Tabulated, 2: Neural
#define PHASE_FUNCTION_NEURAL // 0: HG, 1: Tabulated, 2: Neural


static constexpr uint32_t MASK_VOLUME  = 0x01u;
static constexpr uint32_t MASK_SURFACE = 0x02u;
static constexpr uint32_t MASK_ALL     = MASK_VOLUME | MASK_SURFACE;

// ray type
enum {
    RADIANCE_RAY_TYPE=0, 
    SHADOW_RAY_TYPE, 
    RAY_TYPE_COUNT
};

// material 
enum {
    MATERIAL_TYPE_DIFFUSE=0, 
    MATERIAL_TYPE_PRINCIPLED_BRDF, 
    MATERIAL_TYPE_GLASS, 
    MATERIAL_TYPE_LIGHT, 
    NUM_MATERIAL_TYPE
};


// camera model
enum {
    LENS_TYPE_PINHOLE=0, 
    LENS_TYPE_THIN_LENS, 
    NUM_LENS_TYPE
};

// BRDF, BSDF
enum {
    BXDF_TYPE_DIFFUSE_SAMPLE=0,
    BXDF_TYPE_DIFFUSE_EVAL,
    BXDF_TYPE_PRINCIPLED_SAMPLE,
    BXDF_TYPE_PRINCIPLED_EVAL,
    BXDF_TYPE_GLASS_SAMPLE,
    BXDF_TYPE_GLASS_EVAL,
    NUM_BXDF
};

// light
enum {
    LIGHT_TYPE_ENV_SPHERE=0, 
    LIGHT_TYPE_TRIANGLE, 
    LIGHT_TYPE_SKY, 
    NUM_LIGHT_TYPE
};

enum : unsigned int {
    HIT_KIND_VDB_ENTER = 0,
    HIT_KIND_VDB_INSIDE = 1
};


struct TriangleMeshSBTData {
    uint32_t meshIndex;
    uint32_t materialIndex;
};

struct VDBSBTData {
    uint32_t vdbIndex;
    uint32_t materialIndex;
};

enum class GeomType : uint32_t {
    Triangle    = 0,
    VDB         = 1
};

struct HitgroupSBTData {
    GeomType geomType;
    uint32_t pad0;
    union {
        TriangleMeshSBTData tri;
        VDBSBTData          vdb;
    };
};

struct TextureSlot {
    cudaTextureObject_t texture {0};
};

struct TriangleMeshGeomData {
    float3* vertex;
    float3* normal;
    float4* tangent;
    float2* texcoord;
    // float2* normalTexcoord;
    // float2* emissiveTexcoord;
    uint3*  index;

    bool hasTangent;
    bool hasNormal;
};

struct VDBGeomData {
    CUdeviceptr nanoGrid;
    CUdeviceptr macroMajorant;
    float densityScale;
    float emissionScale;
    float macroMaxDensity;
    int macroCellSizeVoxels;
    int4 macroBaseCoord;
    int4 macroDims;
};

struct MaterialData {
    unsigned int    materialType;
    float3  color;
    float   roughness;
    float   metallic;
    float3  emissive;

    TextureSlot diffuseTexture;
    TextureSlot normalTexture;
    TextureSlot rmTexture;
    TextureSlot emissiveTexture;
};

struct LightDefinition{
    int lightType;
    uint32_t lightIndexInType; // 三角形のインデックス or 環境マップのインデックス
};

struct TriangleLightData {
    float3 v0, v1, v2;
    float3 normal;
    float area;
    float3 constantEmission;

    // emissive テクスチャ用
    float2 uv0, uv1, uv2;
    TextureSlot emissiveTexture;
};

struct SpectralParams {
    alignas(8) cudaTextureObject_t xyzFunc[3];
    alignas(8) cudaTextureObject_t upSampleFunc[3];
    alignas(8) cudaTextureObject_t D65;

    alignas(8) cudaTextureObject_t wavelengthCdfTex = 0;

    float wavelengthMin     {390.0f};
    float wavelengthMax     {830.0f};

    const float* wavelengthPdf;
    // const float* wavelengthCdf;
    int         wavelengthBinCount;
    float       wavelengthBinWidth;
};

struct AtmosphereDeviceData {
    cudaTextureObject_t sunTransmittanceTex = 0;
    cudaTextureObject_t directIrradianceTex = 0;

    CUdeviceptr skyRayleighTexHandles = 0;
    CUdeviceptr skyMieTexHandles = 0;
    CUdeviceptr skyMultipleTexHandles = 0;

    CUdeviceptr wavelengthsNm = 0;

    uint32_t lambdaCount = 0;

    float bottomRadius_m = 0.0f;
    float topRadius_m = 0.0f;
    float observerAltitude_m = 0.0f;
    float muSMin = -0.2f;

    uint32_t skyNu = 0;
    uint32_t skyMu = 0;
    uint32_t skyMuS = 0;

    uint32_t irradianceR = 0;
    uint32_t irradianceMuS = 0;
};

struct alignas(16) LaunchParams {
    struct {
        // レンダリング結果を出力するバッファ
        float4* colorBuffer;
        float4* normalBuffer;
        float4* albedoBuffer;

        mymath::matrix3x4* objectMatrixBuffer;
        mymath::matrix3x3* normalMatrixBuffer;

        int2    size            {make_int2(720, 1280)};
        int     accumID         {0};
        int     numPixelSamples {1};
        int     maxBounce       {2048};
        int     frameID         {0};
    } frame;

    struct {
        // Extrinsic
        float3 position;
        float3 direction;
        float3 horizontal;
        float3 vertical;

        // Intrinsics
        float focalLength       {50.0f};
        float fValue            {2.8f};
        float pintDist          {1.0f};
        float sensitivity       {1.0f};
        float fov               {20.f};
        
        // camera mode
        int cameraMode          {LENS_TYPE_PINHOLE};
    } camera;

    struct {
        LightDefinition*    lightDefinition;
        TriangleLightData*  triangleLightData;
        int numLights                   {0};
        float lightIntensityFactor      {1.0f};
    } light;

    struct {
        bool    hasEnvMap       {false};
        float*  coarseMarginal;
        float*  coarseConditional;
        float*  patchWeight;
        float   totalWeight;
        int2    patchSize;
    } envMapInfo;
    
    struct {
        // RIS
        int risSampleNum        {8};
    } samplingStrategy;

    OptixTraversableHandle traversable;
    cudaTextureObject_t envMap;

    // spectral rendering
    SpectralParams spectral;

    TriangleMeshGeomData* meshes;
    int numMeshes       {0};
    
    VDBGeomData* vdbs {nullptr};
    int numVDBs         {0};
    
    MaterialData* materials;
    int numMaterials    {0};

    OptixAabb* vdbAABBs;

    struct {
        cudaTextureObject_t phaseParameterG;
        cudaTextureObject_t pdf;
        cudaTextureObject_t cdf;
        int numTheta;
        int numLambda;
        int numDiameter;
    } mieTexture;

    // ---- NSF / CoopVec packed weights ----
    static constexpr uint32_t NSF_MAX_TRANSFORMS = 2;
    static constexpr uint32_t NSF_LAYERS_PER_TRANSFORM = 3;
    static constexpr uint32_t NSF_LAYERS_PER_TRANSFORM_PADD = 4;

    struct alignas(16) {
        CUdeviceptr packedBase;   // m_nsfPackedWeights.getDevicePointer()
        uint32_t    packedBytes;  // total bytes (debug/sanity)
        uint32_t    transforms;   // e.g. 3
        uint32_t    inputPad;     // e.g. 16
        uint32_t    bins;         // e.g. 32 (optional but useful)
        uint32_t    hidden;       // e.g. 64 (optional)
        uint32_t    context;      // e.g. 3  (optional)
        uint32_t    phiDim;       // e.g. 31 (optional)

        // offsets in bytes from packedBase
        alignas(16) uint32_t wOffset[NSF_MAX_TRANSFORMS][NSF_LAYERS_PER_TRANSFORM];
        alignas(16) uint32_t bOffset[NSF_MAX_TRANSFORMS][NSF_LAYERS_PER_TRANSFORM];

        // (optional) dims after padding (device側で assert/汎用化に使える)
        alignas(16) uint16_t N[NSF_MAX_TRANSFORMS][NSF_LAYERS_PER_TRANSFORM];
        alignas(16) uint16_t K[NSF_MAX_TRANSFORMS][NSF_LAYERS_PER_TRANSFORM];
    } nsf;

    AtmosphereDeviceData atmo;

    struct {
        float sunZenithRad;        // +Y からの天頂角 [0, pi]
        float sunAzimuthRad;       // 水平面での方位角 [0, 2pi)
    } sunParams;
};

#endif // LAUNCH_PARAMS_H_