#include "renderer.hpp"
#include "texture.hpp"
#include <optix_function_table_definition.h>
#include <optix_types.h>
// #include <filesystem>
// #include <iostream>
// #include <fstream>
#include <algorithm>
#include <mutex>
#include <string.h>


#define STB_IMAGE_IMPLEMENTATION
#include "ptx_data_reader.hpp"
#include "stb_image.h"
#include "my_debug_tool.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

#if defined(_WIN32)
#include <windows.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#else
#include <unistd.h>
#endif

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) CallableRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void* data;
};

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    HitgroupSBTData data;
};


Renderer::Renderer(std::vector<const Model*> models, sceneIO::Scene sceneDesc) : m_models(models), m_sceneDesc(sceneDesc)
{
    m_optixModuleFileNames.resize(static_cast<int>(OptixModuleIdentifier::NUM_OPTIX_MODULE_IDENTIFIERS));
#if defined(USE_OPTIX_IR)
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_RAYGEN)]             = std::string("raygen.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_AH_RADIANCE)]        = std::string("ah_radiance.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_AH_SHADOW)]          = std::string("ah_shadow.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_CH_RADIANCE)]        = std::string("ch_radiance.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_CH_SHADOW)]          = std::string("ch_shadow.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_MISS_RADIANCE)]      = std::string("miss_radiance.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_MISS_SHADOW)]        = std::string("miss_shadow.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_DIFFUSE)]       = std::string("diffuse_brdf.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_PRINCIPLED)]    = std::string("disney_principled_brdf.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_GLASS)]         = std::string("glass_bsdf.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LIGHTSAMPLE)]        = std::string("light_sample.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LENS)]               = std::string("lens.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_IS_VDB)]             = std::string("intersection_vdb.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_CH_VDB_RADIANCE)]    = std::string("ch_radiance_vdb.optixir");
#else
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_RAYGEN)]             = std::string("raygen.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_AH_RADIANCE)]        = std::string("ah_radiance.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_AH_SHADOW)]          = std::string("ah_shadow.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_CH_RADIANCE)]        = std::string("ch_radiance.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_CH_SHADOW)]          = std::string("ch_shadow.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_MISS_RADIANCE)]      = std::string("miss_radiance.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_MISS_SHADOW)]        = std::string("miss_shadow.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_DIFFUSE)]       = std::string("diffuse_brdf.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_PRINCIPLED)]    = std::string("disney_principled_brdf.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_GLASS)]         = std::string("glass_bsdf.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LIGHTSAMPLE)]        = std::string("light_sample.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LENS)]               = std::string("lens.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_IS_VDB)]             = std::string("intersection_vdb.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_CH_VDB_RADIANCE)]    = std::string("ch_radiance_vdb.ptx");
#endif
    
    m_cudaModuleFileNames.resize(static_cast<int>(PostProcessCudaModuleIdentifier::NUM_CUDA_MODULE_IDENTIFIERS));
    m_cudaModuleFileNames[static_cast<int>(PostProcessCudaModuleIdentifier::CUDA_MODULE_ID_TONEMAP)] = std::string("tonemap.ptx");


    m_isAccumulate = m_sceneDesc.integrator.isAccumulate;
    m_launchParams.frame.maxBounce = m_sceneDesc.integrator.maxBounce;
    m_launchParams.frame.numPixelSamples = m_sceneDesc.integrator.spp;

    initOptix();
    
    std::cout << "# Atmospheric RT: creating OptiX context..." << std::endl;
    createContext();

    std::cout << "# Atmospheric RT: setting up OptiX module..." << std::endl;
    createOptiXModule();

    std::cout << "# Atmospheric RT: creating raygen programs ..." << std::endl;
    createRaygenPrograms();
    std::cout << "# Atmospheric RT: creating miss programs ..." << std::endl;
    createMissPrograms();
    std::cout << "# Atmospheric RT: creating callable programs ..." << std::endl;
    createCallablePrograms();
    std::cout << "# Atmospheric RT: creating hitgroup programs ..." << std::endl;
    createHitgroupPrograms();

    std::cout << "# Atmospheric RT: setting up OptiX pipeline ..." << std::endl;
    createPipeline();
    
    
    std::cout << "# Atmospheric RT: load VDB data..." << std::endl;
    loadVDB();

    std::cout << "# Atmospheric RT: Building accelelation structure..." << std::endl;
    buildAccel();
    m_launchParams.traversable = m_iasHandle;

    std::cout << "# Atmospheric RT: Loading assets to GPU..." << std::endl;
    loadAssets();

    std::cout << "# Atmospheric RT: building shader binding table..." << std::endl;
    buildSBT();

    createTextures();

    loadAtmosphere();
    
    m_launchParamsBuffer.alloc(sizeof(m_launchParams)); 
    std::cout << "# Atmospheric RT: context, module, pipeline, etc, all set up ..." << std::endl;
    std::cout << "# Atmospheric RT: Optix 8 fully set up..." << std::endl;

    
    std::cout << "# Atmospheric RT: setting up CUDA module..." << std::endl;
    createCUDAModule();

    std::cout << "# Atmospheric RT: create light table..." << std::endl;
    createLightTable();

    std::cout << "# Atmospheric RT: uploading spectrum data..." << std::endl;
    uploadSpectrumData();

    std::cout << "# Atmospheric RT: uploading Mie data..." << std::endl;
    loadMieData();

#if defined(PHASE_FUNCTION_NEURAL)
    std::cout << "# Atmospheric RT: uploading Network Weights..." << std::endl;
    buildNsfPackedWeightsCoopVec();
#endif
    std::cout << "# Atmospheric RT: CUDA kernel fully set up..." << std::endl;


}


void Renderer::createTextures()
{
    int numTextures = 0;
    for(auto* mdl : m_models) numTextures +=(int)mdl->textures.size();
  
    m_textureArrays.resize(numTextures);
    m_textureObjects.resize(numTextures);

    size_t flat = 0;
    for (auto mdl : m_models){
        for(int textureID = 0; textureID < (int)mdl->textures.size(); ++textureID, ++flat){
            auto texture = mdl->textures[textureID];

            cudaResourceDesc resDesc = {};

            cudaChannelFormatDesc channelDesc;
            int32_t width = texture->resolution.x;
            int32_t height = texture->resolution.y;
            int32_t numComponents = 4; //rgba
            int32_t pitch = width * numComponents * sizeof(uint8_t);
            channelDesc = cudaCreateChannelDesc<uchar4>();

            cudaArray_t &pixelArray = m_textureArrays[flat];
            CUDA_CHECK(cudaMallocArray(&pixelArray, &channelDesc, width, height));
            CUDA_CHECK(cudaMemcpy2DToArray(pixelArray, 0, 0, texture->pixel, pitch, pitch, height, cudaMemcpyHostToDevice));
            
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = pixelArray;

            // テクスチャのふるまいを決定
            cudaTextureDesc texDesc = {};                               
            texDesc.addressMode[0]      = cudaAddressModeWrap;          // テクスチャが範囲外になったときの対処法 (タイリング)
            texDesc.addressMode[1]      = cudaAddressModeWrap;          // テクスチャが範囲外になったときの対処法 (タイリング)
            texDesc.filterMode          = cudaFilterModeLinear;         // テクセルの補間方法
            texDesc.readMode            = cudaReadModeNormalizedFloat;  // GPU 側で読み込むときのフォーマット (0-1)
            texDesc.normalizedCoords    = 1;                            // テクスチャ座標を [0,1] 範囲で指定
            texDesc.maxAnisotropy       = 1;                            // 異方性フィルタリングの強度 (1:無効)
            texDesc.maxMipmapLevelClamp = 99;                           // ミップレベルの最大
            texDesc.minMipmapLevelClamp = 0;                            // ミップレベルの最小
            texDesc.mipmapFilterMode    = cudaFilterModeLinear;         // ミップマップ間の補間
            texDesc.borderColor[0]      = 1.0f;                         // 範囲外で指定される色
            if(texture->isDiffuseTexture){
                texDesc.sRGB            = 1;                            // 1: 読み込む画像は sRGB 画像と考え，ガンマ補正を適用して読み込み
            } else {
                texDesc.sRGB            = 0;                            // そのまま読み込み
            }

            cudaTextureObject_t cudaTex = 0;
            CUDA_CHECK(cudaCreateTextureObject(&cudaTex, &resDesc, &texDesc, nullptr));
            m_textureObjects[flat] = cudaTex;
        }
    }
}

bool Renderer::buildAccel()
{
    PING;
    std::cout << "Building GAS..." << std::endl;
    int numMeshes = 0;
    for(auto* mdl : m_models) {
        numMeshes +=(int)mdl->meshes.size();
        PRINT(mdl->meshes.size());
    }

    bool hasVDB = m_hasVDB;
    const int numVDBInstances = hasVDB ? 1 : 0;
    const int numInstances = numMeshes + numVDBInstances;

    m_vertexBuffer.resize(numMeshes);
    m_indexBuffer.resize(numMeshes);
    m_normalBuffer.resize(numMeshes);
    m_diffuseTexcoordBuffer.resize(numMeshes);
    m_normalTexcoordBuffer.resize(numMeshes);
    m_emissiveTexcoordBuffer.resize(numMeshes);
    m_tangentBuffer.resize(numMeshes);
    m_colorBuffer.resize(numMeshes);
    
    m_GASBuffer.resize(numMeshes);
    m_gasHandle.resize(numMeshes);
  
    // ===========================
    // メッシュの GAS (BLAS) を構築　（メッシュごとに 1 つ）
    // ===========================
    std::vector<mymath::matrix3x4> objectMatrix(numMeshes);
    std::vector<mymath::matrix3x3> normalMatrix(numMeshes);
    size_t flat = 0;
    for(auto* mdl : m_models) {
        mymath::matrix3x4 mat = mdl->modelMatrix;
        for(int meshID = 0; meshID < (int)mdl->meshes.size(); ++meshID, ++flat){

            objectMatrix[flat] = mat;
            normalMatrix[flat] = linear3x3(mat);
        
            // 三角形の入力
            OptixBuildInput     triangleInput;
            CUdeviceptr         d_vertices;
            CUdeviceptr         d_indices;
            uint32_t            triangleInputFlags;

            TriangleMesh &mesh = *mdl->meshes[meshID];
            m_vertexBuffer[flat].allocAndUpload(mesh.vertex);
            m_indexBuffer[flat].allocAndUpload(mesh.index);
            if(!mesh.normal.empty()){
                m_normalBuffer[flat].allocAndUpload(mesh.normal);
            }
            if(!mesh.diffuseTexcoord.empty()){
                m_diffuseTexcoordBuffer[flat].allocAndUpload(mesh.diffuseTexcoord);
            }
            if(!mesh.normalTexcoord.empty()){
                m_normalTexcoordBuffer[flat].allocAndUpload(mesh.normalTexcoord);
            }
            if(!mesh.emissiveTexcoord.empty()){
                m_emissiveTexcoordBuffer[flat].allocAndUpload(mesh.emissiveTexcoord);
            }
            if(!mesh.tangent.empty()){
                m_tangentBuffer[flat].allocAndUpload(mesh.tangent);
            }


            triangleInput = {}; // ここに情報を入れていく
            triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            d_vertices = m_vertexBuffer[flat].getDevicePointer();
            d_indices = m_indexBuffer[flat].getDevicePointer();

            // 頂点情報
            triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
            triangleInput.triangleArray.numVertices         = (int)mesh.vertex.size();
            triangleInput.triangleArray.vertexBuffers       = &d_vertices;

            // 頂点のインデックス情報
            triangleInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangleInput.triangleArray.indexStrideInBytes  = sizeof(uint3);
            triangleInput.triangleArray.numIndexTriplets    = (int)mesh.index.size();
            triangleInput.triangleArray.indexBuffer         = d_indices;

            triangleInputFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;  // 特別な設定を行わない
            // MEMO: 
            // OPTIX_GEOMETRY_FLAG_NONE;  // 特別な設定を行わない
            // OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT : any_hit シェーダを無効化
            // OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL : 1回だけ any-hit をよぶ
            // OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING :  バックフェースカリングを無効化　
            //                                                      （薄いマテリアルをレンダリングしたいときにたてる）

            // 単一の SBT レコードを使用
            triangleInput.triangleArray.flags                       = &triangleInputFlags;
            triangleInput.triangleArray.numSbtRecords               = 1;        // 単一の SBT レコード
            // 単一の場合，未使用なので 0 
            triangleInput.triangleArray.sbtIndexOffsetBuffer        = 0;        
            triangleInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
            triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;


            // GAS のセットアップ
            OptixAccelBuildOptions  accelOptions    = {};
            accelOptions.buildFlags                 = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            if(getNumDevices() == 1){
                // 複数 GPU を使うと性能が悪化する恐れがあるため，単一 GPU の場合のみオプションを追加
                accelOptions.buildFlags             |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            }
            accelOptions.motionOptions.numKeys      = 1;    // モーションブラーなし． numKeys > 1 でブラー補間
            accelOptions.operation                  = OPTIX_BUILD_OPERATION_BUILD; // 新規構築． ..._UPDATE を使うと更新


            // AS に必要なバッファサイズの見積もり
            OptixAccelBufferSizes gasBufferSizes;
            OPTIX_CHECK(
                optixAccelComputeMemoryUsage(
                    m_optixContext,
                    &accelOptions,
                    &triangleInput,
                    1,              // メッシュの個数．今回は1個ずつ作成しているので1
                    &gasBufferSizes
                )
            );

            // Compaction の準備
            // AS を最悪のケースを想定して作るので，作成後に不要となった部分を圧縮することで VRAM を節約可能
            
            CUDABuffer compactedSizeBuffer;
            compactedSizeBuffer.alloc(sizeof(uint64_t));

            OptixAccelEmitDesc emitDesc;                        // AS 構築時の圧縮後サイズ，AABB の範囲，インスタンスの変換行列... を出力してくれる補助出力の構造体
            emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE; // 圧縮後のサイズを返すように指定
            emitDesc.result = compactedSizeBuffer.getDevicePointer();

            // GAS の構築
            CUDABuffer tempBuffer;
            tempBuffer.alloc(gasBufferSizes.tempSizeInBytes);

            CUDABuffer outputBuffer;
            outputBuffer.alloc(gasBufferSizes.outputSizeInBytes);

            OPTIX_CHECK(
                optixAccelBuild(
                    m_optixContext,
                    0, // stream
                    &accelOptions,
                    &triangleInput,
                    1,              // メッシュの個数．今回は1個ずつ作成しているので1
                    
                    // temp
                    tempBuffer.getDevicePointer(),
                    tempBuffer.getSizeInBytes(),

                    // output
                    outputBuffer.getDevicePointer(),
                    outputBuffer.getSizeInBytes(),

                    &m_gasHandle[flat],

                    &emitDesc, 1
                )
            );

            CUDA_SYNC_CHECK();
            // Compaction の実行
            uint64_t compactedSize;
            compactedSizeBuffer.download(&compactedSize, 1);
            m_GASBuffer[flat].alloc(compactedSize);
            OPTIX_CHECK(
                optixAccelCompact(
                    m_optixContext,
                    0,
                    m_gasHandle[flat],
                    m_GASBuffer[flat].getDevicePointer(),
                    m_GASBuffer[flat].getSizeInBytes(),
                    &m_gasHandle[flat]
                )
            );

            CUDA_SYNC_CHECK();

            // クリーンアップ
            outputBuffer.free();
            tempBuffer.free();
            compactedSizeBuffer.free();
        }
    }

    if(numMeshes > 0){
        m_objectMatrix.allocAndUpload(objectMatrix);
        m_normalMatrix.allocAndUpload(normalMatrix);
        m_launchParams.frame.objectMatrixBuffer = (mymath::matrix3x4*)m_objectMatrix.getDevicePointer();
        m_launchParams.frame.normalMatrixBuffer = (mymath::matrix3x3*)m_normalMatrix.getDevicePointer();
    } else {
        m_launchParams.frame.objectMatrixBuffer = nullptr;
        m_launchParams.frame.normalMatrixBuffer = nullptr;
    }
    

    // ===========================
    // VDBの GAS (BLAS) を構築
    // ===========================
    if(hasVDB){
        OptixBuildInput vdbInput = {};
        vdbInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

        CUdeviceptr d_aabb = m_vdbAABBBuffer.getDevicePointer();
        vdbInput.customPrimitiveArray.aabbBuffers = &d_aabb;
        vdbInput.customPrimitiveArray.numPrimitives = 1;

        uint32_t vdbInputFlags = OPTIX_GEOMETRY_FLAG_NONE;
        vdbInput.customPrimitiveArray.flags                         = & vdbInputFlags;
        vdbInput.customPrimitiveArray.numSbtRecords                 = 1;
        vdbInput.customPrimitiveArray.sbtIndexOffsetBuffer          = 0;
        vdbInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes     = 0;
        vdbInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes   = 0;

        OptixAccelBuildOptions vdbAccelOptions = {};
        vdbAccelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        if(getNumDevices() == 1){
            // 複数 GPU を使うと性能が悪化する恐れがあるため，単一 GPU の場合のみオプションを追加
            vdbAccelOptions.buildFlags             |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        }
        vdbAccelOptions.motionOptions.numKeys      = 1;    // モーションブラーなし． numKeys > 1 でブラー補間
        vdbAccelOptions.operation                  = OPTIX_BUILD_OPERATION_BUILD; // 新規構築． ..._UPDATE を使うと更新

        // AS に必要なバッファサイズの見積もり
        OptixAccelBufferSizes vdbGasBufferSizes;
        OPTIX_CHECK(
            optixAccelComputeMemoryUsage(
                m_optixContext,
                &vdbAccelOptions,
                &vdbInput,
                1,              // メッシュの個数．今回は1個ずつ作成しているので1
                &vdbGasBufferSizes
            )
        );

        // Compaction の準備
        // AS を最悪のケースを想定して作るので，作成後に不要となった部分を圧縮することで VRAM を節約可能
        
        CUDABuffer compactedSizeVDBBuffer;
        compactedSizeVDBBuffer.alloc(sizeof(uint64_t));

        OptixAccelEmitDesc vdbEmitDesc = {};                   // AS 構築時の圧縮後サイズ，AABB の範囲，インスタンスの変換行列... を出力してくれる補助出力の構造体
        vdbEmitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE; // 圧縮後のサイズを返すように指定
        vdbEmitDesc.result = compactedSizeVDBBuffer.getDevicePointer();

        // GAS の構築
        CUDABuffer tempVDBGASBuffer;
        tempVDBGASBuffer.alloc(vdbGasBufferSizes.tempSizeInBytes);

        CUDABuffer outputVDBGASBuffer;
        outputVDBGASBuffer.alloc(vdbGasBufferSizes.outputSizeInBytes);

        OPTIX_CHECK(
            optixAccelBuild(
                m_optixContext,
                0, // stream
                &vdbAccelOptions,
                &vdbInput,
                1,              // メッシュの個数．今回は1個ずつ作成しているので1
                
                // temp
                tempVDBGASBuffer.getDevicePointer(),
                tempVDBGASBuffer.getSizeInBytes(),

                // output
                outputVDBGASBuffer.getDevicePointer(),
                outputVDBGASBuffer.getSizeInBytes(),

                &m_vdbGASHandle,

                &vdbEmitDesc, 1
            )
        );

        CUDA_SYNC_CHECK();
        // Compaction の実行
        uint64_t compactedVDBBufferSize = 0;
        compactedSizeVDBBuffer.download(&compactedVDBBufferSize, 1);
        m_vdbGASBuffer.alloc(compactedVDBBufferSize);
        OPTIX_CHECK(
            optixAccelCompact(
                m_optixContext,
                0,
                m_vdbGASHandle,
                m_vdbGASBuffer.getDevicePointer(),
                m_vdbGASBuffer.getSizeInBytes(),
                &m_vdbGASHandle
            )
        );

        CUDA_SYNC_CHECK();

        // クリーンアップ
        outputVDBGASBuffer.free();
        tempVDBGASBuffer.free();
        compactedSizeVDBBuffer.free();
    }
    
    // ==========================
    // IAS (TLAS) の構築
    // ==========================

    std::cout << "Building IAS..." << std::endl;

    if (numInstances == 0) {
        m_iasHandle = 0;
        return true;
    }

    std::vector<OptixInstance> instances(numInstances);

    // mesh instances [0, ..., numMeshes - 1]
    for(unsigned int meshID = 0; meshID < numMeshes; meshID++){
        OptixInstance & inst = instances[meshID];
        memset(&inst, 0, sizeof(OptixInstance));

        mymath::matrix3x4 mat = objectMatrix[meshID];
        float transform[12] = {
            mat.row0.x, mat.row0.y, mat.row0.z, mat.row0.w, 
            mat.row1.x, mat.row1.y, mat.row1.z, mat.row1.w, 
            mat.row2.x, mat.row2.y, mat.row2.z, mat.row2.w, 
        };

        memcpy(inst.transform, transform, sizeof(float) * 12);

        inst.instanceId     = meshID;
        inst.sbtOffset      = meshID * RAY_TYPE_COUNT;
        inst.visibilityMask = (unsigned int)MASK_SURFACE;
        inst.flags          = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;     // インスタンスの挙動を指定．
                                                            // OPTIX_INSTANCE_FLAG_NONE:                 デフォルト
                                                            // OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM:    transform 行列を無視 (ワールド座標に直置き)
                                                            // OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT:       AnyHit シェーダを無視
                                                            // OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT:       AnyHit シェーダを必ず呼ぶ
        inst.traversableHandle  = m_gasHandle[meshID];
    }

    // vdb instance
    if(hasVDB) {
        const int vdbInstanceID = numMeshes;
        OptixInstance & inst = instances[vdbInstanceID];
        memset(&inst, 0, sizeof(OptixInstance));
        float transform[12] = {
            1.0f, 0.0f, 0.0f, 0.0f, 
            0.0f, 1.0f, 0.0f, 0.0f, 
            0.0f, 0.0f, 1.0f, 0.0f, 
        };
        memcpy(inst.transform, transform, sizeof(float) * 12);
        inst.instanceId     = vdbInstanceID;
        inst.sbtOffset      = vdbInstanceID * RAY_TYPE_COUNT;
        inst.visibilityMask = (unsigned int)MASK_VOLUME;
        inst.flags          = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;     // インスタンスの挙動を指定．
                                                            // OPTIX_INSTANCE_FLAG_NONE:                 デフォルト
                                                            // OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM:    transform 行列を無視 (ワールド座標に直置き)
                                                            // OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT:       AnyHit シェーダを無視
                                                            // OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT:       AnyHit シェーダを必ず呼ぶ
        inst.traversableHandle  = m_vdbGASHandle;
    }

    m_instance.allocAndUpload(instances);

    OptixBuildInput instanceInput = {};
    instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceInput.instanceArray.instances    = m_instance.getDevicePointer();
    instanceInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());


    // IAS のセットアップ
    
    OptixAccelBuildOptions  accelOptions    = {};
    accelOptions.buildFlags                 = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    if(getNumDevices() == 1){
        // 複数 GPU を使うと性能が悪化する恐れがあるため，単一 GPU の場合のみオプションを追加
        accelOptions.buildFlags             |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    }
    accelOptions.motionOptions.numKeys      = 1;    // モーションブラーなし． numKeys > 1 でブラー補間
    accelOptions.operation                  = OPTIX_BUILD_OPERATION_BUILD; // 新規構築． ..._UPDATE を使うと更新
    
    // AS に必要なバッファサイズの見積もり
    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(
        optixAccelComputeMemoryUsage(
            m_optixContext,
            &accelOptions,
            &instanceInput,
            1,              // メッシュの個数．今回は1個ずつ作成しているので1
            &iasBufferSizes
        )
    );

    // Compaction の準備
    // IAS を最悪のケースを想定して作るので，作成後に不要となった部分を圧縮することで VRAM を節約可能
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;                        // IAS 構築時の圧縮後サイズ，AABB の範囲，インスタンスの変換行列... を出力してくれる補助出力の構造体
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE; // 圧縮後のサイズを返すように指定
    emitDesc.result = compactedSizeBuffer.getDevicePointer();

    // IAS の構築
    CUDABuffer tempBuffer;
    tempBuffer.alloc(iasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(iasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(
        optixAccelBuild(
            m_optixContext,
            0, // stream
            &accelOptions,
            &instanceInput,
            1,         
            
            // temp
            tempBuffer.getDevicePointer(),
            tempBuffer.getSizeInBytes(),

            // output
            outputBuffer.getDevicePointer(),
            outputBuffer.getSizeInBytes(),

            &m_iasHandle,

            &emitDesc, 1
        )
    );

    CUDA_SYNC_CHECK();
    
    // Compaction の実行
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    m_IASBuffer.alloc(compactedSize);
    OPTIX_CHECK(
        optixAccelCompact(
            m_optixContext,
            0,
            m_iasHandle,
            m_IASBuffer.getDevicePointer(),
            m_IASBuffer.getSizeInBytes(),
            &m_iasHandle
        )
    );

    CUDA_SYNC_CHECK();

    // クリーンアップ
    outputBuffer.free();
    tempBuffer.free();
    compactedSizeBuffer.free();

    return true;
}

// OptiX の初期化とエラーチェック
void Renderer::initOptix()
{
    std::cout << "Initializing optix ..." << std::endl;

    // CUDA を使える GPU が搭載されているかどうか確認
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if(numDevices == 0){
        throw std::runtime_error("# no CUDA capable devices are found :(");
    }

    std::cout << "found" << numDevices << "CUDA devices" << std::endl;

    // Optix の初期化
    OPTIX_CHECK(optixInit());
    std::cout << "Successfully initialised OptiX!" << std::endl;

    setNumDevices(numDevices);
}

//callback 関数
static void contextLogCallback( unsigned int level,
                            const char *tag,
                            const char *message,
                            void *)
{
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

void Renderer::createContext()
{
    const int deviceID = 0;
    CUDA_CHECK(cudaSetDevice(deviceID));
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    cudaGetDeviceProperties(&m_deviceProps, deviceID);
    std::cout << "Running on device: " << m_deviceProps.name << std::endl;

    OptixDeviceContextOptions options = {};
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;

    CUresult cuRes = cuCtxGetCurrent(&m_cudaContext);
    if(cuRes != CUDA_SUCCESS){
        fprintf(stderr, "ERROR querying current context: error code %d\n", cuRes);
    }

    OPTIX_CHECK(
        optixDeviceContextCreate(
            m_cudaContext, 
            &options,              // OptixDeviceContextOption を省略し，デフォルト設定を使用
            &m_optixContext // 生成されたコンテキストをここに出力，格納
        )
    );
    

    // OptixDeviceContextOptions options = {};
    // options.logCallbackFunction = contextLogCallback;
    // options.


    OPTIX_CHECK(
        optixDeviceContextSetLogCallback (
            m_optixContext,         // 対象のコンテキスト
            contextLogCallback,     // ログを受け取るコールバック関数
            nullptr,                // コールバックされるデータ
            4                       // ログレベル (4 が最も詳細)
        )
    );
}

void Renderer::createOptiXModule()
{
    m_moduleCompileOptions                      = {};
    // m_moduleCompileOptions.maxRegisterCount     = 40;
    // m_moduleCompileOptions.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_moduleCompileOptions.optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    m_moduleCompileOptions.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    
    m_pipelineCompileOptions                        = {};
    m_pipelineCompileOptions.traversableGraphFlags  =                               // 設計した AS に合わせて最小限のフラグを使うのが望ましい
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;                 // トップレベル IAS + GAS
    m_pipelineCompileOptions.usesMotionBlur         = false;                        // モーションブラーなし
    m_pipelineCompileOptions.numPayloadValues       = 2;                            // payload のスロット数．
    m_pipelineCompileOptions.numAttributeValues     = 2;                            // ヒットしたプリミティブから返される補助情報数．例えば，重心座標 (u, v)
                                                                                    // プリミティブの中で，必要な attr の最大個数を指定すればよい．
    m_pipelineCompileOptions.exceptionFlags         =                               // 例外処理を有効化するフラグ
        OPTIX_EXCEPTION_FLAG_NONE;                                             
        // OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW                                             // コールスタックが溢れた場合の例外
        // | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH                                              // 最大再帰深度を超えた場合に発生する例外をキャッチ
        // | OPTIX_EXCEPTION_FLAG_USER;                                                    // 
    m_pipelineCompileOptions.pipelineLaunchParamsVariableName   = "optixLaunchParams";  // パラメータのグローバル変数名．CUDA 側と一致させる必要があるので注意
    
    m_pipelineLinkOptions.maxTraceDepth             = 4;                            // __raygen__ から始まるレイパスで，最大何回 OptixTrace() を呼び出すか
                                                                                    // NEE などの可視性の判断を行う
                                                                                    // 追跡可能なレイの最大深さ．(セカンダリレイまで)



    m_module.resize(m_optixModuleFileNames.size());

    for(size_t i = 0; i < m_optixModuleFileNames.size(); i++){
        std::vector<char> ptxCode = readData(m_optixModuleFileNames[i]);
        char log[8192];
        size_t sizeOfLog = sizeof(log);

    // モジュールの作成
#if OPTIX_VERSION >= 70700
        OPTIX_CHECK(optixModuleCreate(m_optixContext,
                                        &m_moduleCompileOptions,
                                        &m_pipelineCompileOptions,
                                        ptxCode.data(),
                                        ptxCode.size(),
                                        log,
                                        &sizeOfLog,
                                        &m_module[i]
                                        ));
        
#else
        OPTIX_CHECK(optixModuleCreateFromPTX(m_optixContext,
                                        &m_moduleCompileOptions,
                                        &m_pipelineCompileOptions,
                                        ptxCode.data(),
                                        ptxCode.size(),
                                        log,
                                        &sizeOfLog,
                                        &m_module[i]
                                        ));
#endif
        if(sizeOfLog > 1) PRINT(log);
    }
        
}

void Renderer::createRaygenPrograms()
{
    m_raygenPrograms.resize(1);

    OptixProgramGroupOptions    pgOptions   = {};
    OptixProgramGroupDesc       pgDesc      = {};
    pgDesc.kind                             = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module                    = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_RAYGEN)];
    if(m_sceneDesc.integrator.applySpectralRendering){
        pgDesc.raygen.entryFunctionName         = "__raygen__renderFrame_spectral";
    } else {
        pgDesc.raygen.entryFunctionName         = "__raygen__renderFrame_rgb";
    }

    // m_raygenPrograms に 登録
    char log[2048];
    size_t sizeOfLog = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &sizeOfLog,
                                        &m_raygenPrograms[0]
                                        ));
    if (sizeOfLog > 1) PRINT(log);
}

void Renderer::createMissPrograms()
{
    m_missPrograms.resize(RAY_TYPE_COUNT);

    OptixProgramGroupOptions    pgOptions   = {};
    OptixProgramGroupDesc       pgDesc      = {};
    pgDesc.kind                             = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module                      = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_MISS_RADIANCE)];

    char log[2048];
    size_t sizeOfLog = sizeof(log);
    
    // m_missPrograms に radiance ray を登録
    if(m_sceneDesc.integrator.applySpectralRendering){
        // pgDesc.miss.entryFunctionName = "__miss__radiance_spectral";
        pgDesc.miss.entryFunctionName = "__miss__radiance_sky_spectral";
    } else {
        // pgDesc.miss.entryFunctionName = "__miss__radiance_rgb";
        pgDesc.miss.entryFunctionName = "__miss__radiance_noEnvMap_rgb";
    }

    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &sizeOfLog,
                                        &m_missPrograms[RADIANCE_RAY_TYPE]
                                        ));
    if (sizeOfLog > 1) PRINT(log);

    // m_missPrograms に shadow ray を登録
    pgDesc.miss.module                      = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_MISS_SHADOW)];
    pgDesc.miss.entryFunctionName = "__miss__shadow";

    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &sizeOfLog,
                                        &m_missPrograms[SHADOW_RAY_TYPE]
                                        ));
    if (sizeOfLog > 1) PRINT(log);
    
}

void Renderer::createHitgroupPrograms()
{
    // mesh
    m_hitgroupProgramsMesh.resize(RAY_TYPE_COUNT);
    m_hitgroupProgramsVDB.resize(RAY_TYPE_COUNT);


    char log[2048];

    auto createHG = [&](OptixProgramGroup& out, 
                        OptixModule ch, const char* chName,
                        OptixModule ah, const char* ahName,
                        OptixModule is, const char* isName
                    )
    {
        OptixProgramGroupOptions    pgOptions   = {};
        OptixProgramGroupDesc       pgDesc      = {};
        pgDesc.kind                             = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        
        pgDesc.hitgroup.moduleCH                = ch;
        pgDesc.hitgroup.entryFunctionNameCH     = chName;

        pgDesc.hitgroup.moduleAH                = ah;
        pgDesc.hitgroup.entryFunctionNameAH     = ahName;
        
        pgDesc.hitgroup.moduleIS                = is;
        pgDesc.hitgroup.entryFunctionNameIS     = isName;

        size_t sizeOfLog = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &sizeOfLog,
                                        &out
                                        ));
        if (sizeOfLog > 1) PRINT(log);
    };
    
    // Mesh: radiance
    const char* chRadiance = m_sceneDesc.integrator.applySpectralRendering
        ? "__closesthit__radiance_spectral"
        : "__closesthit__radiance_rgb";
    const char* ahRadiance = m_sceneDesc.integrator.applySpectralRendering
        ? "__anyhit__radiance_spectral"
        : "__anyhit__radiance_rgb";
    
    createHG(m_hitgroupProgramsMesh[RADIANCE_RAY_TYPE],
        m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_CH_RADIANCE)], chRadiance,
        m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_AH_RADIANCE)], ahRadiance,
        /*IS*/ nullptr, /*IS name*/ nullptr
    );

    // Mesh: shadow
    createHG(m_hitgroupProgramsMesh[SHADOW_RAY_TYPE],
        m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_CH_SHADOW)], "__closesthit__shadow",
        m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_AH_SHADOW)], "__anyhit__shadow",
        /*IS*/ nullptr, /*IS name*/ nullptr
    );

    // VDB: radiance
    const char* chVDBRadiance = m_sceneDesc.integrator.applySpectralRendering
        ? "__closesthit__vdb_radiance_spectral"
        : "__closesthit__vdb_radiance_rgb";

    createHG(m_hitgroupProgramsVDB[RADIANCE_RAY_TYPE],
        m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_CH_VDB_RADIANCE)], chVDBRadiance,
        /*AH*/ nullptr, /*AH name*/ nullptr,
        m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_IS_VDB)], "__intersection__vdb"
    );

    // VDB: shadow
    createHG(m_hitgroupProgramsVDB[SHADOW_RAY_TYPE],
        m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_CH_SHADOW)], "__closesthit__shadow",
        m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_AH_SHADOW)], "__anyhit__shadow",
        m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_IS_VDB)], "__intersection__vdb"
    );
}

void Renderer::createCallablePrograms()
{
    const int numCallablePrograms = NUM_LENS_TYPE + NUM_BXDF + NUM_LIGHT_TYPE;
    m_callablePrograms.resize(numCallablePrograms);

    OptixProgramGroupOptions            pgOptions   = {};
    std::vector<OptixProgramGroupDesc>  pgDesc(numCallablePrograms);
    OptixProgramGroupDesc*              pgd;
    char log[2048];
    size_t sizeOfLog = sizeof(log);
    
    
    // レンズシェーダ
    pgd = &pgDesc[LENS_TYPE_PINHOLE];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LENS)];
    pgd->callables.entryFunctionNameDC    = "__direct_callable__lens_pinhole";

    pgd = &pgDesc[LENS_TYPE_THIN_LENS];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LENS)];
    pgd->callables.entryFunctionNameDC    = "__direct_callable__lens_thinLens";

    // BRDF のサンプリングと評価
    int offset = NUM_LENS_TYPE;
    // Diffuse
    pgd = &pgDesc[offset + BXDF_TYPE_DIFFUSE_SAMPLE];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_DIFFUSE)];
    if(m_sceneDesc.integrator.applySpectralRendering){
        pgd->callables.entryFunctionNameDC    = "__direct_callable__brdf_diffuse_sample_spectral";
    } else {
        pgd->callables.entryFunctionNameDC    = "__direct_callable__brdf_diffuse_sample_rgb";
    }
    
    pgd = &pgDesc[offset + BXDF_TYPE_DIFFUSE_EVAL];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_DIFFUSE)];
    if(m_sceneDesc.integrator.applySpectralRendering){
        pgd->callables.entryFunctionNameDC    = "__direct_callable__brdf_diffuse_eval_spectral";
    } else {
        pgd->callables.entryFunctionNameDC    = "__direct_callable__brdf_diffuse_eval_rgb";
    }
    // Disney principled
    pgd = &pgDesc[offset + BXDF_TYPE_PRINCIPLED_SAMPLE];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_PRINCIPLED)];
    if(m_sceneDesc.integrator.applySpectralRendering){
        pgd->callables.entryFunctionNameDC    = "__direct_callable__brdf_principled_sample_spectral";
    } else {
        pgd->callables.entryFunctionNameDC    = "__direct_callable__brdf_principled_sample_rgb";
    }
    pgd = &pgDesc[offset + BXDF_TYPE_PRINCIPLED_EVAL];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_PRINCIPLED)];
    if(m_sceneDesc.integrator.applySpectralRendering){
    pgd->callables.entryFunctionNameDC    = "__direct_callable__brdf_principled_eval_spectral";
    } else {
        pgd->callables.entryFunctionNameDC    = "__direct_callable__brdf_principled_eval_rgb";
    }
    // Glass
    pgd = &pgDesc[offset + BXDF_TYPE_GLASS_SAMPLE];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_GLASS)];
    if(m_sceneDesc.integrator.applySpectralRendering){
        pgd->callables.entryFunctionNameDC    = "__direct_callable__bsdf_glass_sample_spectral";
    } else {
        pgd->callables.entryFunctionNameDC    = "__direct_callable__bsdf_glass_sample_rgb";
    }
    pgd = &pgDesc[offset + BXDF_TYPE_GLASS_EVAL];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_GLASS)];
    if(m_sceneDesc.integrator.applySpectralRendering){
        pgd->callables.entryFunctionNameDC    = "__direct_callable__bsdf_glass_eval_spectral";
    } else {
        pgd->callables.entryFunctionNameDC    = "__direct_callable__bsdf_glass_eval_rgb";
    }
    // ライトサンプリング
    offset = NUM_LENS_TYPE + NUM_BXDF; 
    
    // 環境マップ
    pgd = &pgDesc[offset + LIGHT_TYPE_ENV_SPHERE];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LIGHTSAMPLE)];
    if(m_sceneDesc.integrator.applySpectralRendering){
        pgd->callables.entryFunctionNameDC    = "__direct_callable__light_env_sphere_is_spectral";
    } else {
        pgd->callables.entryFunctionNameDC    = "__direct_callable__light_env_sphere_is_rgb";
    }
    // メッシュ
    pgd = &pgDesc[offset + LIGHT_TYPE_TRIANGLE];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LIGHTSAMPLE)];
    if(m_sceneDesc.integrator.applySpectralRendering){
        pgd->callables.entryFunctionNameDC    = "__direct_callable__light_triangle_spectral";
    } else {
        pgd->callables.entryFunctionNameDC    = "__direct_callable__light_triangle_rgb";
    }
    // 空
    pgd = &pgDesc[offset + LIGHT_TYPE_SKY];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LIGHTSAMPLE)];
    if(m_sceneDesc.integrator.applySpectralRendering){
        pgd->callables.entryFunctionNameDC    = "__direct_callable__light_sky_spectral";
    } else {
        pgd->callables.entryFunctionNameDC    = "__direct_callable__light_triangle_rgb";
    }
    
    // 登録
    for(int callableID = 0; callableID < numCallablePrograms; callableID++){
        OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                            &pgDesc[callableID],
                                            1,
                                            &pgOptions,
                                            log,
                                            &sizeOfLog,
                                            &m_callablePrograms[callableID]
                                            ));
    }  
}

void Renderer::createPipeline()
{
    std::vector<OptixProgramGroup> programGroups;
    for(auto pg : m_raygenPrograms)
    {
        programGroups.push_back(pg);
    }
    for(auto pg : m_missPrograms)
    {
        programGroups.push_back(pg);
    }
    for(auto pg : m_hitgroupProgramsMesh)
    {
        programGroups.push_back(pg);
    }
    for(auto pg : m_hitgroupProgramsVDB)
    {
        programGroups.push_back(pg);
    }
    for(auto pg : m_callablePrograms)
    {
        programGroups.push_back(pg);
    }

    char log[2048];
    size_t sizeOfLog = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        m_optixContext,
        &m_pipelineCompileOptions,
        &m_pipelineLinkOptions,
        programGroups.data(),
        (int)programGroups.size(),
        log,
        &sizeOfLog,
        &m_pipeline
    ));
    if (sizeOfLog > 1) PRINT(log);

    OPTIX_CHECK(optixPipelineSetStackSize
                    (
                        // スタックサイズを設定するパイプライン
                        m_pipeline,
                        // IS AH から呼び出される Direct callable のスタックサイズの要件
                        2*1024,
                        // Raygen, Miss, Hit から呼び出される Direct callable のスタックサイズの要件
                        2*1024,
                        // レイトレの深さや Any0Hit に備えるスタックの要件
                        2*1024,
                        // トレースに渡される操作可能なグラフの最大深さ
                        2 // SINGLE IAS
                    )
    );
    if(sizeOfLog > 1) PRINT(log);
    
}

void Renderer::buildSBT()
{
    std::cout << "#Atmospheric RT: Raygen records" << std::endl;
    // Raygen records の登録
    std::vector<RaygenRecord> raygenRecords;
    for(int i = 0; i < m_raygenPrograms.size(); i++)
    {
        RaygenRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_raygenPrograms[i], &rec));   // RaygenRecord.header に， m_raygenPrograms[i] を登録
        rec.data = nullptr;
        raygenRecords.push_back(rec);
    }
    m_raygenRecordsBuffer.allocAndUpload(raygenRecords);        // GPU に情報を転送
    m_sbt.raygenRecord  = m_raygenRecordsBuffer.getDevicePointer();   // raygenRecords バッファのデバイス上の先頭アドレスを sbt に登録

    // Miss records の登録
    std::cout << "#Atmospheric RT: Miss records" << std::endl;
    std::vector<MissRecord> missRecords;
    for(int i = 0; i < m_missPrograms.size(); i++)
    {
        MissRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_missPrograms[i], &rec));   // MissRecord.header に， m_missPrograms[i] を登録
        rec.data = nullptr;
        missRecords.push_back(rec);
    }
    m_missRecordsBuffer.allocAndUpload(missRecords);                        // GPU に情報を転送
    m_sbt.missRecordBase            = m_missRecordsBuffer.getDevicePointer();     // missRecords バッファのデバイス上の先頭アドレスを sbt に登録
    m_sbt.missRecordStrideInBytes   = sizeof(MissRecord);
    m_sbt.missRecordCount           = (int)missRecords.size();

    // Callables records の登録
    std::cout << "#Atmospheric RT: Callable records" << std::endl;
    std::vector<CallableRecord> callableRecords;
    for(int i =0; i < m_callablePrograms.size(); i++){
        CallableRecord rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_callablePrograms[i], &rec));   // CallableRecord.header に， m_callablePrograms[i] を登録
        rec.data = nullptr;
        callableRecords.push_back(rec);
    }
    m_callableRecordsBuffer.allocAndUpload(callableRecords);
    m_sbt.callablesRecordBase            = m_callableRecordsBuffer.getDevicePointer();
    m_sbt.callablesRecordStrideInBytes   = sizeof(CallableRecord);
    m_sbt.callablesRecordCount           = (int)callableRecords.size();

    // Hitgroup records の登録
    std::cout << "#Atmospheric RT: Hitgroup records" << std::endl;
    // size_t numObjects = 0;
    // for(auto* mdl: m_models) numObjects += mdl->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;

    const int numMeshes = (int)m_meshTable.size();
    const int numInstances = numMeshes + (m_hasVDB ? 1 : 0);
    hitgroupRecords.reserve(numInstances * RAY_TYPE_COUNT);


    std::cout << "#Atmospheric RT: Hitgroup records (mesh)" << std::endl;
    size_t flat = 0;
    for(auto* mdl: m_models){
        for(int meshID = 0; meshID < mdl->meshes.size(); ++meshID, ++flat)
        {
            for(int rayID = 0; rayID < RAY_TYPE_COUNT; ++rayID)
            {
                HitgroupRecord rec = {};
                OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroupProgramsMesh[rayID], &rec));   // HitgroupRecord.header に， m_hitgroupProgramsMesh[rayID] を登録
                
                rec.data.geomType           = GeomType::Triangle;
                rec.data.tri.meshIndex      = static_cast<uint32_t>(flat);
                rec.data.tri.materialIndex  = m_meshMaterialIndex[static_cast<uint32_t>(flat)];
                
                hitgroupRecords.push_back(rec);
            }
        }
    }

    std::cout << "#Atmospheric RT: Hitgroup records (vdb)" << std::endl;
    if(m_hasVDB){
        // const int vdbInstanceID = flat + 1;
        for(int rayID = 0; rayID < RAY_TYPE_COUNT; ++rayID){
            HitgroupRecord rec = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroupProgramsVDB[rayID], &rec));   // HitgroupRecord.header に， m_hitgroupPrograms[rayID] を登録
            // 幾何情報の登録
            rec.data.geomType       = GeomType::VDB;
            rec.data.vdb.vdbIndex   = 0;

            hitgroupRecords.push_back(rec);
        }
    }

    m_hitgroupRecordsBuffer.allocAndUpload(hitgroupRecords);                        // GPU に情報を転送
    m_sbt.hitgroupRecordBase            = m_hitgroupRecordsBuffer.getDevicePointer();    // hitgroupRecords バッファのデバイス上の先頭アドレスを sbt に登録
    m_sbt.hitgroupRecordStrideInBytes   = sizeof(HitgroupRecord);                   // 
    m_sbt.hitgroupRecordCount           = (int)hitgroupRecords.size();

    // std::cout << "sizeof(HitgroupRecord)=" << sizeof(HitgroupRecord) << "\n";
    // std::cout << "alignof(HitgroupRecord)=" << alignof(HitgroupRecord) << "\n";

    // auto base = (uint64_t)m_hitgroupRecordsBuffer.getDevicePointer();
    // auto size = (uint64_t)hitgroupRecords.size() * sizeof(HitgroupRecord);
    // std::cout << "SBT base=0x" << std::hex << base
    //       << " size=0x" << size
    //       << " end=0x"  << (base + size) << std::dec << "\n";
}

void Renderer::render()
{
    if(m_launchParams.frame.size.x == 0) return;

    if(! m_isAccumulate){
        m_launchParams.frame.frameID = 0;
    }
    m_launchParamsBuffer.upload(&m_launchParams, 1);
    m_launchParams.frame.frameID ++;

    OPTIX_CHECK(optixLaunch(
        m_pipeline, m_stream,
        m_launchParamsBuffer.getDevicePointer(),
        m_launchParamsBuffer.getSizeInBytes(),
        &m_sbt,
        m_launchParams.frame.size.x,
        m_launchParams.frame.size.y,
        1
    ));

    // CUDA_SYNC_CHECK();

    computeFinalPixelColors();

}

void Renderer::setCamera(const Camera &camera)
{
    m_launchParams.frame.frameID = 0;
    
    // レンダラで使用するカメラパラメータを計算してセット

    // MEMO: fov を計算する
    m_lastSetCamera = camera;
    m_launchParams.camera.position  = camera.from;
    m_launchParams.camera.direction = normalize(camera.at - camera.from);
    m_launchParams.camera.horizontal    
        = normalize(cross(m_launchParams.camera.direction, camera.up));
    m_launchParams.camera.vertical      
        = normalize(cross(m_launchParams.camera.horizontal, m_launchParams.camera.direction));
    m_launchParams.camera.fValue        = camera.fValue;
    m_launchParams.camera.focalLength   = camera.focalLength;
    m_launchParams.camera.pintDist      = camera.pintDist;
    m_launchParams.camera.fov           = camera.fov;
    m_launchParams.camera.sensitivity   = camera.sensitivity;

}


void Renderer::downloadPixels(uint32_t m_h_pixels[])
{
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    m_finalColorBuffer.download(m_h_pixels, m_launchParams.frame.size.x * m_launchParams.frame.size.y);
}

void Renderer::resize(const int2 &newSize)
{
    if ((newSize.x == 0) || (newSize.y == 0)) return;

    m_denoisedBuffer.resize(newSize.x * newSize.y * sizeof(float4));
    m_fbColor.resize(newSize.x * newSize.y * sizeof(float4));
    m_fbNormal.resize(newSize.x * newSize.y * sizeof(float4));
    m_fbAlbedo.resize(newSize.x * newSize.y * sizeof(float4));
    m_finalColorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

    m_launchParams.frame.size           = newSize;
    m_launchParams.frame.colorBuffer    = (float4*)m_fbColor.getDevicePointer(); 
    m_launchParams.frame.normalBuffer   = (float4*)m_fbNormal.getDevicePointer(); 
    m_launchParams.frame.albedoBuffer   = (float4*)m_fbAlbedo.getDevicePointer(); 

    setCamera(m_lastSetCamera);
}

void Renderer::setNumDevices(const int numDevices)
{
    m_numDevices = numDevices;
}

int Renderer::getNumDevices() const 
{
    return m_numDevices;
}

// 1 ファイルにつき 1 関数と考える
void Renderer::createCUDAModule()
{
    const int numCudaModule = m_cudaModuleFileNames.size();
    m_cudaFunction.resize(numCudaModule);

    // ptx ファイルの読み出し
    std::vector<char> ptxCode = readData(m_cudaModuleFileNames[static_cast<int>(PostProcessCudaModuleIdentifier::CUDA_MODULE_ID_TONEMAP)]);
    if (ptxCode.empty()) {
        throw std::runtime_error("PTX code is empty. Check file path: " + m_cudaModuleFileNames[static_cast<int>(PostProcessCudaModuleIdentifier::CUDA_MODULE_ID_TONEMAP)]);
    }

    /////////////////////////////

    std::filesystem::path exe_path;
    char path_buffer[MAX_PATH] = {};
#if defined(_WIN32)
    if(GetModuleFileNameA(NULL, path_buffer, MAX_PATH) == 0){
        std::cerr << "ERROR: GetModuleFileNameA failed" << std::endl;
    }
    exe_path = std::filesystem::path(path_buffer);
#else
    ssize_t count = readlink("/proc/self/exe", exe_path, PATH_MAX);
    if(count == -1) {
        std::cerr << "ERROR: readlink() failed" << std::endl;
    }
#endif
    std::filesystem::path ptx_dir = exe_path.parent_path().parent_path() / "ptxes" / m_cudaModuleFileNames[static_cast<int>(PostProcessCudaModuleIdentifier::CUDA_MODULE_ID_TONEMAP)];

    std::cout << "Loading CUDA module..." << std::endl;
    CUDA_DRIVER_CHECK(cuModuleLoad(&m_cudaModule, ptx_dir.string().c_str()));
    std::cout << "Loaded CUDA module." << std::endl;

    // 登録
    std::cout << "Fetching kernel function..." << std::endl;
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&m_cudaFunction[static_cast<int>(PostProcessCudaModuleIdentifier::CUDA_MODULE_ID_TONEMAP)], m_cudaModule, "computeFinalPixelColorsKernel"));
    std::cout << "Fetched kernel function." << std::endl;
}

void Renderer::computeFinalPixelColors()
{
    int2 fbSize = m_launchParams.frame.size;
    int2 blockSize = make_int2(32);
    int2 numBlocks = make_int2(std::ceil((float)fbSize.x / (float)blockSize.x), std::ceil((float)fbSize.y / (float)blockSize.y));
    
    CUdeviceptr finalColorBufferPtr = m_finalColorBuffer.getDevicePointer();
    CUdeviceptr renderTargetBufferPtr;
    switch(m_renderBufferType){
        case COLOR:
            renderTargetBufferPtr = m_fbColor.getDevicePointer();
            break;
        case NORMAL:
            renderTargetBufferPtr = m_fbNormal.getDevicePointer();
            break;
        case ALBEDO:
            renderTargetBufferPtr = m_fbAlbedo.getDevicePointer();
            break;
        default:
            renderTargetBufferPtr = m_fbColor.getDevicePointer();
            break;
    }

    float white = m_white;
    float exposure = m_exposure;
    
    void* arg0 = &finalColorBufferPtr;
    void* arg1 = &renderTargetBufferPtr;
    void* arg2 = &fbSize;
    void* arg3 = &white;
    void* arg4 = &exposure;
    void* args[] = {
        arg0, arg1, arg2, arg3, arg4
    };
    

    CUDA_DRIVER_CHECK(
        cuLaunchKernel(
            m_cudaFunction[static_cast<int>(PostProcessCudaModuleIdentifier::CUDA_MODULE_ID_TONEMAP)],
            numBlocks.x, numBlocks.y, 1,    // スレッドのブロック数
            blockSize.x, blockSize.y, 1,    // 各ブロック内のスレッド数
            0,
            m_stream,
            args,
            nullptr 
        )
    );
    CUDA_SYNC_CHECK();

}

void Renderer::setCameraModel(const int cameraModel){
    m_launchParams.camera.cameraMode = cameraModel;
};

void Renderer::setRenderBufferType(const int renderBufferType){
    m_renderBufferType = renderBufferType;
};

int Renderer::getRenderBufferType() const{
    return m_renderBufferType;
};

void Renderer::setEnvMap(const std::string& envMapFileName)
{

    std::filesystem::path exePath;
    char pathBuffer[MAX_PATH] = {};
#if defined(_WIN32)
    if(GetModuleFileNameA(NULL, pathBuffer, MAX_PATH) == 0){
        std::cerr << "ERROR: GetModuleFileNameA failed" << std::endl;
    }
    exePath = std::filesystem::path(pathBuffer);
#else
    ssize_t count = readlink("/proc/self/exe", exePath, PATH_MAX);
    if(count == -1) {
        std::cerr << "ERROR: readlink() failed" << std::endl;
    }
#endif
    std::filesystem::path envMapDir = exePath.parent_path().parent_path().parent_path().parent_path() / "envMap" / envMapFileName;

    int2 res;
    const char* err = NULL;
    float* image;
    int compornents_per_pixel = 3;
    image = stbi_loadf(envMapDir.string().c_str(), &res.x, &res.y, &compornents_per_pixel, compornents_per_pixel);

    // int ret = LoadEXR(&image, &res.x, &res.y, envMapFileName.c_str(), &err);
    if(image){
        m_launchParams.envMapInfo.hasEnvMap = true;
        int bytes_per_scanline = compornents_per_pixel * res.x;
        float4*  h_tex_env = (float4*)malloc(res.x * res.y  * sizeof(float4));
        std::vector<float>  h_patchWeight(m_envPatchWidth * m_envPatchHeight, 0.0f);
        std::vector<float>  rowSum(m_envPatchHeight, 0.0f);
        std::vector<float>  h_coarseMarginal(m_envPatchHeight, 0.0f);
        std::vector<float>  h_coarseConditional(m_envPatchWidth * m_envPatchHeight, 0.0f);
        float  h_totalWeight = 0.0f;
        
        // -----------------------------------
        // 環境マップのアップロード
        // -----------------------------------

        // 環境マップデータの格納
        for (int i = 0; i < res.y; ++i) {
            for (int j = 0; j < res.x; ++j) {
                auto pixel = image + i * bytes_per_scanline + j * compornents_per_pixel;
                float4 tmp;
        
                tmp.x = pixel[0];
                tmp.y = pixel[1];
                tmp.z = pixel[2];
                tmp.w = 1.f;
                h_tex_env[i * res.x + j] = tmp;
            }
        }

        // アップロード
        cudaResourceDesc res_desc = {};

        cudaChannelFormatDesc channel_desc;
        int32_t width  = (int32_t)res.x;
        int32_t height = (int32_t)res.y;
        int32_t pitch = width * sizeof(float4);
        channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

        cudaArray_t pixelArrayEnv;
        CUDA_CHECK(cudaMallocArray(&pixelArrayEnv, &channel_desc, width, height));
        CUDA_CHECK(cudaMemcpy2DToArray(pixelArrayEnv, 0, 0, h_tex_env, pitch, pitch, height, cudaMemcpyHostToDevice));

        res_desc.resType    = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArrayEnv;
        // textureArrays.push_back(pixelArrayEnv);

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode     = cudaFilterModeLinear;
        tex_desc.readMode       = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy  = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode    = cudaFilterModePoint;
        tex_desc.borderColor[0]      = 1.0f;
        tex_desc.sRGB                = 0;
    
        // Create texture object
        cudaTextureObject_t cudaTex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&cudaTex, &res_desc, &tex_desc, nullptr));
        m_envMapArray = pixelArrayEnv;
        m_envMapTex = cudaTex;
        m_launchParams.envMap = cudaTex;


        // -----------------------------------
        // 環境マップの重点的サンプリングのためのデータ作成とアップロード
        // -----------------------------------

        
        // 環境マップ重点的サンプリングのためのパッチへの集計
        for (int y = 0; y < res.y; ++y) {
            // ヤコビアンの計算
            float v = ((float)y + 0.5f) / (float)res.y;
            float theta = v * M_PI;
            float sinT  = sinf(theta);

            // パッチのインデックス (緯度方向)
            int py = int(v * (float)m_envPatchHeight);
            if (py >= (float)m_envPatchHeight) py = m_envPatchHeight - 1;

            for (int x = 0; x < res.x; ++x) {
                // パッチのインデックス (経度方向)
                float u = ((float)x + 0.5f) / (float)res.x;
                int px = int(u * (float)m_envPatchWidth);
                if (px >= (float)m_envPatchWidth) px = m_envPatchWidth - 1;
                
                const float4 c = h_tex_env[y * res.x + x];
                float luminance = 0.299f * c.x + 0.587f * c.y + 0.114f * c.z;
                float w = sinT * luminance;
                
                h_patchWeight[py * m_envPatchWidth + px] += w;
            }
        }

        // 条件付き CDF の計算
        for (int py = 0; py < m_envPatchHeight; ++py) {
            float sumRow = 0.0f;
            for (int px = 0; px < m_envPatchWidth; ++px) {
                float w = h_patchWeight[py * m_envPatchWidth + px];
                sumRow += w;
            }
            rowSum[py] = sumRow;

            float accum = 0.0f;
            if(sumRow > 0.0f){
                for (int px = 0; px < m_envPatchWidth; ++px) {
                    accum += h_patchWeight[py * m_envPatchWidth + px];
                    h_coarseConditional[py * m_envPatchWidth + px] = accum / sumRow;
                }
                h_coarseConditional[py * m_envPatchWidth + (m_envPatchWidth - 1)] = 1.0f;
            } else {
                for(int px = 0; px < m_envPatchWidth; ++px){
                    h_coarseConditional[py * m_envPatchWidth + px] = float(px + 1) / float(m_envPatchWidth);
                }
            }
        }

        // 周辺 CDF の計算
        double accum = 0.0;
        for(int py = 0; py < m_envPatchHeight; ++py) accum += (double)rowSum[py];
        h_totalWeight = float(accum);

        if(h_totalWeight > 0.0f){
            double prefix = 0.0;
            for (int py = 0; py < m_envPatchHeight; ++py) {
                prefix += (double)rowSum[py];
                h_coarseMarginal[py] = float(prefix / (double)h_totalWeight);
            }
            h_coarseMarginal[m_envPatchHeight - 1] = 1.0f;
        } else {
            for (int py = 0; py < m_envPatchHeight; ++py) {
                h_coarseMarginal[py] = ((float)py + 1.0f) / float(m_envPatchHeight);
            }
            h_totalWeight = 1.0f;
        }
        
        // アップロード
        m_envCDFCoarseMarginal.allocAndUpload(h_coarseMarginal);
        m_envCDFCoarseConditional.allocAndUpload(h_coarseConditional);
        m_envPatchWeight.allocAndUpload(h_patchWeight);

        // ポインタの登録
        m_launchParams.envMapInfo.coarseMarginal = (float*)m_envCDFCoarseMarginal.getDevicePointer();
        m_launchParams.envMapInfo.coarseConditional = (float*)m_envCDFCoarseConditional.getDevicePointer();
        m_launchParams.envMapInfo.patchWeight = (float*)m_envPatchWeight.getDevicePointer();
        m_launchParams.envMapInfo.totalWeight = h_totalWeight;
        m_launchParams.envMapInfo.patchSize   = make_int2(m_envPatchWidth, m_envPatchHeight);


        stbi_image_free(image);
        free(h_tex_env);

    } else {
        std::cerr << envMapDir.string() << std::endl;
        std::cerr << "ERROR: can't find hdr file" << std::endl;
    }
}

void Renderer::createLightTable()
{
    //環境マップを光源として登録 (1枚だけ) 
    // LightDefinition lightDefinition;
    // lightDefinition.lightType = LIGHT_TYPE_ENV_SPHERE;
    // lightDefinition.lightIndexInType = 0;

    // m_lightDefinitionTable.push_back(lightDefinition);

    // 空の LUT を登録
    LightDefinition lightDefinition;
    lightDefinition.lightType = LIGHT_TYPE_SKY;
    lightDefinition.lightIndexInType = 0;
    m_lightDefinitionTable.push_back(lightDefinition);

    // 三角形の面光源を登録
    m_triangleLightDataTable.clear();

    int numMeshes = 0;

    for(auto* mdl : m_models){
        for(int meshID = 0; meshID < (int)mdl->meshes.size(); ++meshID){
            const TriangleMesh &mesh = *mdl->meshes[meshID];
            const int materialID = mesh.materialID;
            const Material& material = *mdl->materials[materialID];

            
            if(material.isLight ==  true){
                const float3 constantEmission = material.emissive;
                const bool hasEmissiveTexture = (material.emissiveTextureID >= 0);
                for(int triangleID = 0; triangleID < mesh.index.size(); triangleID++){
                    
                    const uint3 index = mesh.index[triangleID]; 
                    
                    TriangleLightData triangleLightData;
                    const float3 v0 = mesh.vertex[index.x];
                    const float3 v1 = mesh.vertex[index.y];
                    const float3 v2 = mesh.vertex[index.z];
                    triangleLightData.v0 = v0;
                    triangleLightData.v1 = v1;
                    triangleLightData.v2 = v2;
                    triangleLightData.normal = normalize(cross(v1 - v0, v2 - v0));
                    triangleLightData.area = 0.5f * length(cross(v1 - v0, v2 - v0));
                    triangleLightData.constantEmission = constantEmission;
                    triangleLightData.uv0 = mesh.emissiveTexcoord[index.x];
                    triangleLightData.uv1 = mesh.emissiveTexcoord[index.y];
                    triangleLightData.uv2 = mesh.emissiveTexcoord[index.z];
                    if(hasEmissiveTexture){
                        triangleLightData.emissiveTexture.texture = m_textureObjects[material.emissiveTextureID];
                    }

                    LightDefinition lightDefinition;
                    lightDefinition.lightType = LIGHT_TYPE_TRIANGLE;
                    lightDefinition.lightIndexInType = m_triangleLightDataTable.size();

                    m_triangleLightDataTable.push_back(triangleLightData);
                    m_lightDefinitionTable.push_back(lightDefinition);
                }
                
            }

        }
    }

    std::cout << m_lightDefinitionTable.size() << " of lights and " << m_triangleLightDataTable.size() << "of emissive triangles." << std::endl;
    // バッファのアップロード
    m_lightDefinitionBuffer.allocAndUpload(m_lightDefinitionTable);
    m_triangleLightDataBuffer.allocAndUpload(m_triangleLightDataTable);

    // コンスタントメモリに登録し，GPU から読み込めるようにする
    m_launchParams.light.lightDefinition = (LightDefinition*)m_lightDefinitionBuffer.getDevicePointer();
    m_launchParams.light.triangleLightData = (TriangleLightData*)m_triangleLightDataBuffer.getDevicePointer();
    m_launchParams.light.numLights = m_lightDefinitionTable.size();

}

const LaunchParams Renderer::getLaunchParams() const
{
    return m_launchParams;
};

const CUDABuffer&    Renderer::getFinalColorBuffer() const
{
    return m_finalColorBuffer;
};

void Renderer::setWhite(const float white)
{
    m_white = white;
}

float Renderer::getWhite() const{
    return m_white;
}

void Renderer::setExposure(const float exposure){
    m_exposure = exposure;
}

float Renderer::getExposure() const{
    return m_exposure;
}

void Renderer::setZenith(float zenithRad)
{
    m_launchParams.sunParams.sunZenithRad = zenithRad;
}

float Renderer::getZenith() const 
{
    return m_launchParams.sunParams.sunZenithRad;
}

void Renderer::setAzimuth(float azimuthRad)
{
    m_launchParams.sunParams.sunAzimuthRad = azimuthRad;
}

float Renderer::getAzimuth() const
{
    return m_launchParams.sunParams.sunAzimuthRad;
}

void Renderer::uploadSpectrumData()
{
    std::filesystem::path exePath;
    char pathBuffer[MAX_PATH] = {};
#if defined(_WIN32)
    if(GetModuleFileNameA(NULL, pathBuffer, MAX_PATH) == 0){
        std::cerr << "ERROR: GetModuleFileNameA failed" << std::endl;
    }
    exePath = std::filesystem::path(pathBuffer);
#else
    ssize_t count = readlink("/proc/self/exe", exePath, PATH_MAX);
    if(count == -1) {
        std::cerr << "ERROR: readlink() failed" << std::endl;
    }
#endif
    std::string xyzFuncFileName = m_sceneDesc.spectrum.xyzFuncFile;
    std::string upSampleBasisFileName = m_sceneDesc.spectrum.upSampleBasisFile;
    std::string D65FileName = m_sceneDesc.spectrum.D65File;
    std::filesystem::path xyzFuncDir = exePath.parent_path().parent_path().parent_path().parent_path() / "spectrum" / xyzFuncFileName;
    std::filesystem::path upSampleBasisDir = exePath.parent_path().parent_path().parent_path().parent_path() / "spectrum" / upSampleBasisFileName;
    std::filesystem::path D65Dir = exePath.parent_path().parent_path().parent_path().parent_path() / "spectrum" / D65FileName;

    // CSV ファイルの読み込み
    for(int i = 0; i < 3; ++i){
        SpectrumData xyz = loadSpectrumDataFromCSV(xyzFuncDir.string(), 0, i+1);
        SpectrumData rgbUpSamplingBasis = loadSpectrumDataFromCSV(upSampleBasisDir.string(), 0, i+1);

        m_xyz.push_back(xyz);
        m_rgbUpSamplingBasis.push_back(rgbUpSamplingBasis);
    }
    
    m_D65 = loadSpectrumDataFromCSV(D65Dir.string(), 0, 1);

    m_wavelengthMin = fmaxf(fmaxf(m_xyz[0].lambdaMin, m_D65.lambdaMin), m_rgbUpSamplingBasis[0].lambdaMin);
    m_wavelengthMax = fminf(fminf(m_xyz[0].lambdaMax, m_D65.lambdaMax), m_rgbUpSamplingBasis[0].lambdaMax);
    m_launchParams.spectral.wavelengthMin = m_wavelengthMin;
    m_launchParams.spectral.wavelengthMax = m_wavelengthMax;

    // アップロード
    for(int i = 0; i < 3; ++i) {
        cudaChannelFormatDesc channel_desc;
        channel_desc = cudaCreateChannelDesc<float>();
        const size_t N = m_xyz[i].data.size();
        std::cout << "Spectrum size: " << N << std::endl;
        cudaArray_t textureArray;
        CUDA_CHECK(cudaMallocArray(&textureArray, &channel_desc, N, 1));
        CUDA_CHECK(cudaMemcpyToArray(
            textureArray, 0, 0, 
            m_xyz[i].data.data(), N * sizeof(float), 
            cudaMemcpyHostToDevice
        ));

        cudaResourceDesc res_desc = {};
        res_desc.resType    = cudaResourceTypeArray;
        res_desc.res.array.array = textureArray;
        
        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.filterMode     = cudaFilterModeLinear;
        tex_desc.readMode       = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;

        // Create texture object
        cudaTextureObject_t cudaTex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&cudaTex, &res_desc, &tex_desc, nullptr));
        m_xyzFuncArrays.push_back(textureArray);
        m_xyzFuncObjects.push_back(cudaTex);
        m_launchParams.spectral.xyzFunc[i] = cudaTex;
    }

    for(int i = 0; i < 3; ++i) {
        cudaChannelFormatDesc channel_desc;
        channel_desc = cudaCreateChannelDesc<float>();
        const size_t N = m_rgbUpSamplingBasis[i].data.size();
        std::cout << "Spectrum size: " << N << std::endl;
        cudaArray_t textureArray;
        CUDA_CHECK(cudaMallocArray(&textureArray, &channel_desc, N, 1));
        CUDA_CHECK(cudaMemcpyToArray(
            textureArray, 0, 0, 
            m_rgbUpSamplingBasis[i].data.data(), N * sizeof(float), 
            cudaMemcpyHostToDevice
        ));

        cudaResourceDesc res_desc = {};
        res_desc.resType            = cudaResourceTypeArray;
        res_desc.res.array.array    = textureArray;
        
        cudaTextureDesc tex_desc    = {};
        tex_desc.addressMode[0]     = cudaAddressModeClamp;
        tex_desc.filterMode         = cudaFilterModeLinear;
        tex_desc.readMode           = cudaReadModeElementType;
        tex_desc.normalizedCoords   = 1;

        // Create texture object
        cudaTextureObject_t cudaTex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&cudaTex, &res_desc, &tex_desc, nullptr));
        m_rgbUpSampleFuncArrays.push_back(textureArray);
        m_rgbUpSampleFuncObjects.push_back(cudaTex);
        m_launchParams.spectral.upSampleFunc[i] = cudaTex;

    }

    const size_t N = m_D65.data.size();
    float D65Sum = 0.0f;
    for(int i = 0; i < m_D65.data.size(); ++i){
        D65Sum += m_D65.data[i];
    }
    float D65Average = D65Sum / (float)N;
    std::cout << "D65: " << D65Sum << std::endl;
    // for(int i = 0; i < m_D65.data.size(); ++i){
    //     m_D65.data[i] /= D65Average * 5.0f;
    // }

    cudaChannelFormatDesc channel_desc;
    channel_desc = cudaCreateChannelDesc<float>();
    std::cout << "Spectrum size: " << N << std::endl;
    cudaArray_t textureArray;
    CUDA_CHECK(cudaMallocArray(&textureArray, &channel_desc, N, 1));
    CUDA_CHECK(cudaMemcpyToArray(
        textureArray, 0, 0, 
        m_D65.data.data(), N * sizeof(float), 
        cudaMemcpyHostToDevice
    ));

    cudaResourceDesc res_desc = {};
    res_desc.resType            = cudaResourceTypeArray;
    res_desc.res.array.array    = textureArray;
    
    cudaTextureDesc tex_desc    = {};
    tex_desc.addressMode[0]     = cudaAddressModeClamp;
    tex_desc.filterMode         = cudaFilterModeLinear;
    tex_desc.readMode           = cudaReadModeElementType;
    tex_desc.normalizedCoords   = 1;

    // Create texture object
    cudaTextureObject_t cudaTex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&cudaTex, &res_desc, &tex_desc, nullptr));
    m_D65Array = textureArray;
    m_D65Object = cudaTex;
    m_launchParams.spectral.D65 = cudaTex;

    // wavelength importance sampling table
    const int numBins = int(m_xyz[1].data.size()) - 1;
    m_wavelengthBinCount = numBins;
    m_wavelengthBinWidth = (m_wavelengthMax - m_wavelengthMin) / float(numBins);


    // y のみ: illuminant = nullptr
    // y * D65 なら &m_d65
    buildWavelengthSamplingTable(
        m_xyz[1],
        &m_D65,
        m_wavelengthMin,
        m_wavelengthMax,
        numBins,
        m_wavelengthPdfHost,
        m_wavelengthCdfHost
    );
    m_wavelengthCdfBuffer.allocAndUpload(m_wavelengthCdfHost);
    
    if(m_wavelengthCdfTex){
        CUDA_CHECK(cudaDestroyTextureObject(m_wavelengthCdfTex));
        m_wavelengthCdfTex = 0;
    }

    m_wavelengthCdfTex = createLinearFloatTexture1D(
        m_wavelengthCdfBuffer.getDevicePointer(),
        m_wavelengthCdfBuffer.getSizeInBytes() 
    );

    m_wavelengthPdfBuffer.allocAndUpload(m_wavelengthPdfHost);

    m_launchParams.spectral.wavelengthCdfTex   = m_wavelengthCdfTex;
    m_launchParams.spectral.wavelengthPdf = 
        reinterpret_cast<const float*>(static_cast<uintptr_t>(m_wavelengthPdfBuffer.getDevicePointer()));
    // m_launchParams.spectral.wavelengthPdf = (float*)m_wavelengthPdfBuffer.getDevicePointer();
    // m_launchParams.spectral.wavelengthCdf = (float*)m_wavelengthCdfBuffer.getDevicePointer();
    m_launchParams.spectral.wavelengthBinCount = m_wavelengthBinCount;
    m_launchParams.spectral.wavelengthBinWidth = m_wavelengthBinWidth;


}


SpectrumData Renderer::loadSpectrumDataFromCSV(const std::string path, const int lambdaCol, const int DataCol){
    // CSV ファイルの読み込み
    std::ifstream ifs(path);
    if(!ifs) throw std::runtime_error("cannot_open: " + path);

    SpectrumData out;
    out.lambdaMin = 1e10f;
    out.lambdaMax = -1e10f;
    std::string line;
    bool headerSkipped = false;
    while(std::getline(ifs, line)){
        if(line.empty()) continue;

        if(!headerSkipped){
            std::istringstream hs(line);
            std::string tok;
            std::getline(hs, tok, ',');
            bool numeric = !tok.empty() && (std::isdigit(tok[0]) || tok[0]=='-' || tok[0]=='+');
            if(!numeric) {
                headerSkipped=true; 
                continue;
            }
        }
        std::istringstream ss(line);
        std::array<std::string, 4> t;
        for(int i = 0; i < DataCol + 1; ++i){
            if(!std::getline(ss, t[i], ',')) continue;
        }
        
        const float lambda = stof(t[lambdaCol]);
        if(lambda < out.lambdaMin) out.lambdaMin = lambda;
        if(lambda > out.lambdaMax) out.lambdaMax = lambda;
        
        out.data.push_back(std::stof(t[DataCol]));
    }
    return out;
};

float Renderer::getWavelengthMin() const
{
    return m_wavelengthMin;
}
float Renderer::getWavelengthMax() const
{
    return m_wavelengthMax;
}

float Renderer::getDensityScale() const
{
    if(m_launchParams.vdbs){
        return m_launchParams.vdbs[0].densityScale;
    }
}
void Renderer::setDensityScale(const float densityScale)
{
    if(m_launchParams.vdbs){
        m_launchParams.vdbs[0].densityScale = densityScale;
    }
}


void Renderer::loadVDB()
{
    try {
        m_vdbAssets = std::make_shared<NanoVDBVolumeAsset>();
        for(const auto& obj : m_sceneDesc.objects){
            if(obj.type == "vdb"){

                std::filesystem::path exePath;
                char pathBuffer[MAX_PATH] = {};
    #if defined(_WIN32)
                if(GetModuleFileNameA(NULL, pathBuffer, MAX_PATH) == 0){
                    std::cerr << "ERROR: GetModuleFileNameA failed" << std::endl;
                }
        exePath = std::filesystem::path(pathBuffer);
    #else
                ssize_t count = readlink("/proc/self/exe", exePath, PATH_MAX);
                if(count == -1) {
                    std::cerr << "ERROR: readlink() failed" << std::endl;
                }
    #endif
                std::filesystem::path vdbDir = exePath.parent_path().parent_path().parent_path().parent_path() / "model/vdb" / obj.file;
                std::cout << "VDB Path:" << vdbDir << std::endl;

                // m_vdbAssets->loadAllFloatGrids(vdbDir.string());
                m_vdbAssets->setMacrocellCellSizeVoxels(16);
                m_vdbAssets->loadDensityLikeFloatGrid(vdbDir.string());
                float bMin[3];
                float bMax[3];


                // const auto& grids = m_vdbAssets->m_grids;
                // const NanoVDBGrid& grid = grids.begin()->second;
                const NanoVDBGrid& grid = m_vdbAssets->getPrimaryGrid();
                std::cout << "[NanoVDB] primary grid = " << m_vdbAssets->getPrimaryGridName() << std::endl;

                std::copy(grid.worldMin.begin(), grid.worldMin.end(), bMin);
                std::copy(grid.worldMax.begin(), grid.worldMax.end(), bMax);
                OptixAabb aabb = {bMin[0], bMin[1], bMin[2], bMax[0], bMax[1], bMax[2]};
                std::vector<OptixAabb> h_aabb;
                h_aabb.push_back(aabb);
                m_vdbAABBBuffer.allocAndUpload(h_aabb);
                m_hasVDB = true;
                m_launchParams.vdbAABBs = (OptixAabb*)m_vdbAABBBuffer.getDevicePointer();
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[loadVDB] exception: " << e.what() << std::endl;
        throw;
    }
}

void Renderer::loadAssets()
{
    // -----------------
    // mesh data
    // -----------------

    int numMeshes = 0;
    int numMaterials = 0;
    for(auto* mdl : m_models) {
        numMeshes +=(int)mdl->meshes.size();
        numMaterials +=(int)mdl->materials.size();
        PRINT(mdl->meshes.size());
        PRINT(mdl->materials.size());
    }

    m_meshTable.clear();
    m_meshMaterialIndex.clear();
    m_materialTable.clear();

    m_meshTable.resize(numMeshes);
    m_meshMaterialIndex.resize(numMeshes);

    m_materialTable.reserve(numMaterials);
    
    size_t flat = 0;
    int numTextures = 0;
    unsigned int materialBase = 0;
    
    for(auto* mdl: m_models){
        materialBase = (unsigned int)m_materialTable.size();
        // mesh の格納
        for(int meshID = 0; meshID < mdl->meshes.size(); ++meshID, ++flat)
        {
            auto mesh = mdl->meshes[meshID];
            // 幾何情報の登録
            m_meshTable[flat].vertex            = (float3*)m_vertexBuffer[flat].getDevicePointer();
            m_meshTable[flat].index             = (uint3*)m_indexBuffer[flat].getDevicePointer();
            m_meshTable[flat].normal            = (float3*)m_normalBuffer[flat].getDevicePointer();
            m_meshTable[flat].tangent           = (float4*)m_tangentBuffer[flat].getDevicePointer();
            m_meshTable[flat].texcoord          = (float2*)m_diffuseTexcoordBuffer[flat].getDevicePointer();
            m_meshTable[flat].hasTangent        = mesh->hasTangentSpace;
            m_meshTable[flat].hasNormal         = mesh->hasNormal;
            
            int materialID = mesh->materialID;
            m_meshMaterialIndex[flat] = (materialID >= 0) ? (materialBase + (unsigned int)materialID) : 0;
        }

        // material の格納
        materialBase = (unsigned int)m_materialTable.size();
        for(auto* mat : mdl->materials){
            MaterialData matData{};
            
            matData.color       = mat->diffuse;
            matData.roughness   = mat->roughness;
            matData.metallic    = mat->metallic;
            matData.emissive    = mat->emissive;

            // material type
            if(mat->isLight){
                matData.materialType = MATERIAL_TYPE_LIGHT;
            } else if (mat->isGlass){
                matData.materialType = MATERIAL_TYPE_GLASS;
            } else if (mat->rmTextureID >= 0){
                matData.materialType = MATERIAL_TYPE_PRINCIPLED_BRDF;
            } else {
                matData.materialType = MATERIAL_TYPE_DIFFUSE;
            }

            // Texture
            if(mat->diffuseTextureID >= 0){
                matData.diffuseTexture.texture     = m_textureObjects[numTextures + mat->diffuseTextureID];
            } 
            if(mat->normalTextureID >= 0){
                matData.normalTexture.texture      = m_textureObjects[numTextures + mat->normalTextureID];
            }
            if(mat->rmTextureID >= 0){
                matData.rmTexture.texture          = m_textureObjects[numTextures + mat->rmTextureID];
            }
            if(mat->emissiveTextureID >= 0){
                matData.emissiveTexture.texture      = m_textureObjects[numTextures + mat->emissiveTextureID];
            }

            m_materialTable.push_back(matData);

        }

        numTextures +=(int)mdl->textures.size();
    }

    // GPU にアップロード
    m_meshTableBuffer.allocAndUpload(m_meshTable);
    m_materialTableBuffer.allocAndUpload(m_materialTable);

    // launch params に登録
    m_launchParams.meshes = (TriangleMeshGeomData *)m_meshTableBuffer.getDevicePointer();
    m_launchParams.numMeshes = (int)m_meshTable.size();

    m_launchParams.materials = (MaterialData*)m_materialTableBuffer.getDevicePointer();
    m_launchParams.numMaterials = (int)m_materialTable.size();

    // -----------------
    // vdb data
    // -----------------
    
    m_vdbTable.clear();

    if(!m_hasVDB || !m_vdbAssets){
        m_launchParams.vdbs = nullptr;
        m_launchParams.numVDBs = 0;
        return;
    }

    // const auto& grids = m_vdbAssets->m_grids;
    // const NanoVDBGrid& grid = grids.begin()->second;
    const NanoVDBGrid& grid = m_vdbAssets->getPrimaryGrid();
    
    VDBGeomData vdbData {};
    vdbData.nanoGrid        = grid.deviceGridPtr();
    vdbData.densityScale    = 1.0f;
    vdbData.emissionScale   = 1.0f;
    vdbData.nanoGrid              = grid.deviceGridPtr();
    vdbData.macroMajorant         = grid.macro.buffer.getDevicePointer();
    vdbData.densityScale          = 1.0f;
    vdbData.emissionScale         = 1.0f;
    vdbData.macroMaxDensity       = grid.macro.maxValue;
    vdbData.macroCellSizeVoxels   = grid.macro.cellSizeVoxels;
    vdbData.macroBaseCoord        = make_int4(
        grid.macro.baseCoord.x,
        grid.macro.baseCoord.y,
        grid.macro.baseCoord.z,
        0
    );
    vdbData.macroDims             = make_int4(
        grid.macro.dims.x,
        grid.macro.dims.y,
        grid.macro.dims.z,
        0
    );

    m_vdbTable.push_back(vdbData);
    
    // GPU にアップロード
    m_vdbTableBuffer.allocAndUpload(m_vdbTable);

    // launch params に登録
    m_launchParams.vdbs = (VDBGeomData*)m_vdbTableBuffer.getDevicePointer();
    m_launchParams.numVDBs = (int)m_vdbTable.size();
}


static bool roundtrip_check_matrix_fp16_bits(
    OptixDeviceContext ctx, cudaStream_t stream,
    const uint16_t* h_W_rowmajor_bits,  // (N x K) row-major
    uint32_t N, uint32_t K,
    CUdeviceptr d_packed_base,          // packed weights base
    size_t packed_off_bytes             // offset of this matrix
){
    const size_t bytesRM = (size_t)N * (size_t)K * sizeof(uint16_t);

    // out: ROW_MAJOR を受け取るデバイスバッファ
    void* tmp = nullptr;
    cudaMalloc(&tmp, bytesRM);
    CUdeviceptr d_back = (CUdeviceptr)tmp;

    // in desc: INFERENCING_OPTIMAL
    OptixNetworkDescription inNet{};
    OptixCoopVecMatrixDescription inMat{};
    setSingleMatrixDesc(
        ctx, inNet, inMat,
        N, K,
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
        OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
        packed_off_bytes
    );

    // out desc: ROW_MAJOR
    OptixNetworkDescription outNet{};
    OptixCoopVecMatrixDescription outMat{};
    setSingleMatrixDesc(
        ctx, outNet, outMat,
        N, K,
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
        OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR,
        0
    );
    // ROW_MAJOR の stride は必須（1行あたりのバイト数）
    outMat.rowColumnStrideInBytes = (uint32_t)(K * sizeof(uint16_t));

    OPTIX_CHECK(optixCoopVecMatrixConvert(
        ctx,
        stream,
        1,
        &inNet,
        d_packed_base,
        0,
        &outNet,
        d_back,
        0
    ));
    cudaStreamSynchronize(stream);

    // host に戻して bitwise 比較
    std::vector<uint16_t> back(size_t(N) * size_t(K));
    cudaMemcpy(back.data(), (void*)d_back, bytesRM, cudaMemcpyDeviceToHost);
    cudaFree((void*)d_back);

    return 0 == std::memcmp(back.data(), h_W_rowmajor_bits, bytesRM);
}


void Renderer::loadMieData()
{
    std::filesystem::path exePath;
    char pathBuffer[MAX_PATH] = {};
#if defined(_WIN32)
    if(GetModuleFileNameA(NULL, pathBuffer, MAX_PATH) == 0){
        std::cerr << "ERROR: GetModuleFileNameA failed" << std::endl;
    }
    exePath = std::filesystem::path(pathBuffer);
#else
    ssize_t count = readlink("/proc/self/exe", exePath, PATH_MAX);
    if(count == -1) {
        std::cerr << "ERROR: readlink() failed" << std::endl;
    }
#endif
    std::filesystem::path dataDir = exePath.parent_path().parent_path().parent_path().parent_path() / "mieSimData";
            
    m_mieHostTables = mie::load_all_txt_tables_from_directory(dataDir.string());
    m_mieGpuTextures = mie::upload_to_gpu_textures(m_mieHostTables);

    m_launchParams.mieTexture.phaseParameterG = m_mieGpuTextures.tex_g;
    m_launchParams.mieTexture.pdf = m_mieGpuTextures.tex_pdf;
    m_launchParams.mieTexture.cdf = m_mieGpuTextures.tex_cdf;
    m_launchParams.mieTexture.numTheta      = m_mieHostTables.Ntheta;
    m_launchParams.mieTexture.numLambda     = m_mieHostTables.Nlambda;
    m_launchParams.mieTexture.numDiameter   = m_mieHostTables.Nd;
}

void Renderer::buildNsfPackedWeightsCoopVec(uint32_t inputPad)
{
    std::filesystem::path exePath;
    char pathBuffer[MAX_PATH] = {};
#if defined(_WIN32)
    if(GetModuleFileNameA(NULL, pathBuffer, MAX_PATH) == 0){
        std::cerr << "ERROR: GetModuleFileNameA failed" << std::endl;
    }
    exePath = std::filesystem::path(pathBuffer);
#else
    ssize_t count = readlink("/proc/self/exe", exePath, PATH_MAX);
    if(count == -1) {
        std::cerr << "ERROR: readlink() failed" << std::endl;
    }
#endif
    std::filesystem::path dataDir = exePath.parent_path().parent_path().parent_path().parent_path() / "weights/flow_flexible.safetensors";
    
    try {
    m_nsfHyperCheckPoint = NsfHyperCheckpoint::Load(
        dataDir.string(),
        static_cast<int>(LaunchParams::NSF_MAX_TRANSFORMS)
        );
    } catch (const std::exception& e) {
        std::cerr << "NSF load failed: " << e.what() << "\n";
        throw; // もしくは return;
    }
    m_nsfTransforms = (uint32_t)m_nsfHyperCheckPoint.transforms();
    m_nsfInputPad   = inputPad;

    constexpr uint32_t kLayerPerTransform   = LaunchParams::NSF_LAYERS_PER_TRANSFORM;
    constexpr uint32_t kMaxTransforms       = LaunchParams::NSF_MAX_TRANSFORMS;

    if(m_nsfTransforms > kMaxTransforms) {
        throw std::runtime_error(
            "NSF checkpoint has more transforms than LaunchParams::NSF_MAX_TRANSFORMS"
        );
    }
    // 3 transforms / 3 layers（あなたの NSF hyper MLP 前提）
    m_nsfWOffsets.assign((size_t)m_nsfTransforms * kLayerPerTransform, 0u);
    m_nsfBOffsets.assign((size_t)m_nsfTransforms * kLayerPerTransform, 0u);


    // ----------------------------
    // Pass 1: offsets と total bytes を計算
    // ----------------------------
    uint32_t cur = 0;
    for(uint32_t t=0; t < m_nsfTransforms; ++t){
        for(uint32_t l=0; l < LaunchParams::NSF_LAYERS_PER_TRANSFORM; ++l){
            // layer0: (64 x 3) を (64 x inputPad) に拡張する
            const auto& H = m_nsfHyperCheckPoint.hyperAt((int)t);
            uint32_t N = 0, K = 0, Kpad = 0;
            if(l==0){ N = (uint32_t)H.l0.out; K = (uint32_t)H.l0.in;  Kpad = inputPad; }
            if(l==1){ N = (uint32_t)H.l1.out; K = (uint32_t)H.l1.in;  Kpad = K; }
            if(l==2){ N = (uint32_t)H.l2.out; K = (uint32_t)H.l2.in;  Kpad = K; }

            uint32_t Npack = N;
            if(l == 2) Npack = 32;

            std::cout << "[" << l << "] N:" << N << ", Npack" << Npack << ", K: " << K << ", Kpad: " << Kpad << std::endl;

            // packed weight（INFERENCING_OPTIMAL）
            cur = roundUp64_u32(cur);
            m_nsfWOffsets[idxTL(t,l)] = cur;
            cur += coopSizeFP16(m_optixContext, Npack, Kpad, OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL);

            // bias（FP16 raw）も同一バッファに入れる
            cur = roundUp64_u32(cur);
            m_nsfBOffsets[idxTL(t,l)] = cur;
            cur += roundUp64_u32(Npack * (uint32_t)sizeof(uint16_t));
        }
    }
    const uint32_t totalBytes = roundUp64_u32(cur);

    // ----------------------------
    // GPU packed バッファ確保
    // ----------------------------
    m_nsfPackedWeights.resize((size_t)totalBytes);
    std::cout << "total bytes: "  << totalBytes << std::endl;

    // ----------------------------
    // Pass 2: 各 layer を Convert + bias copy
    // ----------------------------
    for(uint32_t t=0; t < m_nsfTransforms; ++t){
        const auto& H = m_nsfHyperCheckPoint.hyperAt((int)t);

        for(uint32_t l = 0; l < LaunchParams::NSF_LAYERS_PER_TRANSFORM; ++l){
            // 取り出し（CPU側の生 weight/bias）
            const NsfHyperCheckpoint::LinearFP16* L = nullptr;
            if(l==0) L = &H.l0;
            if(l==1) L = &H.l1;
            if(l==2) L = &H.l2;
            const uint32_t N = (uint32_t)L->out;
            const uint32_t K = (uint32_t)L->in;
            const uint32_t Kpad = (l==0) ? inputPad : K;

            uint32_t Npack = N;
            if(l == 2) Npack = 32;
            std::vector<uint16_t> Wnxkpad;
            if(l == 2){
                Wnxkpad.assign((size_t)Npack * (size_t)Kpad, uint16_t(0)); // 96x64
                for(uint32_t r=0; r<N; ++r){                               // N = 95
                    std::memcpy(
                        Wnxkpad.data() + (size_t)r * (size_t)Kpad,
                        L->W.data()    + (size_t)r * (size_t)K,
                        (size_t)K * sizeof(uint16_t)
                    );
                }
            }else{
                // (N x Kpad) row-major（layer0のみ pad）
                // L->W は (out=N, in=K) の順で row-major を想定（safetensorsからの読み出しと一致）
                Wnxkpad = makeNxKpad_fp16_bits(L->W.data(), Npack, K, Kpad);
            }
            // in/out network（あなたの既存 Convert 呼び出し形式）
            OptixNetworkDescription inNet{};
            OptixCoopVecMatrixDescription inLayer{};
            OptixNetworkDescription outNet{};
            OptixCoopVecMatrixDescription outLayer{};

            // 入力行列: ROW_MAJOR FP16, offset=0
            setSingleMatrixDesc(
                m_optixContext,
                inNet, inLayer,
                Npack, Kpad,
                OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR,
                0
            );

            // 出力行列: INFERENCING_OPTIMAL FP16, offset = wOffset
            const uint32_t wOff = m_nsfWOffsets[idxTL(t,l)];
            setSingleMatrixDesc(
                m_optixContext,
                outNet, outLayer,
                Npack, Kpad,
                OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
                (size_t)wOff
            );

            // src を一時 device に上げる（Convert は device ptr 前提）
            CUdeviceptr d_src = 0;
            const size_t srcBytes = (size_t)Npack * (size_t)Kpad * sizeof(uint16_t);
            CUDA_CHECK(cudaMalloc((void**)&d_src, roundUp64_u32((uint32_t)srcBytes)));
            CUDA_CHECK(cudaMemcpyAsync((void*)d_src, Wnxkpad.data(), srcBytes, cudaMemcpyHostToDevice, m_stream));
            CUDA_CHECK(cudaStreamSynchronize(m_stream));

            // Convert: packedWeights に書き込む（outLayer.offsetInBytes を内部で参照）
            OPTIX_CHECK(optixCoopVecMatrixConvert(
                m_optixContext,
                m_stream,
                1,
                &inNet,
                d_src,
                0,
                &outNet,
                m_nsfPackedWeights.getDevicePointer(),
                0
            ));
            CUDA_CHECK(cudaStreamSynchronize(m_stream));
            CUDA_CHECK(cudaGetLastError());

            if (true) {
                const uint32_t wOff = m_nsfWOffsets[idxTL(t,l)];
                bool ok = roundtrip_check_matrix_fp16_bits(
                    m_optixContext, m_stream,
                    Wnxkpad.data(), Npack, Kpad,
                    m_nsfPackedWeights.getDevicePointer(),  // packed base
                    (size_t)wOff                            // packed offset in bytes
                );
                if (!ok) {
                    std::cerr << "[NSF] roundtrip failed at t=" << t << " l=" << l
                              << " (N=" << Npack << " K=" << Kpad << ")\n";
                    // ここで throw して止めるのがおすすめです（原因が convert/offset 側と確定するので）
                    throw std::runtime_error("NSF coopvec pack roundtrip failed");
                } else {
                    std::cout << "OK!!" << std::endl;
                }
            }


            CUDA_CHECK(cudaFree((void*)d_src));

            // bias を packed buffer の bias offset へコピー
            const uint32_t bOff = m_nsfBOffsets[idxTL(t,l)];
            const size_t bBytes = (size_t)Npack * sizeof(uint16_t);
            CUdeviceptr d_bias = m_nsfPackedWeights.getDevicePointer() + (CUdeviceptr)bOff;


            if(l == 2){
                std::vector<uint16_t> bpack(Npack, uint16_t(0));  // 96
                std::memcpy(bpack.data(), L->b.data(), (size_t)N * sizeof(uint16_t)); // 95
                CUDA_CHECK(cudaMemcpyAsync((void*)d_bias, bpack.data(),
                                           (size_t)Npack*sizeof(uint16_t),
                                           cudaMemcpyHostToDevice, m_stream));
            }else{
                CUDA_CHECK(cudaMemcpyAsync((void*)d_bias, L->b.data(),
                                           (size_t)N*sizeof(uint16_t),
                                           cudaMemcpyHostToDevice, m_stream));
            }
            
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    // ----------------------------
    // launchParams.nsf を更新（ホスト計算済みのオフセットを共有）
    // ----------------------------
    {
        auto& P = m_launchParams.nsf;

        P.packedBase  = m_nsfPackedWeights.getDevicePointer();
        P.packedBytes = totalBytes;
        P.transforms  = m_nsfTransforms;
        P.inputPad    = m_nsfInputPad;
        std::cout << "packedBytes: " << totalBytes << ", transforms: " << m_nsfTransforms << ", inputPad: " <<  m_nsfInputPad << std::endl;

        P.bins    = (uint32_t)m_nsfHyperCheckPoint.bins();
        P.hidden  = (uint32_t)m_nsfHyperCheckPoint.hidden();
        P.context = (uint32_t)m_nsfHyperCheckPoint.context();
        P.phiDim  = (uint32_t)m_nsfHyperCheckPoint.spline_out_dim();

        std::cout << "bins: " << P.bins << ", hidden: " << P.hidden << ", context: " << P.context << std::endl;

        // 未使用領域をゼロクリア（安全）
        for(uint32_t t=0; t< LaunchParams::NSF_MAX_TRANSFORMS; ++t){
            for(uint32_t l=0; l < LaunchParams::NSF_LAYERS_PER_TRANSFORM; ++l){
                P.wOffset[t][l] = 0;
                P.bOffset[t][l] = 0;
                P.N[t][l] = 0;
                P.K[t][l] = 0;
            }
        }

        // offsets コピー
        for(uint32_t t=0; t < LaunchParams::NSF_MAX_TRANSFORMS; ++t){
            const auto& H = m_nsfHyperCheckPoint.hyperAt((int)t); 
            for(uint32_t l = 0; l < LaunchParams::NSF_LAYERS_PER_TRANSFORM; ++l){
                const uint32_t wi = m_nsfWOffsets[idxTL(t,l)];
                const uint32_t bi = m_nsfBOffsets[idxTL(t,l)];
                P.wOffset[t][l] = wi;
                P.bOffset[t][l] = bi;

                // dims（任意）
                const NsfHyperCheckpoint::LinearFP16* L = (l==0)? &H.l0 : (l==1)? &H.l1 : &H.l2;
                const uint32_t N = (uint32_t)L->out;
                const uint32_t K = (uint32_t)L->in;
                const uint32_t Kpad = (l==0)? m_nsfInputPad : K;

                uint32_t Npack = N;
                if(l == 2) Npack = 32;
                
                P.N[t][l] = (uint16_t)Npack;
                P.K[t][l] = (uint16_t)Kpad;
            }
        }

        uint32_t maxW = 0, maxB = 0;
        for (uint32_t t = 0; t < m_nsfTransforms; ++t)
        for (uint32_t l = 0; l < LaunchParams::NSF_LAYERS_PER_TRANSFORM; ++l){
            maxW = std::max(maxW, m_nsfWOffsets[idxTL(t,l)]);
            maxB = std::max(maxB, m_nsfBOffsets[idxTL(t,l)]);
        }
        std::cout << "maxWOff=" << maxW << " maxBOff=" << maxB << "\n";

        // std::cout << "LP.wOffset[2][2]=" << P.wOffset[2][2]
        //     << " LP.bOffset[2][2]=" << P.bOffset[2][2] << "\n";
        if (P.transforms > 0) {
            const uint32_t tLast = P.transforms - 1;
            const uint32_t lLast = LaunchParams::NSF_LAYERS_PER_TRANSFORM - 1;
            std::cout << "LP.wOffset[" << tLast << "][" << lLast << "]="
                      << P.wOffset[tLast][lLast]
                      << " LP.bOffset[" << tLast << "][" << lLast << "]="
                      << P.bOffset[tLast][lLast] << "\n";
        }

        auto base = (uint64_t)m_nsfPackedWeights.getDevicePointer();
        std::cout << "packedBase=" << (void*)base
          << " mod64=" << (base & 63ull) << "\n";
    }
}

float Renderer::sampleSpectrumLinearCPU(const SpectrumData& spectrumData, float lambda)
{
    if(spectrumData.data.empty()) return 0.0f;
    if(spectrumData.data.size() == 1) return spectrumData.data[0];

    const float t = (lambda - spectrumData.lambdaMin) / (spectrumData.lambdaMax - spectrumData.lambdaMin);
    const float x = std::clamp(t, 0.0f, 1.0f) * float (spectrumData.data.size() - 1);
    const int i0 = std::min(int(std::floor(x)), int(spectrumData.data.size() - 2));
    const int i1 = i0 + 1;
    const float a = x - float(i0);

    return (1.0f - a) * spectrumData.data[i0] + a * spectrumData.data[i1];
}

void Renderer::buildWavelengthSamplingTable(
    const SpectrumData& yBar, // 等色関数
    const SpectrumData* illuminant,
    float lambdaMin,
    float lambdaMax,
    int numBins,
    std::vector<float>& outPdf,
    std::vector<float>& outCdf
)
{
    outPdf.assign(numBins, 0.0f);
    outCdf.assign(numBins + 1, 0.0f);

    const float dLambda = (lambdaMax - lambdaMin) / float(numBins);

    float sum = 0.0f;
    for(int i = 0; i < numBins; ++i){
        const float l0 = lambdaMin + dLambda * float(i);
        const float l1 = l0 + dLambda;

        const float y0 = std::max(0.0f, sampleSpectrumLinearCPU(yBar, l0));
        const float y1 = std::max(0.0f, sampleSpectrumLinearCPU(yBar, l1));

        float e0 = 1.0f;
        float e1 = 1.0f;
        if(illuminant){
            e0 = std::max(0.0f, sampleSpectrumLinearCPU(*illuminant, l0));
            e1 = std::max(0.0f, sampleSpectrumLinearCPU(*illuminant, l1));
        }

        const float f0 = y0 * e0;
        const float f1 = y1 * e1;
        const float w = 0.5f * (f0 + f1) * dLambda;

        outPdf[i] = w;
        sum += w;
    }

    if(sum <= 0.0f){
        // fallback
        for(int i = 0; i < numBins; ++i){
            outPdf[i] = 1.0f / float(numBins);
            outCdf[i + 1] = float(i + 1) / float(numBins);
        }
        outCdf[0] = 0.0f;
        outCdf[numBins] = 1.0f;
        return;
    }

    outCdf[0] = 0.0f;
    for(int i = 0; i < numBins; ++i){
        outPdf[i] /= sum;
        outCdf[i + 1] = outCdf[i] + outPdf[i];
    }

    outCdf[numBins] = 1.0f;
}

bool Renderer::loadAtmosphere()
{
    std::filesystem::path exePath;
    char pathBuffer[MAX_PATH] = {};
#if defined(_WIN32)
    if(GetModuleFileNameA(NULL, pathBuffer, MAX_PATH) == 0){
        std::cerr << "ERROR: GetModuleFileNameA failed" << std::endl;
    }
    exePath = std::filesystem::path(pathBuffer);
#else
    ssize_t count = readlink("/proc/self/exe", exePath, PATH_MAX);
    if(count == -1) {
        std::cerr << "ERROR: readlink() failed" << std::endl;
    }
#endif
    std::filesystem::path dataDir = exePath.parent_path().parent_path().parent_path().parent_path() / "atmosphericLUT";
    std::string error;
    m_atmosphereLUTs.setDirectory(dataDir);
    if(!m_atmosphereLUTs.load(error)) {
        std::cerr << "Failed to load atmosphere LUTs: " << error << std::endl;
        return false;
    }

    const AtmosphereDeviceData& atmo = m_atmosphereLUTs.deviceData();

    // launchParams に登録していく
    m_launchParams.atmo.sunTransmittanceTex = atmo.sunTransmittanceTex;
    m_launchParams.atmo.directIrradianceTex = atmo.directIrradianceTex;

    m_launchParams.atmo.skyRayleighTexHandles   = atmo.skyRayleighTexHandles;
    m_launchParams.atmo.skyMieTexHandles        = atmo.skyMieTexHandles;
    m_launchParams.atmo.skyMultipleTexHandles   = atmo.skyMultipleTexHandles;
    
    m_launchParams.atmo.lambdaCount   = atmo.lambdaCount;

    m_launchParams.atmo.bottomRadius_m      = atmo.bottomRadius_m;
    m_launchParams.atmo.topRadius_m         = atmo.topRadius_m;
    m_launchParams.atmo.observerAltitude_m  = atmo.observerAltitude_m;
    m_launchParams.atmo.muSMin              = atmo.muSMin;

    m_launchParams.atmo.skyNu   = atmo.skyNu;
    m_launchParams.atmo.skyMu   = atmo.skyMu;
    m_launchParams.atmo.skyMuS  = atmo.skyMuS;

    m_launchParams.atmo.irradianceR     = atmo.irradianceR;
    m_launchParams.atmo.irradianceMuS   = atmo.irradianceMuS;

    return true;
}

CUstream Renderer::getStream() const { return m_stream; }