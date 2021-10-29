// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <string>
#include <iterator>
#include <samples/common.hpp>

#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>
#include <samples/classification_results.h>
#include <gpu/gpu_context_api_dx.hpp>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <atlcomcli.h>

#define SURFACE_SHARE_OPENVINO 0
#define SURFACE_SHARE_USER 1

using namespace InferenceEngine;

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
#define tcout std::wcout
#define file_name_t std::wstring
#define imread_t imreadW
#define ClassificationResult_t ClassificationResultW
#else
#define tcout std::cout
#define file_name_t std::string
#define imread_t cv::imread
#define ClassificationResult_t ClassificationResult
#endif

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
cv::Mat imreadW(std::wstring input_image_path) {
    cv::Mat image;
    std::ifstream input_image_stream;
    input_image_stream.open(
        input_image_path.c_str(),
        std::iostream::binary | std::ios_base::ate | std::ios_base::in);
    if (input_image_stream.is_open()) {
        if (input_image_stream.good()) {
            input_image_stream.seekg(0, std::ios::end);
            std::size_t file_size = input_image_stream.tellg();
            input_image_stream.seekg(0, std::ios::beg);
            std::vector<char> buffer(0);
            std::copy(
                std::istreambuf_iterator<char>(input_image_stream),
                std::istreambuf_iterator<char>(),
                std::back_inserter(buffer));
            image = cv::imdecode(cv::Mat(1, file_size, CV_8UC1, &buffer[0]), cv::IMREAD_COLOR);
        } else {
            tcout << "Input file '" << input_image_path << "' processing error" << std::endl;
        }
        input_image_stream.close();
    } else {
        tcout << "Unable to read input file '" << input_image_path << "'" << std::endl;
    }
    return image;
}

std::string simpleConvert(const std::wstring & wstr) {
    std::string str;
    for (auto && wc : wstr)
        str += static_cast<char>(wc);
    return str;
}

int wmain(int argc, wchar_t *argv[]) {
#else

int main(int argc, char *argv[]) {
#endif
    try {
        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (argc != 4) {
            tcout << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <device_name>" << std::endl;
            return EXIT_FAILURE;
        }

        const file_name_t input_model{argv[1]};
        const file_name_t input_image_path{argv[2]};
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        const std::string device_name = simpleConvert(argv[3]);
#else
        const std::string device_name{argv[3]};
#endif
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine instance -------------------------------------
        Core ie;
        // -----------------------------------------------------------------------------------------------------

        // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        CNNNetwork network = ie.ReadNetwork(input_model);
        if (network.getOutputsInfo().size() != 1) throw std::logic_error("Sample supports topologies with 1 output only");
        if (network.getInputsInfo().size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
        std::string input_name = network.getInputsInfo().begin()->first;

        /* Mark input as resizable by setting of a resize algorithm.
         * In this case we will be able to set an input blob of any shape to an infer request.
         * Resize and layout conversions are executed automatically during inference */
        input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        input_info->setLayout(Layout::NHWC);
        input_info->setPrecision(Precision::U8);

        // --------------------------- Prepare output blobs ----------------------------------------------------
        DataPtr output_info = network.getOutputsInfo().begin()->second;
        std::string output_name = network.getOutputsInfo().begin()->first;

        output_info->setPrecision(Precision::FP32);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
#if SURFACE_SHARE_USER
        // https://docs.openvino.ai/latest/openvino_docs_IE_DG_supported_plugins_GPU_RemoteBlob_API.html
        // https://docs.openvino.ai/latest/namespaceInferenceEngine_1_1gpu.html

        ID3D11Device* g_pD3D11Device = NULL;
        ID3D11DeviceContext* g_pD3D11Ctx = NULL;
        IDXGIFactory2* g_pDXGIFactory = NULL;
        IDXGIAdapter* g_pAdapter = NULL;

        HRESULT hres = CreateDXGIFactory1(__uuidof(IDXGIFactory2),
            (void**)(&g_pDXGIFactory));
        if (FAILED(hres))
        {
            printf("Failed to CreateDXGIFactory: %d", hres);
            return 1;
        }

        hres = g_pDXGIFactory->EnumAdapters(0, &g_pAdapter);
        if (FAILED(hres))
        {
            printf("Failed to EnumAdapters: %d", hres);
            return 1;
        }

        static D3D_FEATURE_LEVEL FeatureLevels[] = { D3D_FEATURE_LEVEL_11_1,
                            D3D_FEATURE_LEVEL_11_0,
                            D3D_FEATURE_LEVEL_10_1,
                            D3D_FEATURE_LEVEL_10_0 };
        D3D_FEATURE_LEVEL pFeatureLevelsOut;
        UINT dxFlags = 0;
        //UINT dxFlags = D3D11_CREATE_DEVICE_DEBUG;

        hres = D3D11CreateDevice(
            g_pAdapter, D3D_DRIVER_TYPE_UNKNOWN, NULL, dxFlags,
            FeatureLevels,
            (sizeof(FeatureLevels) / sizeof(FeatureLevels[0])),
            D3D11_SDK_VERSION, &g_pD3D11Device, &pFeatureLevelsOut,
            &g_pD3D11Ctx);
        if (FAILED(hres))
        {
            printf("Failed to D3D11CreateDevice: %d", hres);
            return 1;
        }

        // turn on multithreading for the DX11 context
        CComQIPtr<ID3D10Multithread> p_mt(g_pD3D11Ctx);
        if (p_mt)
            p_mt->SetMultithreadProtected(true);

        // LoadNetwork with ID3D11Device
        auto remote_context = gpu::make_shared_context(ie, "GPU", g_pD3D11Device);
        ExecutableNetwork executable_network = ie.LoadNetwork(network, remote_context);
#elif SURFACE_SHARE_OPENVINO
        ExecutableNetwork executable_network = ie.LoadNetwork(network, "GPU");
        auto d3d11_context = executable_network.GetContext();
        //::Context ctx = std::dynamic_pointer_cast<cl::Context>(cldnn_context);
#else
        ExecutableNetwork executable_network = ie.LoadNetwork(network, device_name);
#endif
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        /* Read input image to a blob and set it to an infer request without resize and layout conversions. */
        cv::Mat image = imread_t(input_image_path);
#if SURFACE_SHARE_USER
        // create default surface
        D3D11_TEXTURE2D_DESC desc = { 0 };
        desc.Width = image.cols;
        desc.Height = image.rows;
        desc.MipLevels = 1;
        desc.ArraySize = 1; // number of subresources is 1 in this case
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET;
        desc.MiscFlags = 0;
        ID3D11Texture2D* pSurf = NULL;
        hres = g_pD3D11Device->CreateTexture2D(&desc, NULL, &pSurf);
        if (FAILED(hres))
        {
            printf("Failed to CreateTexture2D (default): %d", hres);
            return 1;
        }

        // Create staging surface
        desc.ArraySize = 1;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
        desc.BindFlags = 0;
        desc.MiscFlags = 0;
        ID3D11Texture2D* pStagingSurf = NULL;
        hres = g_pD3D11Device->CreateTexture2D(&desc, NULL, &pStagingSurf);
        if (FAILED(hres))
        {
            printf("Failed to CreateTexture2D (staging): %d", hres);
            return 1;
        }

        // image (RGB) -> CPU buffer (ARGB)
        char* buffer;
        int width = image.cols;
        int height = image.rows;
        int bufferLen = width * height * 4;
        buffer = (char*) malloc(bufferLen);
        for (int i = 0; i < image.rows; i++)
        {
            for (int j = 0; j < image.cols; j++)
            {
                buffer[(i * width + j) * 4] = image.data[(i * image.cols + j) * 3];
                buffer[(i * width + j) * 4 + 1] = image.data[(i * image.cols + j) * 3 + 1];
                buffer[(i * width + j) * 4 + 2] = image.data[(i * image.cols + j) * 3 + 2];
            }
        }

        // CPU buffer -> staging buffer
        D3D11_MAPPED_SUBRESOURCE lockedRect = { 0 };
        do {
            hres = g_pD3D11Ctx->Map(pStagingSurf, 0, D3D11_MAP_READ_WRITE,
                D3D11_MAP_FLAG_DO_NOT_WAIT, &lockedRect);
            if (S_OK != hres && DXGI_ERROR_WAS_STILL_DRAWING != hres)
            {
                printf("Failed to Map: %d", hres);
                return 1;
            }
        } while (DXGI_ERROR_WAS_STILL_DRAWING == hres);
        memcpy(lockedRect.pData, buffer, bufferLen);
        g_pD3D11Ctx->Unmap(pStagingSurf, 0);

        // staging buffer -> GPU buffer
        //g_pD3D11Ctx->CopySubresourceRegion(pSurf, 0, 0, 0, 0, pStagingSurf, 0, NULL);
        g_pD3D11Ctx->CopyResource(pSurf, pStagingSurf);

        // ID3D11Texture2D -> Blob
        InputsDataMap inputInfo(network.getInputsInfo());
        if (inputInfo.size() != 1) {
            THROW_IE_EXCEPTION << "The network should have only one input";
        }
        InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
        TensorDesc tensorDesc = inputInfoFirst->getInputData()->getTensorDesc();
        SizeVector input_dims = tensorDesc.getDims();
        // Todo:
        // Model input_dims is 224x224, but pSurf is 300x300 (the same as input image)
        // Is this OK?
        Blob::Ptr imgBlob = make_shared_blob(tensorDesc, remote_context, pSurf);
#elif SURFACE_SHARE_OPENVINO
        Todo
#else
        Blob::Ptr imgBlob = wrapMat2Blob(image);  // just wrap Mat data by Blob::Ptr without allocating of new memory
        infer_request.SetBlob(input_name, imgBlob);  // infer_request accepts input blob of any size
#endif
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference --------------------------------------------------------
        /* Running the request synchronously */
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output ------------------------------------------------------
        Blob::Ptr output = infer_request.GetBlob(output_name);
        // Print classification results
        ClassificationResult_t classificationResult(output, {input_image_path});
        classificationResult.print();
        // -----------------------------------------------------------------------------------------------------

#if SURFACE_SHARE_USER
        // release DirectX
        if (buffer)
        {
            free(buffer);
        }
        if (pSurf)
        {
            pSurf->Release();
            pSurf = NULL;
        }
        if (pStagingSurf)
        {
            pStagingSurf->Release();
            pStagingSurf = NULL;
        }
        if (g_pAdapter) {
            g_pAdapter->Release();
            g_pAdapter = NULL;
        }
        if (g_pD3D11Device) {
            g_pD3D11Device->Release();
            g_pD3D11Device = NULL;
        }
        if (g_pD3D11Ctx) {
            g_pD3D11Ctx->Release();
            g_pD3D11Ctx = NULL;
        }
        if (g_pDXGIFactory) {
            g_pDXGIFactory->Release();
            g_pDXGIFactory = NULL;
        }
#elif SURFACE_SHARE_OPENVINO
        Todo
#endif
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "This sample is an API example, for any performance measurements "
                 "please use the dedicated benchmark_app tool" << std::endl;
    return EXIT_SUCCESS;
}
