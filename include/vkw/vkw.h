#ifndef VKW_BASE_H_20191130
#define VKW_BASE_H_20191130

#include <vkw/warning_suppressor.h>

// Set Vulkan flag for Android
#if defined(__ANDROID__) && !defined(VK_USE_PLATFORM_ANDROID_KHR)
#define VK_USE_PLATFORM_ANDROID_KHR
#endif

// -----------------------------------------------------------------------------
// ------------------------- Begin third party include -------------------------
// -----------------------------------------------------------------------------
BEGIN_VKW_SUPPRESS_WARNING
// Vulkan-Hpp
#include <vulkan/vulkan.hpp>

#if defined(__ANDROID__)
// ANativeWindow for android
#include <android/native_window.h>
#include <android/native_window_jni.h>
#else
// GLFW for desktop
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif
// CUDA
#if VKW_USE_CUDA
#include <cuda_runtime_api.h>
extern const vk::ExternalMemoryHandleTypeFlagBits g_DefaultMemoryHandleType;
extern const vk::ExternalSemaphoreHandleTypeFlagBits
        g_DefaultSemaphoreHandleType;
#endif
END_VKW_SUPPRESS_WARNING
// -----------------------------------------------------------------------------
// -------------------------- End third party include --------------------------
// -----------------------------------------------------------------------------

#include <atomic>
#include <functional>
#include <limits>
#include <map>

namespace vkw {

// -----------------------------------------------------------------------------
// --------------------------------- Constants ---------------------------------
// -----------------------------------------------------------------------------
const uint64_t NO_TIMEOUT = std::numeric_limits<uint64_t>::max();

// -----------------------------------------------------------------------------
// --------------------------------- Printers ----------------------------------
// -----------------------------------------------------------------------------
void PrintInfo(const std::string& str);
void PrintErr(const std::string& str);
void SetPrintInfoFunc(const std::function<void(const std::string& str)>& func);
void SetPrintErrFunc(const std::function<void(const std::string& str)>& func);

// -----------------------------------------------------------------------------
// -------------------------- Info Getters / Printers --------------------------
// -----------------------------------------------------------------------------
std::string GetInstanceLayerPropsStr();
std::string GetInstanceExtensionPropsStr();
std::string GetQueueFamilyPropsStr(const vk::PhysicalDevice& physical_device);
std::string GetPhysicalPropsStr(const vk::PhysicalDevice& physical_device);
void PrintInstanceLayerProps();
void PrintInstanceExtensionProps();
void PrintQueueFamilyProps(const vk::PhysicalDevice& physical_device);
void PrintPhysicalProps(const vk::PhysicalDevice& physical_device);

// -----------------------------------------------------------------------------
// -------------------------------- FPS counter --------------------------------
// -----------------------------------------------------------------------------
void PrintFps(std::function<void(float)> print_func = nullptr,
              float show_interval_sec = 1.f);

// -----------------------------------------------------------------------------
// ---------------------------------- Float16 ----------------------------------
// -----------------------------------------------------------------------------
uint16_t CastFloat32To16(const float& f32);
float CastFloat16To32(const uint16_t& f16);
void CastFloat32To16(const float* src_p, uint16_t* dst_p, size_t n);
void CastFloat16To32(const uint16_t* src_p, float* dst_p, size_t n);
std::vector<uint16_t> CastFloat32To16(const std::vector<float>& f32);
std::vector<float> CastFloat16To32(const std::vector<uint16_t>& f16);

// -----------------------------------------------------------------------------
// ----------------------------------- Window ----------------------------------
// -----------------------------------------------------------------------------
#if defined(__ANDROID__)
// Android version (ANativeWindow)
using ANativeWindowPtr = std::shared_ptr<ANativeWindow>;
using WindowPtr = ANativeWindowPtr;
WindowPtr InitANativeWindow(JNIEnv* jenv, jobject jsurface);

#else
// GLFW version (GLFWWindow)
using GLFWWindowPtr = std::shared_ptr<GLFWwindow>;
using WindowPtr = GLFWWindowPtr;
WindowPtr InitGLFWWindow(const std::string& win_name, uint32_t win_w = 512,
                         uint32_t win_h = 512, int32_t glfw_api = GLFW_NO_API);
#endif

// -----------------------------------------------------------------------------
// --------------------------------- Instance ----------------------------------
// -----------------------------------------------------------------------------
struct InstancePack {
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT debug_utils_messenger;
    vk::UniqueDebugReportCallbackEXT debug_report_callback;
};
using InstancePackPtr = std::shared_ptr<InstancePack>;
InstancePackPtr CreateInstance(const std::string& app_name,
                               uint32_t app_version,
                               const std::string& engine_name,
                               uint32_t engine_version,
                               bool debug_enable = true,
                               bool surface_enable = true,
                               const std::vector<char const*>& layer_names = {},
                               const std::vector<char const*>& ext_names = {},
                               bool export_enable = false);

// -----------------------------------------------------------------------------
// ------------------------------ PhysicalDevice -------------------------------
// -----------------------------------------------------------------------------
std::vector<vk::PhysicalDevice> GetPhysicalDevices(
        const InstancePackPtr& instance_pack);

vk::PhysicalDevice GetFirstPhysicalDevice(const InstancePackPtr& instance_pack);

using Features2Ptr = std::shared_ptr<vk::PhysicalDeviceFeatures2>;
Features2Ptr GetPhysicalFeatures2(const vk::PhysicalDevice& physical_device);

using PropertiesPtr = std::shared_ptr<vk::PhysicalDeviceProperties>;
PropertiesPtr GetPhysicalProperties(const vk::PhysicalDevice& physical_device);

// -----------------------------------------------------------------------------
// ---------------------------------- Surface ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueSurfaceKHR CreateSurface(const InstancePackPtr& instance_pack,
                                   const WindowPtr& window);

vk::Format GetSurfaceFormat(const vk::PhysicalDevice& physical_device,
                            const vk::UniqueSurfaceKHR& surface);

// -----------------------------------------------------------------------------
// -------------------------------- Queue Family -------------------------------
// -----------------------------------------------------------------------------
std::vector<uint32_t> GetQueueFamilyIdxs(
        const vk::PhysicalDevice& physical_device,
        const vk::QueueFlags& queue_flags = vk::QueueFlagBits::eGraphics |
                                            vk::QueueFlagBits::eCompute |
                                            vk::QueueFlagBits::eTransfer);
uint32_t GetQueueFamilyIdx(
        const vk::PhysicalDevice& physical_device,
        const vk::QueueFlags& queue_flags = vk::QueueFlagBits::eGraphics |
                                            vk::QueueFlagBits::eCompute |
                                            vk::QueueFlagBits::eTransfer);

uint32_t GetGraphicPresentQueueFamilyIdx(
        const vk::PhysicalDevice& physical_device,
        const vk::UniqueSurfaceKHR& surface,
        const vk::QueueFlags& queue_flags = vk::QueueFlagBits::eGraphics |
                                            vk::QueueFlagBits::eCompute |
                                            vk::QueueFlagBits::eTransfer);

// -----------------------------------------------------------------------------
// ----------------------------------- Device ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueDevice CreateDevice(uint32_t queue_family_idx,
                              const vk::PhysicalDevice& physical_device,
                              uint32_t n_queues = 1,
                              bool swapchain_support = true,
                              const Features2Ptr& features = nullptr,
                              const std::vector<const char*>& ext_names = {},
                              bool export_enable = false);

// -----------------------------------------------------------------------------
// ------------------------------- Asynchronous --------------------------------
// -----------------------------------------------------------------------------
using FencePtr = std::shared_ptr<vk::UniqueFence>;
FencePtr CreateFence(const vk::UniqueDevice& device);
void ResetFence(const vk::UniqueDevice& device, const FencePtr& fence);

vk::Result WaitForFence(const vk::UniqueDevice& device, const FencePtr& fence,
                        uint64_t timeout = NO_TIMEOUT);
vk::Result WaitForFences(const vk::UniqueDevice& device,
                         const std::vector<FencePtr>& fences,
                         bool wait_all = true, uint64_t timeout = NO_TIMEOUT);

using EventPtr = std::shared_ptr<vk::UniqueEvent>;
EventPtr CreateEvent(const vk::UniqueDevice& device);

using SemaphorePtr = std::shared_ptr<vk::UniqueSemaphore>;
SemaphorePtr CreateSemaphore(const vk::UniqueDevice& device,
                             bool export_enable = false);

// -----------------------------------------------------------------------------
// ------------------------------- Device Memory -------------------------------
// -----------------------------------------------------------------------------
struct DeviceMemoryPack {
    vk::UniqueDeviceMemory dev_mem;
    vk::DeviceSize dev_mem_size;
    vk::MemoryPropertyFlags mem_prop;
};
using DeviceMemoryPackPtr = std::shared_ptr<DeviceMemoryPack>;
DeviceMemoryPackPtr CreateDeviceMemoryPack(
        const vk::UniqueDevice& device,
        const vk::PhysicalDevice& physical_device,
        const vk::MemoryRequirements& mem_req,
        const vk::MemoryPropertyFlags& mem_prop, bool is_exportable = false);

void SendToDevice(const vk::UniqueDevice& device,
                  const DeviceMemoryPackPtr& dev_mem_pack, const void* data,
                  uint64_t n_bytes);
void RecvFromDevice(const vk::UniqueDevice& device,
                    const DeviceMemoryPackPtr& dev_mem_pack, void* data,
                    uint64_t n_bytes);
#if VKW_USE_CUDA
void ImportCudaExternalMemory(const vk::UniqueDevice& device, void** cuda_ptr,
                              cudaExternalMemory_t& cuda_mem,
                              vk::DeviceMemory& vk_mem, vk::DeviceSize size,
                              vk::ExternalMemoryHandleTypeFlagBits handle_type =
                                      g_DefaultMemoryHandleType);
#endif
// -----------------------------------------------------------------------------
// ----------------------------------- Buffer ----------------------------------
// -----------------------------------------------------------------------------
static constexpr auto HOST_VISIB_COHER_PROPS =
        vk::MemoryPropertyFlagBits::eHostCoherent |
        vk::MemoryPropertyFlagBits::eHostVisible;

struct BufferPack {
    vk::UniqueBuffer buf;
    vk::DeviceSize size;
    vk::BufferUsageFlags usage;
    DeviceMemoryPackPtr dev_mem_pack;
};
using BufferPackPtr = std::shared_ptr<BufferPack>;
BufferPackPtr CreateBufferPack(const vk::PhysicalDevice& physical_device,
                               const vk::UniqueDevice& device,
                               const vk::DeviceSize& size = 256,
                               const vk::BufferUsageFlags& usage =
                                       vk::BufferUsageFlagBits::eVertexBuffer,
                               const vk::MemoryPropertyFlags& mem_prop =
                                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                               bool is_exportable = false);

void SendToDevice(const vk::UniqueDevice& device, const BufferPackPtr& buf_pack,
                  const void* data, uint64_t n_bytes);
void RecvFromDevice(const vk::UniqueDevice& device,
                    const BufferPackPtr& buf_pack, void* data,
                    uint64_t n_bytes);

void CopyBuffer(const vk::UniqueCommandBuffer& cmd_buf,
                const BufferPackPtr& src_buf_pack,
                const BufferPackPtr& dst_buf_pack);

// -----------------------------------------------------------------------------
// ----------------------------------- Image -----------------------------------
// -----------------------------------------------------------------------------
static constexpr uint32_t MAX_MIP_LEVEL = uint32_t(~0);
static constexpr vk::ImageLayout LAYOUT_DONT_CARE = vk::ImageLayout::eUndefined;

struct ImageResPack {
    vk::UniqueImage img;
    vk::Format format;
    vk::Extent2D size;
    uint32_t miplevel_cnt;
    vk::ImageUsageFlags usage;
    vk::MemoryPropertyFlags mem_prop;
    bool is_tiling;
    vk::ImageAspectFlags aspects;
    bool is_shared;
    DeviceMemoryPackPtr dev_mem_pack;
};
using ImageResPackPtr = std::shared_ptr<ImageResPack>;

struct ImagePack {
    vk::UniqueImageView view;
    vk::Format view_format;
    vk::Extent2D view_size;
    vk::ImageAspectFlags view_aspects;
    uint32_t view_miplevel_base;
    uint32_t view_miplevel_cnt;
    ImageResPackPtr img_res_pack;
};
using ImagePackPtr = std::shared_ptr<ImagePack>;
ImagePackPtr CreateImagePack(
        const vk::PhysicalDevice& physical_device,
        const vk::UniqueDevice& device,
        const vk::Format& format = vk::Format::eR8G8B8A8Uint,
        const vk::Extent2D& size = {256, 256}, uint32_t miplevel_cnt = 1,
        const vk::ImageUsageFlags& usage = vk::ImageUsageFlagBits::eSampled,
        const vk::MemoryPropertyFlags& mem_prop = {}, bool is_tiling = true,
        const vk::ImageAspectFlags& aspects = vk::ImageAspectFlagBits::eColor,
        const vk::ImageLayout& init_layout = vk::ImageLayout::eUndefined,
        bool is_shared = false, bool is_exportable = false);
ImagePackPtr CreateImagePack(  // ROI
        const ImageResPackPtr& img_res_pack, const vk::UniqueDevice& device,
        const vk::Format& format = vk::Format::eUndefined,
        uint32_t miplevel_base = 0, uint32_t miplevel_cnt = MAX_MIP_LEVEL,
        const vk::ImageAspectFlags& aspects = vk::ImageAspectFlagBits::eColor);
ImagePackPtr CreateImagePack(  // No resource (only view)
        vk::UniqueImageView&& img_view,
        const vk::Format& format = vk::Format::eUndefined,
        const vk::Extent2D& size = {256, 256},
        const vk::ImageAspectFlags& aspects = vk::ImageAspectFlagBits::eColor);

uint32_t GetMaxMipLevelCount(const vk::Extent2D& base_size);
vk::Extent2D GetMippedSize(const vk::Extent2D& base_size, uint32_t miplevel);

void SendToDevice(const vk::UniqueDevice& device,
                  const ImageResPackPtr& img_res_pack, const void* data,
                  uint64_t n_bytes);
void RecvFromDevice(const vk::UniqueDevice& device,
                    const ImageResPackPtr& img_res_pack, void* data,
                    uint64_t n_bytes);

void BarrierImage(const vk::UniqueCommandBuffer& cmd_buf,
                  const ImagePackPtr& img_pack,
                  const vk::ImageLayout& init_layout,
                  const vk::PipelineStageFlags& src_stage =
                          vk::PipelineStageFlagBits::eColorAttachmentOutput,
                  const vk::AccessFlags& src_access_mask =
                          vk::AccessFlagBits::eColorAttachmentWrite,
                  const vk::PipelineStageFlags& dst_stage =
                          vk::PipelineStageFlagBits::eFragmentShader,
                  const vk::AccessFlags& dst_access_mask =
                          vk::AccessFlagBits::eShaderRead);
void SetImageLayout(const vk::UniqueCommandBuffer& cmd_buf,
                    const ImagePackPtr& img_pack,
                    const vk::ImageLayout& init_layout,
                    const vk::ImageLayout& final_layout);

void CopyBufferToImage(const vk::UniqueCommandBuffer& cmd_buf,
                       const BufferPackPtr& src_buf_pack,
                       const ImagePackPtr& dst_img_pack,
                       const vk::ImageLayout& init_layout,
                       const vk::ImageLayout& final_layout = LAYOUT_DONT_CARE,
                       const uint32_t dst_miplevel = 0);

void CopyImageToBuffer(const vk::UniqueCommandBuffer& cmd_buf,
                       const ImagePackPtr& src_img_pack,
                       const BufferPackPtr& dst_buf_pack,
                       const vk::ImageLayout& init_layout,
                       const vk::ImageLayout& final_layout = LAYOUT_DONT_CARE,
                       const uint32_t src_miplevel = 0);

void CopyImage(const vk::UniqueCommandBuffer& cmd_buf,
               const ImagePackPtr& src_img_pack,
               const ImagePackPtr& dst_img_pack,
               const vk::ImageLayout& src_init_layout,
               const vk::ImageLayout& dst_init_layout,
               const vk::ImageLayout& src_final_layout = LAYOUT_DONT_CARE,
               const vk::ImageLayout& dst_final_layout = LAYOUT_DONT_CARE,
               const uint32_t src_miplevel = 0,
               const uint32_t dst_miplevel = 0);

void BlitImage(const vk::UniqueCommandBuffer& cmd_buf,
               const ImagePackPtr& src_img_pack,
               const ImagePackPtr& dst_img_pack,
               const std::array<vk::Offset2D, 2>& src_offsets,
               const std::array<vk::Offset2D, 2>& dst_offsets,
               const vk::ImageLayout& src_init_layout,
               const vk::ImageLayout& dst_init_layout,
               const vk::ImageLayout& src_final_layout = LAYOUT_DONT_CARE,
               const vk::ImageLayout& dst_final_layout = LAYOUT_DONT_CARE,
               const uint32_t src_miplevel = 0, const uint32_t dst_miplevel = 0,
               const vk::Filter& filter = vk::Filter::eLinear);

void ClearColorImage(const vk::UniqueCommandBuffer& cmd_buf,
                     const ImagePackPtr& dst_img_pack,
                     const vk::ClearColorValue& color,
                     const vk::ImageLayout& init_layout,
                     const vk::ImageLayout& final_layout = LAYOUT_DONT_CARE,
                     const uint32_t dst_miplevel = 0,
                     const uint32_t miplevel_cnt = 1);

// -----------------------------------------------------------------------------
// ------------------------------ Buffer & Image -------------------------------
// -----------------------------------------------------------------------------
std::tuple<BufferPackPtr, ImagePackPtr> CreateBufferImagePack(
        const vk::PhysicalDevice& physical_device,
        const vk::UniqueDevice& device,
        const bool strict_size_check = true,  // check memory size matching
        const vk::DeviceSize& buf_size = 256 * 256 * 4,
        const vk::Format& format = vk::Format::eR8G8B8A8Uint,
        const vk::Extent2D& img_size = {256, 256}, uint32_t miplevel_cnt = 1,
        const vk::BufferUsageFlags& buf_usage =
                vk::BufferUsageFlagBits::eVertexBuffer,
        const vk::ImageUsageFlags& img_usage = vk::ImageUsageFlagBits::eSampled,
        const vk::MemoryPropertyFlags& mem_prop =
                vk::MemoryPropertyFlagBits::eDeviceLocal,
        bool is_tiling = false,
        const vk::ImageAspectFlags& aspects = vk::ImageAspectFlagBits::eColor,
        const vk::ImageLayout& init_layout = vk::ImageLayout::eUndefined,
        bool is_shared = false, bool is_exportable = false);

// -----------------------------------------------------------------------------
// ---------------------------------- Texture ----------------------------------
// -----------------------------------------------------------------------------
struct TexturePack {
    ImagePackPtr img_pack;
    vk::UniqueSampler sampler;
};
using TexturePackPtr = std::shared_ptr<TexturePack>;
TexturePackPtr CreateTexturePack(
        const ImagePackPtr& img_pack, const vk::UniqueDevice& device,
        const vk::Filter& mag_filter = vk::Filter::eLinear,
        const vk::Filter& min_filter = vk::Filter::eLinear,
        const vk::SamplerMipmapMode& mipmap = vk::SamplerMipmapMode::eLinear,
        const vk::SamplerAddressMode& addr_u = vk::SamplerAddressMode::eRepeat,
        const vk::SamplerAddressMode& addr_v = vk::SamplerAddressMode::eRepeat,
        const vk::SamplerAddressMode& addr_w = vk::SamplerAddressMode::eRepeat);

vk::UniqueSampler CreateSampler(
        const vk::UniqueDevice& device,
        const vk::Filter& mag_filter = vk::Filter::eLinear,
        const vk::Filter& min_filter = vk::Filter::eLinear,
        const vk::SamplerMipmapMode& mipmap = vk::SamplerMipmapMode::eLinear,
        const vk::SamplerAddressMode& addr_u = vk::SamplerAddressMode::eRepeat,
        const vk::SamplerAddressMode& addr_v = vk::SamplerAddressMode::eRepeat,
        const vk::SamplerAddressMode& addr_w = vk::SamplerAddressMode::eRepeat,
        const uint32_t& miplevel_cnt = 1);

// -----------------------------------------------------------------------------
// --------------------------------- Swapchain ---------------------------------
// -----------------------------------------------------------------------------
struct SwapchainPack {
    vk::UniqueSwapchainKHR swapchain;
    std::vector<ImagePackPtr> imgs;
    vk::Extent2D size;
};
using SwapchainPackPtr = std::shared_ptr<SwapchainPack>;
SwapchainPackPtr CreateSwapchainPack(
        const vk::PhysicalDevice& physical_device,
        const vk::UniqueDevice& device, const vk::UniqueSurfaceKHR& surface,
        const vk::Format& surface_format = vk::Format::eUndefined,
        const vk::ImageUsageFlags& usage =
                vk::ImageUsageFlagBits::eColorAttachment,
        const vk::PresentModeKHR& present_mode = vk::PresentModeKHR::eFifo,
        const SwapchainPackPtr& old_swapchain_pack = nullptr);

uint32_t AcquireNextImage(const vk::UniqueDevice& device,
                          const SwapchainPackPtr& swapchain_pack,
                          const SemaphorePtr& signal_semaphore = nullptr,
                          const FencePtr& signal_fence = nullptr,
                          uint64_t timeout = NO_TIMEOUT);

// -----------------------------------------------------------------------------
// ------------------------------- DescriptorSet -------------------------------
// -----------------------------------------------------------------------------
using DescCnt = uint32_t;
using DescSetInfo =
        std::tuple<vk::DescriptorType, DescCnt, vk::ShaderStageFlags>;
struct DescSetPack {
    vk::UniqueDescriptorSetLayout desc_set_layout;
    vk::UniqueDescriptorPool desc_pool;
    vk::UniqueDescriptorSet desc_set;
    std::vector<DescSetInfo> desc_set_info;
};
using DescSetPackPtr = std::shared_ptr<DescSetPack>;
DescSetPackPtr CreateDescriptorSetPack(
        const vk::UniqueDevice& device,
        const std::vector<DescSetInfo>& binding_infos);

struct WriteDescSetPack {
    std::vector<vk::WriteDescriptorSet> write_desc_sets;
    std::vector<std::vector<vk::DescriptorBufferInfo>> desc_buf_info_vecs;
    std::vector<std::vector<vk::DescriptorImageInfo>> desc_img_info_vecs;
};
using WriteDescSetPackPtr = std::shared_ptr<WriteDescSetPack>;
WriteDescSetPackPtr CreateWriteDescSetPack();

void AddWriteDescSet(WriteDescSetPackPtr& write_pack,
                     const DescSetPackPtr& desc_set_pack,
                     const uint32_t binding_idx,
                     const std::vector<BufferPackPtr>& buf_packs);  // For buf
void AddWriteDescSet(WriteDescSetPackPtr& write_pack,
                     const DescSetPackPtr& desc_set_pack,
                     const uint32_t binding_idx,
                     const std::vector<TexturePackPtr>& tex_packs,  // For tex
                     const std::vector<vk::ImageLayout>& tex_layouts);
void AddWriteDescSet(WriteDescSetPackPtr& write_pack,
                     const DescSetPackPtr& desc_set_pack,
                     const uint32_t binding_idx,
                     const std::vector<ImagePackPtr>& img_packs,  // For img
                     const std::vector<vk::ImageLayout>& img_layouts);
void AddWriteDescSet(WriteDescSetPackPtr& write_pack,
                     const DescSetPackPtr& desc_set_pack,
                     const uint32_t binding_idx,  // for raw buf
                     const std::vector<vk::DescriptorBufferInfo>& buf_infos);
void AddWriteDescSet(WriteDescSetPackPtr& write_pack,
                     const DescSetPackPtr& desc_set_pack,
                     const uint32_t binding_idx,  // for raw tex, img
                     const std::vector<vk::DescriptorImageInfo>& img_infos);

void UpdateDescriptorSets(const vk::UniqueDevice& device,
                          const WriteDescSetPackPtr& write_desc_set_pack);

// -----------------------------------------------------------------------------
// -------------------------------- RenderPass ---------------------------------
// -----------------------------------------------------------------------------
struct RenderPassPack {
    std::vector<vk::AttachmentDescription> attachment_descs;
    std::vector<std::vector<vk::AttachmentReference>> attachment_ref_vecs;
    std::vector<vk::SubpassDescription> subpass_descs;
    std::vector<vk::SubpassDependency> subpass_depends;
    vk::UniqueRenderPass render_pass;
};
using RenderPassPackPtr = std::shared_ptr<RenderPassPack>;
RenderPassPackPtr CreateRenderPassPack();

void AddAttachientDesc(
        RenderPassPackPtr& render_pass_pack,
        const vk::Format& format = vk::Format::eB8G8R8A8Unorm,
        const vk::ImageLayout& init_layout = vk::ImageLayout::eUndefined,
        const vk::ImageLayout& final_layout = vk::ImageLayout::ePresentSrcKHR,
        const vk::AttachmentLoadOp& load_op = vk::AttachmentLoadOp::eClear,
        const vk::AttachmentStoreOp& store_op = vk::AttachmentStoreOp::eStore);

using AttachmentIdx = uint32_t;
using AttachmentRefInfo = std::tuple<AttachmentIdx, vk::ImageLayout>;
void AddSubpassDesc(RenderPassPackPtr& render_pass_pack,
                    const std::vector<AttachmentRefInfo>& inp_attach_refs,
                    const std::vector<AttachmentRefInfo>& col_attach_refs,
                    const AttachmentRefInfo& depth_stencil_attach_ref = {
                            uint32_t(~0), vk::ImageLayout::eUndefined});

struct DependInfo {
    uint32_t subpass_idx;
    vk::PipelineStageFlags stage_mask;
    vk::AccessFlags access_mask;
};
void AddSubpassDepend(RenderPassPackPtr& render_pass_pack,
                      const DependInfo& src_depend,
                      const DependInfo& dst_depend,
                      const vk::DependencyFlags& depend_flags = {});

void UpdateRenderPass(const vk::UniqueDevice& device,
                      RenderPassPackPtr& render_pass_pack);

// -----------------------------------------------------------------------------
// -------------------------------- FrameBuffer --------------------------------
// -----------------------------------------------------------------------------
struct FrameBufferPack {
    vk::UniqueFramebuffer frame_buffer;
    uint32_t width;
    uint32_t height;
    uint32_t n_layers;
};
using FrameBufferPackPtr = std::shared_ptr<FrameBufferPack>;

FrameBufferPackPtr CreateFrameBuffer(const vk::UniqueDevice& device,
                                     const RenderPassPackPtr& render_pass_pack,
                                     const std::vector<ImagePackPtr>& imgs);
FrameBufferPackPtr CreateFrameBuffer(const vk::UniqueDevice& device,
                                     const RenderPassPackPtr& render_pass_pack,
                                     const std::vector<vk::ImageView>& views,
                                     const vk::Extent2D& view_size);

std::vector<FrameBufferPackPtr> CreateFrameBuffers(
        const vk::UniqueDevice& device,
        const RenderPassPackPtr& render_pass_pack,
        const std::vector<ImagePackPtr>& imgs,
        const SwapchainPackPtr& swapchain);

// -----------------------------------------------------------------------------
// -------------------------------- ShaderModule -------------------------------
// -----------------------------------------------------------------------------
struct ShaderModulePack {
    vk::UniqueShaderModule shader_module;
    vk::ShaderStageFlagBits stage;
    size_t spv_size = 0;
};
using ShaderModulePackPtr = std::shared_ptr<ShaderModulePack>;

class GLSLCompiler {
public:
    GLSLCompiler(bool enable_optim = true, bool enable_optim_size = true,
                 bool enable_gen_debug = false, bool enable_cache = true);
    ~GLSLCompiler();

    ShaderModulePackPtr compileFromString(
            const vk::UniqueDevice& device, const std::string& source,
            const vk::ShaderStageFlagBits& stage =
                    vk::ShaderStageFlagBits::eVertex) const;
    void clearCache();

    // Compile flags
    bool enable_optim;
    bool enable_optim_size;
    bool enable_gen_debug;
    // Feature flags
    bool enable_cache;

private:
    using CacheKey = std::tuple<std::string, vk::ShaderStageFlagBits>;
    mutable std::map<CacheKey, std::vector<uint32_t>> m_spv_data_cache;

    static std::atomic<uint32_t> s_instance_cnt;
};

// -----------------------------------------------------------------------------
// ------------------------------ Pipeline Cache -------------------------------
// -----------------------------------------------------------------------------
using PipelineCachePtr = std::shared_ptr<vk::UniquePipelineCache>;
PipelineCachePtr CreatePipelineCache(const vk::UniqueDevice& device);

// -----------------------------------------------------------------------------
// ----------------------------- Graphics Pipeline -----------------------------
// -----------------------------------------------------------------------------
struct VtxInputBindingInfo {
    uint32_t binding_idx = 0;
    uint32_t stride = sizeof(float);
    vk::VertexInputRate input_rate = vk::VertexInputRate::eVertex;
};
struct VtxInputAttribInfo {
    uint32_t location = 0;     // Location in shader
    uint32_t binding_idx = 0;  // Select a binding
    vk::Format format = vk::Format::eR32G32B32A32Sfloat;
    uint32_t offset = 0;  // Offset in the binding buffer
};

struct PipelineColorBlendAttachInfo {
    bool blend_enable = false;
    vk::BlendFactor blend_src_col_factor = vk::BlendFactor::eSrcAlpha;
    vk::BlendFactor blend_dst_col_factor = vk::BlendFactor::eOneMinusSrcAlpha;
    vk::BlendOp blend_color_op = vk::BlendOp::eAdd;
    vk::BlendFactor blend_src_alpha_factor = vk::BlendFactor::eSrcAlpha;
    vk::BlendFactor blend_dst_alpha_factor = vk::BlendFactor::eOneMinusSrcAlpha;
    vk::BlendOp blend_alpha_op = vk::BlendOp::eAdd;
    vk::ColorComponentFlags blend_write_mask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
};
struct PipelineInfo {
    vk::PrimitiveTopology prim_type = vk::PrimitiveTopology::eTriangleList;
    vk::CullModeFlags face_culling = vk::CullModeFlagBits::eBack;
    float line_width = 1.f;
    // Depth
    bool depth_test_enable = true;
    bool depth_write_enable = true;
    vk::CompareOp depth_comp_op = vk::CompareOp::eLessOrEqual;
    // Blend (Must be same number as color attachments) TODO: Automatic
    std::vector<PipelineColorBlendAttachInfo> color_blend_infos;
};

struct PipelinePack {
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;
};
using PipelinePackPtr = std::shared_ptr<PipelinePack>;
PipelinePackPtr CreateGraphicsPipeline(
        const vk::UniqueDevice& device,
        const std::vector<ShaderModulePackPtr>& shader_modules,
        const std::vector<VtxInputBindingInfo>& vtx_inp_binding_info,
        const std::vector<VtxInputAttribInfo>& vtx_inp_attrib_info,
        const PipelineInfo& pipeline_info,
        const std::vector<DescSetPackPtr>& desc_set_packs,
        const RenderPassPackPtr& render_pass_pack, uint32_t subpass_idx = 0,
        const PipelineCachePtr& pipeline_cache = nullptr);

// -----------------------------------------------------------------------------
// ----------------------------- Compute Pipeline ------------------------------
// -----------------------------------------------------------------------------
PipelinePackPtr CreateComputePipeline(
        const vk::UniqueDevice& device,
        const ShaderModulePackPtr& shader_module,
        const std::vector<DescSetPackPtr>& desc_set_packs,
        const PipelineCachePtr& pipeline_cache = nullptr);

// -----------------------------------------------------------------------------
// ------------------------------- Command Buffer ------------------------------
// -----------------------------------------------------------------------------
struct CommandBuffersPack {
    vk::UniqueCommandPool pool;
    std::vector<vk::UniqueCommandBuffer> cmd_bufs;
};
using CommandBuffersPackPtr = std::shared_ptr<CommandBuffersPack>;
CommandBuffersPackPtr CreateCommandBuffersPack(const vk::UniqueDevice& device,
                                               uint32_t queue_family_idx,
                                               uint32_t n_cmd_buffers = 1,
                                               bool reset_enable = true);

void BeginCommand(const vk::UniqueCommandBuffer& cmd_buf,
                  bool one_time_submit = false);
void EndCommand(const vk::UniqueCommandBuffer& cmd_buf);
void ResetCommand(const vk::UniqueCommandBuffer& cmd_buf);

void CmdBeginRenderPass(
        const vk::UniqueCommandBuffer& cmd_buf,
        const RenderPassPackPtr& render_pass_pack,
        const FrameBufferPackPtr& frame_buffer_pack,
        const std::vector<vk::ClearValue>& clear_vals,  // Resp to Attachments
        const vk::Rect2D& render_area = {});
void CmdNextSubPass(const vk::UniqueCommandBuffer& cmd_buf);
void CmdEndRenderPass(const vk::UniqueCommandBuffer& cmd_buf);

void CmdBindPipeline(const vk::UniqueCommandBuffer& cmd_buf,
                     const PipelinePackPtr& pipeline_pack,
                     const vk::PipelineBindPoint& bind_point =
                             vk::PipelineBindPoint::eGraphics);

void CmdBindDescSets(const vk::UniqueCommandBuffer& cmd_buf,
                     const PipelinePackPtr& pipeline_pack,
                     const std::vector<DescSetPackPtr>& desc_set_packs,
                     const std::vector<uint32_t>& dynamic_offsets = {},
                     const vk::PipelineBindPoint& bind_point =
                             vk::PipelineBindPoint::eGraphics);

void CmdBindVertexBuffers(const vk::UniqueCommandBuffer& cmd_buf,
                          uint32_t binding_idx,
                          const std::vector<BufferPackPtr>& vtx_buf_packs);
void CmdBindIndexBuffer(const vk::UniqueCommandBuffer& cmd_buf,
                        const BufferPackPtr& index_buf_pack,
                        uint64_t offset = 0,
                        vk::IndexType index_type = vk::IndexType::eUint32);

void CmdSetViewport(const vk::UniqueCommandBuffer& cmd_buf,
                    const vk::Viewport& viewport);
void CmdSetViewport(const vk::UniqueCommandBuffer& cmd_buf,
                    const vk::Extent2D& viewport_size);
void CmdSetScissor(const vk::UniqueCommandBuffer& cmd_buf,
                   const vk::Rect2D& scissor);
void CmdSetScissor(const vk::UniqueCommandBuffer& cmd_buf,
                   const vk::Extent2D& scissor_size);

void CmdDraw(const vk::UniqueCommandBuffer& cmd_buf, uint32_t n_vtxs,
             uint32_t n_instances = 1, uint32_t first_vtx = 0,
             uint32_t first_instance = 0);
void CmdDrawIndexed(const vk::UniqueCommandBuffer& cmd_buf, uint32_t n_idxs,
                    uint32_t n_instances = 1, uint32_t first_idx = 0,
                    int32_t vtx_offset = 0, uint32_t first_instance = 0);

void CmdDispatch(const vk::UniqueCommandBuffer& cmd_buf, uint32_t n_group_x,
                 uint32_t n_group_y, uint32_t n_group_z = 1);

// -----------------------------------------------------------------------------
// ----------------------------------- Queue -----------------------------------
// -----------------------------------------------------------------------------
vk::Queue GetQueue(const vk::UniqueDevice& device, uint32_t queue_family_idx,
                   uint32_t queue_idx = 0);
std::vector<vk::Queue> GetQueues(const vk::UniqueDevice& device,
                                 uint32_t queue_family_idx, uint32_t n_queues);

using WaitSemaphoreInfo =
        std::tuple<SemaphorePtr, vk::PipelineStageFlags /* dst_stage */>;
void QueueSubmit(
        const vk::Queue& queue, const vk::UniqueCommandBuffer& cmd_buf,
        const FencePtr& signal_fence = nullptr,
        const std::vector<WaitSemaphoreInfo>& wait_semaphore_infos = {},
        const std::vector<SemaphorePtr>& signal_semaphores = {});

void QueuePresent(const vk::Queue& queue,
                  const SwapchainPackPtr& swapchain_pack, uint32_t img_idx,
                  const std::vector<SemaphorePtr>& wait_semaphores = {});

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
}  // namespace vkw

#endif  // end of include guard
