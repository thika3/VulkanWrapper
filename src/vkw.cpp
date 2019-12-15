#include "vkw.h"

#include "warning_suppressor.h"

VKW_SUPPRESS_WARNING_PUSH
#include <SPIRV/GlslangToSpv.h>
VKW_SUPPRESS_WARNING_POP

#include <iostream>
#include <stdexcept>

// Storage for dispatcher
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace vkw {

namespace {

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
template <typename T>
inline T Clamp(const T &x, const T &min_v, const T &max_v) {
    return std::min(std::max(x, min_v), max_v);
}

template <typename BitType, typename MaskType = VkFlags>
bool IsSufficient(const vk::Flags<BitType, MaskType> &actual_flags,
                  const vk::Flags<BitType, MaskType> &require_flags) {
    return (actual_flags & require_flags) == require_flags;
}

template <typename T>
T &EmplaceBackEmpty(std::vector<T> &vec) {
    vec.emplace_back();
    return vec.back();
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
static VkBool32 DebugMessengerCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity,
        VkDebugUtilsMessageTypeFlagsEXT msg_types,
        VkDebugUtilsMessengerCallbackDataEXT const *callback, void *) {
    // Create corresponding strings
    const std::string &severity_str =
            vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(
                    msg_severity));
    const std::string &type_str = vk::to_string(
            static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(msg_types));

    // Print messages
    std::cerr << "-----------------------------------------------" << std::endl;
    std::cerr << severity_str << ": " << type_str << ":" << std::endl;
    std::cerr << "  Message ID Name (number) = <" << callback->pMessageIdName
              << "> (" << callback->messageIdNumber << ")" << std::endl;
    std::cerr << "  Message = \"" << callback->pMessage << "\"" << std::endl;
    if (0 < callback->queueLabelCount) {
        std::cerr << "  Queue Labels:" << std::endl;
        for (uint8_t i = 0; i < callback->queueLabelCount; i++) {
            const auto &name = callback->pQueueLabels[i].pLabelName;
            std::cerr << "    " << i << ": " << name << std::endl;
        }
    }
    if (0 < callback->cmdBufLabelCount) {
        std::cerr << "  CommandBuffer Labels:" << std::endl;
        for (uint8_t i = 0; i < callback->cmdBufLabelCount; i++) {
            const auto &name = callback->pCmdBufLabels[i].pLabelName;
            std::cerr << "    " << i << ": " << name << std::endl;
        }
    }
    if (0 < callback->objectCount) {
        std::cerr << "  Objects:" << std::endl;
        for (uint8_t i = 0; i < callback->objectCount; i++) {
            const auto &type = vk::to_string(static_cast<vk::ObjectType>(
                    callback->pObjects[i].objectType));
            const auto &handle = callback->pObjects[i].objectHandle;
            std::cerr << "    " << static_cast<int>(i) << ":" << std::endl;
            std::cerr << "      objectType   = " << type << std::endl;
            std::cerr << "      objectHandle = " << handle << std::endl;
            if (callback->pObjects[i].pObjectName) {
                const auto &on = callback->pObjects[i].pObjectName;
                std::cerr << "      objectName   = <" << on << ">" << std::endl;
            }
        }
    }
    std::cerr << "-----------------------------------------------" << std::endl;
    return VK_TRUE;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
static auto SelectSwapchainProps(const vk::PhysicalDevice &physical_device,
                                 const vk::UniqueSurfaceKHR &surface,
                                 uint32_t win_w, uint32_t win_h) {
    // Get all capabilities
    const vk::SurfaceCapabilitiesKHR &surface_capas =
            physical_device.getSurfaceCapabilitiesKHR(surface.get());

    // Get the surface extent size
    VkExtent2D swapchain_extent = surface_capas.currentExtent;
    if (swapchain_extent.width == std::numeric_limits<uint32_t>::max()) {
        // If the surface size is undefined, setting screen size is requested.
        const auto &min_ex = surface_capas.minImageExtent;
        const auto &max_ex = surface_capas.maxImageExtent;
        swapchain_extent.width = Clamp(win_w, min_ex.width, max_ex.width);
        swapchain_extent.height = Clamp(win_h, min_ex.height, max_ex.height);
    }

    // Set swapchain pre-transform
    vk::SurfaceTransformFlagBitsKHR pre_trans = surface_capas.currentTransform;
    if (surface_capas.supportedTransforms &
        vk::SurfaceTransformFlagBitsKHR::eIdentity) {
        pre_trans = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    }

    // Set swapchain composite alpha
    vk::CompositeAlphaFlagBitsKHR composite_alpha;
    const auto &suppored_flag = surface_capas.supportedCompositeAlpha;
    if (suppored_flag & vk::CompositeAlphaFlagBitsKHR::ePreMultiplied) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::ePreMultiplied;
    } else if (suppored_flag & vk::CompositeAlphaFlagBitsKHR::ePostMultiplied) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::ePostMultiplied;
    } else if (suppored_flag & vk::CompositeAlphaFlagBitsKHR::eInherit) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::eInherit;
    } else {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    }

    const uint32_t min_img_cnt = surface_capas.minImageCount;

    return std::make_tuple(swapchain_extent, pre_trans, composite_alpha,
                           min_img_cnt);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
vk::UniqueDeviceMemory AllocMemory(
        const vk::UniqueDevice &device,
        const vk::PhysicalDevice &physical_device,
        const vk::MemoryRequirements &memory_requs,
        const vk::MemoryPropertyFlags &require_flags) {
    auto actual_memory_props = physical_device.getMemoryProperties();
    uint32_t type_bits = memory_requs.memoryTypeBits;
    uint32_t type_idx = uint32_t(~0);
    for (uint32_t i = 0; i < actual_memory_props.memoryTypeCount; i++) {
        if ((type_bits & 1) &&
            IsSufficient(actual_memory_props.memoryTypes[i].propertyFlags,
                         require_flags)) {
            type_idx = i;
            break;
        }
        type_bits >>= 1;
    }
    if (type_idx == uint32_t(~0)) {
        throw std::runtime_error("Failed to allocate requested memory");
    }

    return device->allocateMemoryUnique({memory_requs.size, type_idx});
}

static vk::UniqueImageView CreateImageView(const vk::Image &img,
                                           const vk::Format &format,
                                           const vk::ImageAspectFlags &aspects,
                                           const vk::UniqueDevice &device) {
    const vk::ComponentMapping comp_mapping(
            vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
            vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);
    vk::ImageSubresourceRange sub_res_range(aspects, 0, 1, 0, 1);
    auto view = device->createImageViewUnique({vk::ImageViewCreateFlags(), img,
                                               vk::ImageViewType::e2D, format,
                                               comp_mapping, sub_res_range});
    return view;
}

static void SendToDevice(const vk::UniqueDevice &device,
                         const vk::UniqueDeviceMemory &dev_mem,
                         const vk::DeviceSize &dev_mem_size, const void *data,
                         uint64_t n_bytes) {
    uint8_t *dev_p = static_cast<uint8_t *>(
            device->mapMemory(dev_mem.get(), 0, dev_mem_size));
    memcpy(dev_p, data, n_bytes);
    device->unmapMemory(dev_mem.get());
}

static vk::UniqueSampler CreateSampler(const vk::UniqueDevice &device,
                                       const vk::Filter &mag_filter,
                                       const vk::Filter &min_filter,
                                       const vk::SamplerMipmapMode &mipmap,
                                       const vk::SamplerAddressMode &addr_u,
                                       const vk::SamplerAddressMode &addr_v,
                                       const vk::SamplerAddressMode &addr_w) {
    const float mip_lod_bias = 0.f;
    const bool anisotropy_enable = false;
    const float max_anisotropy = 16.f;
    const bool compare_enable = false;
    const vk::CompareOp compare_op = vk::CompareOp::eNever;
    const float min_lod = 0.f;
    const float max_lod = 0.f;
    const vk::BorderColor border_color = vk::BorderColor::eFloatOpaqueBlack;

    return device->createSamplerUnique(
            {vk::SamplerCreateFlags(), mag_filter, min_filter, mipmap, addr_u,
             addr_v, addr_w, mip_lod_bias, anisotropy_enable, max_anisotropy,
             compare_enable, compare_op, min_lod, max_lod, border_color});
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
auto PrepareFrameBuffer(const RenderPassPackPtr &render_pass_pack,
                        const std::vector<ImagePackPtr> &imgs,
                        const vk::Extent2D &size_org) {
    // Check the number of input images
    const auto &attachment_descs = render_pass_pack->attachment_descs;
    if (attachment_descs.size() != imgs.size() || imgs.empty()) {
        throw std::runtime_error("n_imgs != n_att_descs (FrameBuffer)");
    }

    // Extract size from the first image
    vk::Extent2D size = size_org;
    if (size.width == 0 || size.height == 0) {
        size = imgs[0]->size;
    }

    // Check image sizes
    for (uint32_t i = 0; i < imgs.size(); i++) {
        if (imgs[i] && imgs[i]->size != size) {
            throw std::runtime_error("Image sizes are not match (FrameBuffer)");
        }
    }

    // Create attachments
    std::vector<vk::ImageView> attachments;
    attachments.reserve(imgs.size());
    for (auto &&img : imgs) {
        if (img) {
            attachments.push_back(*img->view);
        } else {
            attachments.emplace_back();
        }
    }
    return std::make_tuple(size, std::move(attachments));
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
}  // namespace

// -----------------------------------------------------------------------------
// ----------------------------------- GLFW ------------------------------------
// -----------------------------------------------------------------------------
UniqueGLFWWindow InitGLFWWindow(const std::string &win_name, uint32_t win_w,
                                uint32_t win_h) {
    // Initialize GLFW
    glfwInit();
    atexit([]() { glfwTerminate(); });

    // Create GLFW window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    UniqueGLFWWindow window(
            glfwCreateWindow(static_cast<int>(win_w), static_cast<int>(win_h),
                             win_name.c_str(), nullptr, nullptr));
    if (!glfwVulkanSupported()) {
        throw std::runtime_error("No Vulkan support");
    }
    return window;
}

// -----------------------------------------------------------------------------
// --------------------------------- Instance ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueInstance CreateInstance(const std::string &app_name,
                                  uint32_t app_version,
                                  const std::string &engine_name,
                                  uint32_t engine_version, bool debug_enable) {
    // Print extension names required by GLFW
    uint32_t n_glfw_exts = 0;
    const char **glfw_exts = glfwGetRequiredInstanceExtensions(&n_glfw_exts);

    // Initialize dispatcher with `vkGetInstanceProcAddr`, to get the instance
    // independent function pointers
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
            vk::DynamicLoader().getProcAddress<PFN_vkGetInstanceProcAddr>(
                    "vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    // Create a Vulkan instance
    std::vector<char const *> enabled_layer = {"VK_LAYER_KHRONOS_validation"};
    std::vector<char const *> enabled_exts = {
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    for (uint32_t i = 0; i < n_glfw_exts; i++) {
        enabled_exts.push_back(glfw_exts[i]);
    }
    vk::ApplicationInfo app_info = {app_name.c_str(), app_version,
                                    engine_name.c_str(), engine_version,
                                    VK_API_VERSION_1_1};
    vk::UniqueInstance instance = vk::createInstanceUnique(
            {vk::InstanceCreateFlags(), &app_info,
             static_cast<uint32_t>(enabled_layer.size()), enabled_layer.data(),
             static_cast<uint32_t>(enabled_exts.size()), enabled_exts.data()});

    // Initialize dispatcher with Instance to get all the other function ptrs.
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

    if (debug_enable) {
        // Create debug messenger
        vk::UniqueDebugUtilsMessengerEXT debug_messenger =
                instance->createDebugUtilsMessengerEXTUnique(
                        {{},
                         {vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                          vk::DebugUtilsMessageSeverityFlagBitsEXT::eError},
                         {vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                          vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                          vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation},
                         &DebugMessengerCallback});
    }

    return instance;
}

// -----------------------------------------------------------------------------
// ------------------------------ PhysicalDevice -------------------------------
// -----------------------------------------------------------------------------
std::vector<vk::PhysicalDevice> GetPhysicalDevices(
        const vk::UniqueInstance &instance) {
    return instance->enumeratePhysicalDevices();
}

// -----------------------------------------------------------------------------
// ---------------------------------- Surface ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueSurfaceKHR CreateSurface(const vk::UniqueInstance &instance,
                                   const UniqueGLFWWindow &window) {
    // Create a window surface (GLFW)
    VkSurfaceKHR s_raw = nullptr;
    VkResult err =
            glfwCreateWindowSurface(*instance, window.get(), nullptr, &s_raw);
    if (err) {
        throw std::runtime_error("Failed to create window surface");
    }

    // TODO: Android (non-GLFW) version
    // vkCreateAndroidSurfaceKHR

    // Smart wrapper
    using Deleter = vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderDynamic>;
    vk::UniqueSurfaceKHR surface(s_raw, Deleter(*instance));

    return surface;
}

vk::Format GetSurfaceFormat(const vk::PhysicalDevice &physical_device,
                            const vk::UniqueSurfaceKHR &surface) {
    auto surface_formats = physical_device.getSurfaceFormatsKHR(surface.get());
    assert(!surface_formats.empty());
    vk::Format surface_format = surface_formats[0].format;
    if (surface_format == vk::Format::eUndefined) {
        surface_format = vk::Format::eB8G8R8A8Unorm;
    }
    return surface_format;
}

// -----------------------------------------------------------------------------
// -------------------------------- Queue Family -------------------------------
// -----------------------------------------------------------------------------
void PrintQueueFamilyProps(const vk::PhysicalDevice &physical_device) {
    const auto &props = physical_device.getQueueFamilyProperties();

    std::cout << "QueueFamilyProperties" << std::endl;
    for (uint32_t i = 0; i < props.size(); i++) {
        const auto prop = props[i];
        const auto &flags_str = vk::to_string(prop.queueFlags);
        const auto &max_queue_cnt = prop.queueCount;
        std::cout << "  " << i << ": " << flags_str
                  << "  (max_cnt:" << max_queue_cnt << ")" << std::endl;
    }
}

std::vector<uint32_t> GetQueueFamilyIdxs(
        const vk::PhysicalDevice &physical_device,
        const vk::QueueFlags &queue_flags) {
    const auto &props = physical_device.getQueueFamilyProperties();

    // Search sufficient queue family indices
    std::vector<uint32_t> queue_family_idxs;
    for (uint32_t i = 0; i < props.size(); i++) {
        if (IsSufficient(props[i].queueFlags, queue_flags)) {
            queue_family_idxs.push_back(i);
        }
    }
    return queue_family_idxs;
}

uint32_t GetGraphicPresentQueueFamilyIdx(
        const vk::PhysicalDevice &physical_device,
        const vk::UniqueSurfaceKHR &surface,
        const vk::QueueFlags &queue_flags) {
    // Get graphics queue family indices
    auto graphic_idxs = GetQueueFamilyIdxs(physical_device, queue_flags);

    // Extract present queue indices
    std::vector<uint32_t> graphic_present_idxs;
    for (auto &&idx : graphic_idxs) {
        if (physical_device.getSurfaceSupportKHR(idx, surface.get())) {
            graphic_present_idxs.push_back(idx);
        }
    }

    // Check status
    if (graphic_present_idxs.size() == 0) {
        throw std::runtime_error("No sufficient queue for graphic present");
    }
    // Return the first index
    return graphic_present_idxs.front();
}

// -----------------------------------------------------------------------------
// ----------------------------------- Device ----------------------------------
// -----------------------------------------------------------------------------
vk::UniqueDevice CreateDevice(uint32_t queue_family_idx,
                              const vk::PhysicalDevice &physical_device,
                              uint32_t n_queues, bool swapchain_support) {
    // Create queue create info
    std::vector<float> queue_priorites(n_queues, 0.f);
    vk::DeviceQueueCreateInfo device_queue_create_info = {
            vk::DeviceQueueCreateFlags(), queue_family_idx, n_queues,
            queue_priorites.data()};

    // Create device extension strings
    std::vector<const char *> device_exts;
    if (swapchain_support) {
        device_exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    // Create a logical device
    vk::UniqueDevice device = physical_device.createDeviceUnique(
            {vk::DeviceCreateFlags(), 1, &device_queue_create_info, 0, nullptr,
             static_cast<uint32_t>(device_exts.size()), device_exts.data()});

    // Initialize dispatcher for device
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

    return device;
}

// -----------------------------------------------------------------------------
// ------------------------------- Command Buffer ------------------------------
// -----------------------------------------------------------------------------
CommandBuffersPackPtr CreateCommandBuffersPack(const vk::UniqueDevice &device,
                                               uint32_t queue_family_idx,
                                               uint32_t n_cmd_buffers) {
    // Create a command pool
    vk::UniqueCommandPool command_pool = device->createCommandPoolUnique(
            {vk::CommandPoolCreateFlags(), queue_family_idx});

    // Allocate a command buffer from the command pool
    auto cmd_bufs = device->allocateCommandBuffersUnique(
            {command_pool.get(), vk::CommandBufferLevel::ePrimary,
             n_cmd_buffers});

    return CommandBuffersPackPtr(new CommandBuffersPack{std::move(command_pool),
                                                        std::move(cmd_bufs)});
}

// -----------------------------------------------------------------------------
// --------------------------------- Swapchain ---------------------------------
// -----------------------------------------------------------------------------
SwapchainPackPtr CreateSwapchainPack(const vk::PhysicalDevice &physical_device,
                                     const vk::UniqueDevice &device,
                                     const vk::UniqueSurfaceKHR &surface,
                                     uint32_t win_w, uint32_t win_h,
                                     const vk::Format &surface_format_,
                                     const vk::ImageUsageFlags &usage) {
    // Set swapchain present mode
    const vk::PresentModeKHR swapchain_present_mode = vk::PresentModeKHR::eFifo;

    // Get the supported surface VkFormats
    auto surface_format = (surface_format_ == vk::Format::eUndefined) ?
                                  GetSurfaceFormat(physical_device, surface) :
                                  surface_format_;

    // Select properties from capabilities
    auto props = SelectSwapchainProps(physical_device, surface, win_w, win_h);
    const auto &swapchain_extent = std::get<0>(props);
    const auto &pre_trans = std::get<1>(props);
    const auto &composite_alpha = std::get<2>(props);
    const auto &min_img_cnt = std::get<3>(props);

    // Create swapchain
    vk::UniqueSwapchainKHR swapchain = device->createSwapchainKHRUnique(
            {vk::SwapchainCreateFlagsKHR(), surface.get(), min_img_cnt,
             surface_format, vk::ColorSpaceKHR::eSrgbNonlinear,
             swapchain_extent, 1, usage, vk::SharingMode::eExclusive, 0,
             nullptr, pre_trans, composite_alpha, swapchain_present_mode, true,
             nullptr});

    // Create image views
    auto swapchain_imgs = device->getSwapchainImagesKHR(swapchain.get());
    std::vector<vk::UniqueImageView> img_views;
    img_views.reserve(swapchain_imgs.size());
    for (auto img : swapchain_imgs) {
        auto img_view = CreateImageView(
                img, surface_format, vk::ImageAspectFlagBits::eColor, device);
        img_views.push_back(std::move(img_view));
    }

    return SwapchainPackPtr(new SwapchainPack{
            std::move(swapchain), std::move(img_views), swapchain_extent});
}

// -----------------------------------------------------------------------------
// ----------------------------------- Image -----------------------------------
// -----------------------------------------------------------------------------
ImagePackPtr CreateImagePack(const vk::PhysicalDevice &physical_device,
                             const vk::UniqueDevice &device,
                             const vk::Format &format, const vk::Extent2D &size,
                             const vk::ImageUsageFlags &usage,
                             const vk::MemoryPropertyFlags &memory_props,
                             const vk::ImageAspectFlags &aspects,
                             bool is_staging, bool is_shared) {
    // Select tiling mode
    vk::ImageTiling tiling =
            is_staging ? vk::ImageTiling::eOptimal : vk::ImageTiling::eLinear;
    // Select sharing mode
    vk::SharingMode sharing = is_shared ? vk::SharingMode::eConcurrent :
                                          vk::SharingMode::eExclusive;

    // Create image
    auto img = device->createImageUnique(
            {vk::ImageCreateFlags(), vk::ImageType::e2D, format,
             vk::Extent3D(size, 1), 1, 1, vk::SampleCountFlagBits::e1, tiling,
             usage, sharing});

    // Allocate memory
    auto memory_requs = device->getImageMemoryRequirements(*img);
    auto device_mem =
            AllocMemory(device, physical_device, memory_requs, memory_props);

    // Bind memory
    device->bindImageMemory(img.get(), device_mem.get(), 0);

    // Create image view
    auto img_view = CreateImageView(*img, format, aspects, device);

    // Construct image pack
    return ImagePackPtr(new ImagePack{std::move(img), std::move(img_view), size,
                                      std::move(device_mem),
                                      memory_requs.size});
}

void SendToDevice(const vk::UniqueDevice &device, const ImagePackPtr &img_pack,
                  const void *data, uint64_t n_bytes) {
    SendToDevice(device, img_pack->dev_mem, img_pack->dev_mem_size, data,
                 n_bytes);
}

TexturePackPtr CreateTexturePack(const ImagePackPtr &img_pack,
                                 const vk::UniqueDevice &device,
                                 const vk::Filter &mag_filter,
                                 const vk::Filter &min_filter,
                                 const vk::SamplerMipmapMode &mipmap,
                                 const vk::SamplerAddressMode &addr_u,
                                 const vk::SamplerAddressMode &addr_v,
                                 const vk::SamplerAddressMode &addr_w) {
    // Create sampler
    auto sampler = CreateSampler(device, mag_filter, min_filter, mipmap, addr_u,
                                 addr_v, addr_w);
    // Construct texture pack
    return TexturePackPtr(new TexturePack{img_pack, std::move(sampler)});
}

// -----------------------------------------------------------------------------
// ----------------------------------- Buffer ----------------------------------
// -----------------------------------------------------------------------------
BufferPackPtr CreateBufferPack(const vk::PhysicalDevice &physical_device,
                               const vk::UniqueDevice &device,
                               const vk::DeviceSize &size,
                               const vk::BufferUsageFlags &usage,
                               const vk::MemoryPropertyFlags &memory_props) {
    // Create buffer
    auto buf =
            device->createBufferUnique({vk::BufferCreateFlags(), size, usage});

    // Allocate memory
    auto memory_requs = device->getBufferMemoryRequirements(*buf);
    auto device_mem =
            AllocMemory(device, physical_device, memory_requs, memory_props);

    // Bind memory
    device->bindBufferMemory(buf.get(), device_mem.get(), 0);

    return BufferPackPtr(new BufferPack{
            std::move(buf), size, std::move(device_mem), memory_requs.size});
}

void SendToDevice(const vk::UniqueDevice &device, const BufferPackPtr &buf_pack,
                  const void *data, uint64_t n_bytes) {
    SendToDevice(device, buf_pack->dev_mem, buf_pack->dev_mem_size, data,
                 n_bytes);
}

// -----------------------------------------------------------------------------
// ------------------------------- DescriptorSet -------------------------------
// -----------------------------------------------------------------------------
DescSetPackPtr CreateDescriptorSetPack(const vk::UniqueDevice &device,
                                       const std::vector<DescSetInfo> &infos) {
    const uint32_t n_bindings = static_cast<uint32_t>(infos.size());

    // Parse into raw array of bindings, pool sizes
    std::vector<vk::DescriptorSetLayoutBinding> bindings_raw;
    std::vector<vk::DescriptorPoolSize> poolsizes_raw;
    bindings_raw.reserve(n_bindings);
    poolsizes_raw.reserve(n_bindings);
    uint32_t desc_cnt_sum = 0;
    for (uint32_t i = 0; i < n_bindings; i++) {
        // Fetch from tuple
        const vk::DescriptorType &desc_type = std::get<0>(infos[i]);
        const uint32_t &desc_cnt = std::get<1>(infos[i]);
        const vk::ShaderStageFlags &shader_stage = std::get<2>(infos[i]);
        // Sum up descriptor count
        desc_cnt_sum += desc_cnt;
        // Push to bindings
        bindings_raw.emplace_back(i, desc_type, desc_cnt, shader_stage);
        // Push to pool sizes
        poolsizes_raw.emplace_back(desc_type, desc_cnt);
    }

    // Create DescriptorSetLayout
    auto desc_set_layout = device->createDescriptorSetLayoutUnique(
            {vk::DescriptorSetLayoutCreateFlags(), n_bindings,
             bindings_raw.data()});
    // Create DescriptorPool
    auto desc_pool = device->createDescriptorPoolUnique(
            {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, desc_cnt_sum,
             n_bindings, poolsizes_raw.data()});
    // Create DescriptorSet
    auto desc_sets = device->allocateDescriptorSetsUnique(
            {*desc_pool, 1, &*desc_set_layout});
    auto &desc_set = desc_sets[0];

    return DescSetPackPtr(new DescSetPack{std::move(desc_set_layout),
                                          std::move(desc_pool),
                                          std::move(desc_set), infos});
}

WriteDescSetPackPtr CreateWriteDescSetPack() {
    return std::make_shared<WriteDescSetPack>();
}

void AddWriteDescSet(WriteDescSetPackPtr &write_pack,
                     const DescSetPackPtr &desc_set_pack,
                     const uint32_t binding_idx,
                     const std::vector<TexturePackPtr> &tex_packs) {
    // Fetch form and check with DescSetInfo
    const DescSetInfo &desc_set_info =
            desc_set_pack->desc_set_info[binding_idx];
    const vk::DescriptorType desc_type = std::get<0>(desc_set_info);
    const uint32_t desc_cnt = std::get<1>(desc_set_info);
    if (desc_cnt != static_cast<uint32_t>(tex_packs.size())) {
        throw std::runtime_error("Invalid descriptor count to write images");
    }
    // Note: desc_type should be `vk::DescriptorType::eCombinedImageSampler`

    // Create vector of DescriptorImageInfo in the result pack
    auto &img_infos = EmplaceBackEmpty(write_pack->desc_img_info_vecs);
    // Create DescriptorImageInfo
    for (auto &&tex_pack : tex_packs) {
        img_infos.emplace_back(*tex_pack->sampler, *tex_pack->img_pack->view,
                               vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    // Create and Add WriteDescriptorSet
    write_pack->write_desc_sets.emplace_back(
            *desc_set_pack->desc_set, binding_idx, 0, desc_cnt, desc_type,
            img_infos.data(), nullptr, nullptr);
}

void AddWriteDescSet(WriteDescSetPackPtr &write_pack,
                     const DescSetPackPtr &desc_set_pack,
                     const uint32_t binding_idx,
                     const std::vector<BufferPackPtr> &buf_packs) {
    // Fetch form and check with DescSetInfo
    const DescSetInfo &desc_set_info =
            desc_set_pack->desc_set_info[binding_idx];
    const vk::DescriptorType desc_type = std::get<0>(desc_set_info);
    const uint32_t desc_cnt = std::get<1>(desc_set_info);
    if (desc_cnt != static_cast<uint32_t>(buf_packs.size())) {
        throw std::runtime_error("Invalid descriptor count to write buffers");
    }

    // Create vector of DescriptorBufferInfo in the result pack
    auto &buf_infos = EmplaceBackEmpty(write_pack->desc_buf_info_vecs);
    // Create DescriptorBufferInfo
    for (auto &&buf_pack : buf_packs) {
        buf_infos.emplace_back(*buf_pack->buf, 0, VK_WHOLE_SIZE);
    }

    // Create and Add WriteDescriptorSet
    write_pack->write_desc_sets.emplace_back(
            *desc_set_pack->desc_set, binding_idx, 0, desc_cnt, desc_type,
            nullptr, buf_infos.data(), nullptr);  // TODO: buffer view
}

void UpdateDescriptorSets(const vk::UniqueDevice &device,
                          const WriteDescSetPackPtr &write_desc_set_pack) {
    device->updateDescriptorSets(write_desc_set_pack->write_desc_sets, nullptr);
}

// -----------------------------------------------------------------------------
// -------------------------------- RenderPass ---------------------------------
// -----------------------------------------------------------------------------
RenderPassPackPtr CreateRenderPassPack() {
    return std::make_shared<RenderPassPack>();
}

void AddAttachientDesc(RenderPassPackPtr &render_pass_pack,
                       const vk::Format &format,
                       const vk::AttachmentLoadOp &load_op,
                       const vk::AttachmentStoreOp &store_op,
                       const vk::ImageLayout &final_layout) {
    const auto sample_cnt = vk::SampleCountFlagBits::e1;
    const auto stencil_load_op = vk::AttachmentLoadOp::eDontCare;
    const auto stencil_store_op = vk::AttachmentStoreOp::eDontCare;
    const auto initial_layout = vk::ImageLayout::eUndefined;

    // Add attachment description
    render_pass_pack->attachment_descs.emplace_back(
            vk::AttachmentDescriptionFlags(), format, sample_cnt, load_op,
            store_op, stencil_load_op, stencil_store_op, initial_layout,
            final_layout);
}

void AddSubpassDesc(RenderPassPackPtr &render_pass_pack,
                    const std::vector<AttachmentRefInfo> &inp_attach_ref_infos,
                    const std::vector<AttachmentRefInfo> &col_attach_ref_infos,
                    const AttachmentRefInfo &depth_stencil_attach_ref_info) {
    // Allocate reference vectors
    auto &attachment_ref_vecs = render_pass_pack->attachment_ref_vecs;
    attachment_ref_vecs.resize(attachment_ref_vecs.size() + 3);

    // Add attachment references for inputs
    auto &inp_refs = attachment_ref_vecs.end()[-3];
    for (auto &&info : inp_attach_ref_infos) {
        inp_refs.emplace_back(std::get<0>(info), std::get<1>(info));
    }

    // Add attachment references for color outputs
    auto &col_refs = attachment_ref_vecs.end()[-2];
    for (auto &&info : col_attach_ref_infos) {
        col_refs.emplace_back(std::get<0>(info), std::get<1>(info));
    }

    // Add an attachment reference for depth-stencil
    auto &dep_refs = attachment_ref_vecs.end()[-1];
    auto &dep_info = depth_stencil_attach_ref_info;
    const bool depth_empty = (std::get<0>(dep_info) == uint32_t(~0));
    if (!depth_empty) {
        dep_refs.emplace_back(std::get<0>(dep_info), std::get<1>(dep_info));
    }

    // Collect attachment sizes and pointers
    const uint32_t n_inp = static_cast<uint32_t>(inp_refs.size());
    const vk::AttachmentReference *inp_refs_data =
            inp_refs.empty() ? nullptr : inp_refs.data();
    const uint32_t n_col = static_cast<uint32_t>(col_refs.size());
    const vk::AttachmentReference *col_refs_data =
            col_refs.empty() ? nullptr : col_refs.data();
    const vk::AttachmentReference *dep_ref_data =
            dep_refs.empty() ? nullptr : dep_refs.data();
    // Unused options
    const vk::AttachmentReference *resolve_ref_data = nullptr;
    const uint32_t n_preserve_attachment = 0;
    const uint32_t *preserve_attachments_p = nullptr;

    // Add subpass description
    render_pass_pack->subpass_descs.emplace_back(
            vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics,
            n_inp, inp_refs_data, n_col, col_refs_data, resolve_ref_data,
            dep_ref_data, n_preserve_attachment, preserve_attachments_p);
}

void UpdateRenderPass(const vk::UniqueDevice &device,
                      RenderPassPackPtr &render_pass_pack) {
    const auto &att_descs = render_pass_pack->attachment_descs;
    const uint32_t n_att_descs = static_cast<uint32_t>(att_descs.size());
    const auto &sub_descs = render_pass_pack->subpass_descs;
    const uint32_t n_sub_descs = static_cast<uint32_t>(sub_descs.size());

    // Create render pass instance
    render_pass_pack->render_pass = device->createRenderPassUnique(
            {vk::RenderPassCreateFlags(), n_att_descs, att_descs.data(),
             n_sub_descs, sub_descs.data()});
}

// -----------------------------------------------------------------------------
// -------------------------------- FrameBuffer --------------------------------
// -----------------------------------------------------------------------------
vk::UniqueFramebuffer CreateFrameBuffer(
        const vk::UniqueDevice &device,
        const RenderPassPackPtr &render_pass_pack,
        const std::vector<ImagePackPtr> &imgs, const vk::Extent2D &size_org) {
    // Prepare frame buffer creation
    auto info = PrepareFrameBuffer(render_pass_pack, imgs, size_org);
    const vk::Extent2D &size = std::get<0>(info);
    const std::vector<vk::ImageView> &attachments = std::get<1>(info);
    const uint32_t n_layers = 1;

    // Create Frame Buffer
    return device->createFramebufferUnique(
            {vk::FramebufferCreateFlags(), *render_pass_pack->render_pass,
             static_cast<uint32_t>(attachments.size()), attachments.data(),
             size.width, size.height, n_layers});
}

std::vector<vk::UniqueFramebuffer> CreateFrameBuffers(
        const vk::UniqueDevice &device,
        const RenderPassPackPtr &render_pass_pack,
        const std::vector<ImagePackPtr> &imgs,
        const uint32_t swapchain_attach_idx,
        const SwapchainPackPtr &swapchain) {
    // Prepare frame buffer creation
    auto info = PrepareFrameBuffer(render_pass_pack, imgs, swapchain->size);
    const vk::Extent2D &size = std::get<0>(info);
    std::vector<vk::ImageView> &attachments = std::get<1>(info);
    const uint32_t n_layers = 1;

    // Create Frame Buffers
    std::vector<vk::UniqueFramebuffer> frame_buffers;
    frame_buffers.reserve(swapchain->views.size());
    for (auto &&view : swapchain->views) {
        // Overwrite swapchain image view
        attachments[swapchain_attach_idx] = *view;
        // Create one Frame Buffer
        frame_buffers.push_back(device->createFramebufferUnique(
                {vk::FramebufferCreateFlags(), *render_pass_pack->render_pass,
                 static_cast<uint32_t>(attachments.size()), attachments.data(),
                 size.width, size.height, n_layers}));
    }
    return frame_buffers;
}

}  // namespace vkw
