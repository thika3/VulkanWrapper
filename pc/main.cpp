#include <vkw/vkw.h>

#include <cstdlib>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>

// vertex shader with (P)osition and (C)olor in and (C)olor out
const std::string VERT_SOURCE = R"(
#version 400

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (std140, binding = 0) uniform buffer
{
    mat4 mvp;
} uniformBuffer;

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 inColor;

layout (location = 0) out vec4 outColor;

void main()
{
    outColor = inColor;
    gl_Position = uniformBuffer.mvp * pos;
}
)";

// fragment shader with (C)olor in and (C)olor out
const std::string FRAG_SOURCE = R"(
#version 400

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec4 color;

layout (location = 0) out vec4 outColor;

void main()
{
    outColor = color;
}
)";

struct Vertex {
    float x, y, z, w;  // Position
    float r, g, b, a;  // Color
};
const std::vector<Vertex> CUBE_VERTICES = {
        // red face
        {-1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
        // green face
        {-1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        {-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        {-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
        // blue face
        {-1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
        // yellow face
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
        // magenta face
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
        // cyan face
        {1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
};

int main(int argc, char const* argv[]) {
    (void)argc, (void)argv;

    const std::string app_name = "app name";
    const uint32_t app_version = 1;
    uint32_t win_w = 600;
    uint32_t win_h = 600;
    const uint32_t n_queues = 2;
    const bool debug_enable = true;

    vkw::WindowPtr window = vkw::InitGLFWWindow(app_name, win_w, win_h);

    auto context = vkw::GraphicsContext::Create(app_name, app_version, n_queues,
                                                window, debug_enable);

    const auto& physical_device = context->getPhysicalDevice();
    const auto& device = context->getDevice();
    const auto& swapchain_pack = context->getSwapchainPack();
    const auto& queues = context->getQueues();
    const auto& queue_family_idx = context->getQueueFamilyIdx();
    const auto& surface_format = context->getSurfaceFormat();

    vkw::PrintInstanceLayerProps();
    vkw::PrintInstanceExtensionProps();
    vkw::PrintQueueFamilyProps(physical_device);

    const auto depth_format = vk::Format::eD16Unorm;
    auto depth_img = context->createImage(
            depth_format, swapchain_pack->size,
                               vk::ImageUsageFlagBits::eDepthStencilAttachment,
                               vk::MemoryPropertyFlagBits::eDeviceLocal,
                               vk::ImageAspectFlagBits::eDepth, true, false);

    auto uniform_buf = context->createBuffer(
            sizeof(glm::mat4),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);

#if 1
    auto desc_set_pack = vkw::CreateDescriptorSetPack(
            device, {{vk::DescriptorType::eUniformBufferDynamic, 1,
                      vk::ShaderStageFlagBits::eVertex}});
#else
    auto tex_pack = vkw::CreateTexture(
            vkw::CreateImage(physical_device, device), device);
    auto desc_set_pack = vkw::CreateDescriptorSet(
            device, {{vk::DescriptorType::eUniformBuffer, 1,
                      vk::ShaderStageFlagBits::eVertex},
                     {vk::DescriptorType::eCombinedImageSampler, 1,
                      vk::ShaderStageFlagBits::eVertex}});
#endif

    auto write_desc_set_pack = vkw::CreateWriteDescSetPack();
    vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 0,
                         {uniform_buf->getBufferPack()});
#if 0
    vkw::AddWriteDescSet(write_desc_set_pack, desc_set_pack, 1, {tex_pack});
#endif
    vkw::UpdateDescriptorSets(device, write_desc_set_pack);

    auto render_pass_pack = vkw::CreateRenderPassPack();
    vkw::AddAttachientDesc(
            render_pass_pack, surface_format, vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore, vk::ImageLayout::ePresentSrcKHR);
    vkw::AddAttachientDesc(render_pass_pack, depth_format,
                           vk::AttachmentLoadOp::eClear,
                           vk::AttachmentStoreOp::eDontCare,
                           vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vkw::AddSubpassDesc(render_pass_pack,
                        {
                                // No input attachments
                        },
                        {
                                {0, vk::ImageLayout::eColorAttachmentOptimal},
                        },
                        {1, vk::ImageLayout::eDepthStencilAttachmentOptimal});
    vkw::UpdateRenderPass(device, render_pass_pack);

    auto frame_buffer_packs = vkw::CreateFrameBuffers(
            device, render_pass_pack, {nullptr, depth_img->getImagePack()}, 0,
            swapchain_pack);

    vkw::GLSLCompiler glsl_compiler;
    auto vert_shader_module_pack = glsl_compiler.compileFromString(
            device, VERT_SOURCE, vk::ShaderStageFlagBits::eVertex);
    auto frag_shader_module_pack = glsl_compiler.compileFromString(
            device, FRAG_SOURCE, vk::ShaderStageFlagBits::eFragment);

    const size_t vertex_buf_size = CUBE_VERTICES.size() * sizeof(Vertex);
    auto vertex_buf_pack = vkw::CreateBufferPack(
            physical_device, device, vertex_buf_size,
            vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
    vkw::SendToDevice(device, vertex_buf_pack, CUBE_VERTICES.data(),
                      vertex_buf_size);

    vkw::PipelineInfo pipeline_info;
    pipeline_info.color_blend_infos.resize(1);
    auto pipeline_pack = vkw::CreatePipeline(
            device, {vert_shader_module_pack, frag_shader_module_pack},
            {{0, sizeof(Vertex), vk::VertexInputRate::eVertex}},
            {{0, 0, vk::Format::eR32G32B32A32Sfloat, 0},
             {1, 0, vk::Format::eR32G32B32A32Sfloat, 16}},
            pipeline_info, {desc_set_pack}, render_pass_pack);

    const uint32_t n_cmd_bufs = 1;
    auto cmd_bufs_pack =
            vkw::CreateCommandBuffersPack(device, queue_family_idx, n_cmd_bufs);
    auto& cmd_buf = cmd_bufs_pack->cmd_bufs[0];

    // ------------------
    const glm::mat4 model_mat = glm::mat4(1.0f);
    const glm::mat4 view_mat = glm::lookAt(glm::vec3(-5.0f, 3.0f, -10.0f),
                                           glm::vec3(0.0f, 0.0f, 0.0f),
                                           glm::vec3(0.0f, -1.0f, 0.0f));
    const glm::mat4 proj_mat =
            glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    // vulkan clip space has inverted y and half z !
    const glm::mat4 clip_mat = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,
                                0.0f, 0.0f, 0.5f, 1.0f};
    glm::mat4 rot_mat(1.f);
    while (!glfwWindowShouldClose(window.get())) {
        rot_mat = glm::rotate(0.1f, glm::vec3(1.f, 0.f, 0.f)) * rot_mat;
        glm::mat4 mvpc_mat =
                clip_mat * proj_mat * view_mat * rot_mat * model_mat;
        uniform_buf->sendToDevice(mvpc_mat);

        vkw::ResetCommand(cmd_buf);

        auto img_acquired_semaphore = vkw::CreateSemaphore(device);
        uint32_t curr_img_idx = 0;
        vkw::AcquireNextImage(&curr_img_idx, device, swapchain_pack,
                              img_acquired_semaphore, nullptr);

        vkw::BeginCommand(cmd_buf);

        const std::array<float, 4> clear_color = {0.2f, 0.2f, 0.2f, 0.2f};
        vkw::CmdBeginRenderPass(cmd_buf, render_pass_pack,
                                frame_buffer_packs[curr_img_idx],
                                {
                                        vk::ClearColorValue(clear_color),
                                        vk::ClearDepthStencilValue(1.0f, 0),
                                });

        vkw::CmdBindPipeline(cmd_buf, pipeline_pack);

        const std::vector<uint32_t> dynamic_offsets = {0};
        vkw::CmdBindDescSets(cmd_buf, pipeline_pack, {desc_set_pack},
                             dynamic_offsets);

        vkw::CmdBindVertexBuffers(cmd_buf, {vertex_buf_pack});

        vkw::CmdSetViewport(cmd_buf, swapchain_pack->size);
        vkw::CmdSetScissor(cmd_buf, swapchain_pack->size);

        const uint32_t n_instances = 1;
        vkw::CmdDraw(cmd_buf, CUBE_VERTICES.size(), n_instances);

        // vkw::CmdNextSubPass(cmd_buf);
        vkw::CmdEndRenderPass(cmd_buf);

        vkw::EndCommand(cmd_buf);

        auto draw_fence = vkw::CreateFence(device);

        vkw::QueueSubmit(queues[0], cmd_buf, draw_fence,
                         {{img_acquired_semaphore,
                           vk::PipelineStageFlagBits::eColorAttachmentOutput}},
                         {});

        vkw::QueuePresent(queues[1], swapchain_pack, curr_img_idx);

        vkw::WaitForFences(device, {draw_fence});

        glfwPollEvents();
    }

    std::cout << "exit" << std::endl;

    return 0;
}
