//! GPU compositors that render into `wgpu::Texture`s shared with Slint.
//!
//! The crate uses Slint's re-exported `wgpu` (via `slint::wgpu_28::wgpu`) so
//! the `Device`, `Queue` and `Texture` types are exactly the ones Slint
//! expects when the texture is later imported with `slint::Image::try_from`.
//!
//! Two compositors are provided:
//!
//! * [`QuadCompositor`]   — colored triangles, used as a simple smoke test.
//! * [`VideoCompositor`]  — samples an NV12 frame (Y + interleaved CbCr,
//!   typically straight off a VideoToolbox IOSurface) and converts to RGB
//!   in a single full-screen pass.

use bytemuck::{Pod, Zeroable};
use slint::wgpu_28::wgpu;

/// Texture format used for the shared render target.
///
/// Slint only accepts `Rgba8Unorm` and `Rgba8UnormSrgb` when importing a
/// `wgpu::Texture` as a `slint::Image`; we stick with linear `Rgba8Unorm`.
pub const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
            shader_location: 1,
        },
    ];

    const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

// A unit quad covering most of the framebuffer with a different color at each
// corner. The two triangles are wound counter-clockwise so the default
// `PrimitiveState` keeps both visible.
const QUAD_VERTICES: &[Vertex] = &[
    Vertex { position: [-0.8, -0.8], color: [1.0, 0.0, 0.0] },
    Vertex { position: [ 0.8, -0.8], color: [0.0, 1.0, 0.0] },
    Vertex { position: [ 0.8,  0.8], color: [0.0, 0.0, 1.0] },
    Vertex { position: [-0.8, -0.8], color: [1.0, 0.0, 0.0] },
    Vertex { position: [ 0.8,  0.8], color: [0.0, 0.0, 1.0] },
    Vertex { position: [-0.8,  0.8], color: [1.0, 1.0, 0.0] },
];

/// Renders a colored quad into a `wgpu::Texture` that can be handed to Slint.
pub struct QuadCompositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    target: wgpu::Texture,
    clear_color: wgpu::Color,
}

impl QuadCompositor {
    /// Build the pipeline using the same `Device`/`Queue` that drives Slint's renderer.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compositor.quad.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("quad.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compositor.quad.pipeline_layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("compositor.quad.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TEXTURE_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = create_vertex_buffer(device, QUAD_VERTICES);
        let target = create_target(device, 1, 1);

        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            vertex_buffer,
            target,
            clear_color: wgpu::Color { r: 0.05, g: 0.05, b: 0.08, a: 1.0 },
        }
    }

    /// Background color used to clear the render target before drawing the quad.
    pub fn set_clear_color(&mut self, color: wgpu::Color) {
        self.clear_color = color;
    }

    /// Render the quad into the (re)sized render target and return the texture.
    pub fn render(&mut self, width: u32, height: u32) -> wgpu::Texture {
        let width = width.max(1);
        let height = height.max(1);

        let size = self.target.size();
        if size.width != width || size.height != height {
            self.target = create_target(&self.device, width, height);
        }

        let view = self.target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compositor.quad.encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("compositor.quad.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..QUAD_VERTICES.len() as u32, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        self.target.clone()
    }
}

fn create_target(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("compositor.quad.target"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn create_vertex_buffer(device: &wgpu::Device, vertices: &[Vertex]) -> wgpu::Buffer {
    let bytes = bytemuck::cast_slice(vertices);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("compositor.quad.vertices"),
        size: bytes.len() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytes);
    buffer.unmap();
    buffer
}

// ---------------------------------------------------------------------------
// VideoCompositor: NV12 → RGB
// ---------------------------------------------------------------------------

/// A pair of NV12 plane textures the [`VideoCompositor`] samples through a
/// YUV → RGB shader.
///
/// `y` must be `R8Unorm` and `cbcr` must be `Rg8Unorm`. Typically these come
/// from `decoder::HwFrameTextures` (zero-copy IOSurface import), but the
/// compositor doesn't care where they were allocated as long as the formats
/// match.
pub struct Nv12Planes<'a> {
    pub y: &'a wgpu::Texture,
    pub cbcr: &'a wgpu::Texture,
}

/// Renders an NV12 video frame into an RGBA `wgpu::Texture` Slint can import.
///
/// One texture, one sampler, one full-screen triangle, one YUV → RGB
/// fragment shader. No CPU touch of pixel data.
pub struct VideoCompositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    target: wgpu::Texture,
    clear_color: wgpu::Color,
}

impl VideoCompositor {
    /// Build the pipeline using the same `Device`/`Queue` that drives Slint's renderer.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compositor.video.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("nv12.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compositor.video.bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compositor.video.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("compositor.video.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TEXTURE_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("compositor.video.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let target = create_target(device, 1, 1);

        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            bind_group_layout,
            sampler,
            target,
            clear_color: wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        }
    }

    /// Color used to clear the render target when no frame is present.
    pub fn set_clear_color(&mut self, color: wgpu::Color) {
        self.clear_color = color;
    }

    /// Render `planes` into a fresh `wgpu::Texture` of `width` x `height` and return it.
    ///
    /// The returned texture is `Rgba8Unorm` and ready to be handed to Slint
    /// via `slint::Image::try_from`.
    pub fn render(&mut self, planes: Nv12Planes<'_>, width: u32, height: u32) -> wgpu::Texture {
        let width = width.max(1);
        let height = height.max(1);

        let size = self.target.size();
        if size.width != width || size.height != height {
            self.target = create_target(&self.device, width, height);
        }

        let y_view = planes.y.create_view(&wgpu::TextureViewDescriptor {
            label: Some("compositor.video.y.view"),
            ..Default::default()
        });
        let cbcr_view = planes.cbcr.create_view(&wgpu::TextureViewDescriptor {
            label: Some("compositor.video.cbcr.view"),
            ..Default::default()
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compositor.video.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&cbcr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let view = self
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compositor.video.encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("compositor.video.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        self.target.clone()
    }
}
