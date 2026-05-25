//! Zero-copy bridge from VideoToolbox `CVPixelBuffer`s (NV12, IOSurface-backed)
//! to a pair of `wgpu::Texture`s that sample the Y and CbCr planes directly.
//!
//! The path is:
//!
//! 1. `(*AVFrame).data[3]` for an `AV_PIX_FMT_VIDEOTOOLBOX` frame is a
//!    `CVPixelBufferRef`.
//! 2. `CVPixelBufferGetIOSurface(buf)` exposes the underlying IOSurface
//!    that the decoder hardware wrote into.
//! 3. We borrow the same `metal::Device` wgpu is using (via
//!    `wgpu::Device::as_hal::<Metal>()` and `hal::metal::Device::raw_device`)
//!    and call `[MTLDevice newTextureWithDescriptor:iosurface:plane:]` for
//!    each NV12 plane.
//! 4. Each resulting `MTLTexture` is wrapped back into wgpu via
//!    `wgpu::hal::metal::Device::texture_from_raw` +
//!    `wgpu::Device::create_texture_from_hal::<Metal>`.
//!
//! The resulting [`HwFrameTextures`] holds the two wgpu textures plus a
//! retained `CVPixelBuffer`. Keeping the CVPixelBuffer alive guarantees the
//! IOSurface (and therefore the GPU-visible memory the textures reference)
//! survives until the consumer drops the frame.

use std::ffi::c_void;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferGetHeight, CVPixelBufferGetIOSurface,
    CVPixelBufferGetPixelFormatType, CVPixelBufferGetPlaneCount, CVPixelBufferGetWidth,
};
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{
    MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTextureDescriptor, MTLTextureUsage,
};
use slint::wgpu_28::wgpu;

use crate::DecoderError;

/// Two `wgpu::Texture`s aliasing an NV12 IOSurface in-place plus the
/// `CVPixelBuffer` that keeps the underlying memory alive.
///
/// * `y_texture` is `R8Unorm`, full resolution; sample `.r` to get luma.
/// * `cbcr_texture` is `Rg8Unorm`, half resolution; sample `.r` / `.g`
///   to get Cb / Cr.
/// * `_pixel_buffer` holds a `CFRetain` on the source `CVPixelBuffer` so
///   that the IOSurface is not recycled by VideoToolbox while the GPU is
///   still reading from it.
pub struct HwFrameTextures {
    pub y_texture: wgpu::Texture,
    pub cbcr_texture: wgpu::Texture,
    pub width: u32,
    pub height: u32,
    pub pixel_format: u32,
    /// Keeps the underlying `CVPixelBuffer` (and its IOSurface) alive for
    /// at least as long as the wgpu textures.
    _pixel_buffer: Retained<CVPixelBuffer>,
}

impl std::fmt::Debug for HwFrameTextures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HwFrameTextures")
            .field("width", &self.width)
            .field("height", &self.height)
            .field(
                "pixel_format",
                &format_args!("{:#010x}", self.pixel_format),
            )
            .finish()
    }
}

/// Pull the NV12 pair of textures out of a `VIDEOTOOLBOX` AVFrame.
///
/// `cv_pixel_buffer_ptr` must be `(*av_frame).data[3]` for a frame whose
/// `format` is `AV_PIX_FMT_VIDEOTOOLBOX`. Anything else is undefined behaviour.
///
/// `device` and the underlying wgpu/Metal device this function inspects via
/// `as_hal::<Metal>` must be the same one used by Slint, so the resulting
/// textures can be sampled by any pipeline created against that device.
pub fn extract_nv12_textures(
    cv_pixel_buffer_ptr: *mut c_void,
    device: &wgpu::Device,
) -> Result<HwFrameTextures, DecoderError> {
    if cv_pixel_buffer_ptr.is_null() {
        return Err(DecoderError::HwFrameMissingPixelBuffer);
    }

    // SAFETY: We trust the caller that this pointer is a CVPixelBufferRef.
    // Retain it ourselves so the IOSurface outlives the local AVFrame.
    let pixel_buffer: Retained<CVPixelBuffer> = unsafe {
        let raw = cv_pixel_buffer_ptr as *const CVPixelBuffer;
        Retained::retain(raw as *mut CVPixelBuffer)
            .ok_or(DecoderError::HwFrameMissingPixelBuffer)?
    };

    let width = CVPixelBufferGetWidth(&pixel_buffer) as u32;
    let height = CVPixelBufferGetHeight(&pixel_buffer) as u32;
    let plane_count = CVPixelBufferGetPlaneCount(&pixel_buffer);
    let pixel_format = CVPixelBufferGetPixelFormatType(&pixel_buffer);

    if plane_count < 2 {
        return Err(DecoderError::HwFrameUnsupportedPixelFormat {
            pixel_format,
            plane_count,
        });
    }

    let iosurface = CVPixelBufferGetIOSurface(Some(&pixel_buffer))
        .ok_or(DecoderError::HwFrameMissingIoSurface)?;
    let iosurface: &IOSurfaceRef = &iosurface;

    // SAFETY: As long as we keep `_hal_guard` borrowed we hold the wgpu hal
    // device alive. The metal::Device pointer is the exact one wgpu uses, so
    // every texture we mint from it is allocator-compatible with wgpu.
    let hal_guard = unsafe {
        device
            .as_hal::<wgpu::hal::api::Metal>()
            .ok_or(DecoderError::NotMetalDevice)?
    };
    let metal_device: &metal::Device = hal_guard.raw_device();
    let objc2_device: &ProtocolObject<dyn MTLDevice> = unsafe {
        // metal::Device wraps the same `*mut MTLDevice` Objective-C object
        // that objc2-metal `ProtocolObject<dyn MTLDevice>` describes.
        // `ProtocolObject<P>` is repr(transparent) over the underlying
        // `AnyObject`, so the cast is layout-compatible.
        let raw = metal::foreign_types::ForeignType::as_ptr(metal_device);
        &*(raw as *const ProtocolObject<dyn MTLDevice>)
    };

    // NV12 plane layout:
    //   plane 0: Y  (R8Unorm,  width x height)
    //   plane 1: CbCr interleaved (Rg8Unorm, width/2 x height/2)
    let y_texture = build_iosurface_texture(
        device,
        objc2_device,
        iosurface,
        0,
        MTLPixelFormat::R8Unorm,
        wgpu::TextureFormat::R8Unorm,
        width,
        height,
        "decoder.iosurface.y",
    )?;

    let cbcr_texture = build_iosurface_texture(
        device,
        objc2_device,
        iosurface,
        1,
        MTLPixelFormat::RG8Unorm,
        wgpu::TextureFormat::Rg8Unorm,
        width / 2,
        height / 2,
        "decoder.iosurface.cbcr",
    )?;

    // Release the hal device borrow before we hand the wgpu textures back.
    drop(hal_guard);

    Ok(HwFrameTextures {
        y_texture,
        cbcr_texture,
        width,
        height,
        pixel_format,
        _pixel_buffer: pixel_buffer,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_iosurface_texture(
    wgpu_device: &wgpu::Device,
    mtl_device: &ProtocolObject<dyn MTLDevice>,
    iosurface: &IOSurfaceRef,
    plane: usize,
    mtl_format: MTLPixelFormat,
    wgpu_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    label: &'static str,
) -> Result<wgpu::Texture, DecoderError> {
    let width = width.max(1);
    let height = height.max(1);

    // Build a 2D, mip-less, IOSurface-backed texture descriptor. The
    // documented requirement is `MTLStorageModeShared` (UMA on Apple Silicon
    // makes that a free choice; on discrete Intel it picks the right path).
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setPixelFormat(mtl_format);
    // SAFETY: setters mutate the fresh Retained descriptor we just allocated.
    unsafe {
        descriptor.setWidth(width as usize);
        descriptor.setHeight(height as usize);
        descriptor.setDepth(1);
        descriptor.setMipmapLevelCount(1);
        descriptor.setArrayLength(1);
    }
    descriptor.setUsage(MTLTextureUsage::ShaderRead);
    descriptor.setStorageMode(MTLStorageMode::Shared);

    let mtl_texture = mtl_device
        .newTextureWithDescriptor_iosurface_plane(&descriptor, iosurface, plane)
        .ok_or(DecoderError::MtlTextureFromIosurfaceFailed { plane })?;

    // Hand the +1 retain off to the metal crate so its Drop will balance us.
    let raw_ptr = Retained::into_raw(mtl_texture);
    let metal_texture: metal::Texture = unsafe {
        <metal::Texture as metal::foreign_types::ForeignType>::from_ptr(
            raw_ptr as *mut metal::MTLTexture,
        )
    };

    let hal_texture = unsafe {
        wgpu::hal::metal::Device::texture_from_raw(
            metal_texture,
            wgpu_format,
            metal::MTLTextureType::D2,
            1,
            1,
            wgpu::hal::CopyExtent {
                width,
                height,
                depth: 1,
            },
        )
    };

    let texture = unsafe {
        wgpu_device.create_texture_from_hal::<wgpu::hal::api::Metal>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_format,
                // TEXTURE_BINDING is what the compositor needs for sampling.
                // COPY_SRC lets tests blit Y/CbCr bytes back to the CPU.
                // Both are no-ops on Metal once the underlying MTLTexture is
                // created — we just have to tell wgpu about them.
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
    };

    Ok(texture)
}
