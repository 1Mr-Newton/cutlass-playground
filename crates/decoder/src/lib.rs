//! FFmpeg-backed video decoder with optional hardware acceleration that
//! writes frames straight into shared `wgpu::Texture`s.
//!
//! Two output paths are supported:
//!
//! * [`VideoDecoder::decode_next`] — software / CPU path. Frames are converted
//!   to RGBA with `libswscale` and uploaded with `queue.write_texture`.
//!   Works on any platform.
//!
//! * [`VideoDecoder::decode_next_hw`] *(macOS only)* — zero-copy path. The
//!   VideoToolbox `CVPixelBuffer` (IOSurface-backed NV12) is wrapped, plane
//!   by plane, into Metal textures, which are then imported into wgpu via
//!   `wgpu::Device::create_texture_from_hal::<Metal>`. No CPU touch, no
//!   `av_hwframe_transfer_data`, no `sws_scale`. The caller is expected to
//!   sample the returned Y / CbCr textures through a YUV → RGB shader.
//!
//! On macOS the decoder uses Apple VideoToolbox by default. The `wgpu` types
//! resolve through `slint::wgpu_28::wgpu` so produced textures are usable by
//! the `compositor` crate without extra conversion.

use std::os::raw::c_int;
use std::path::Path;
use std::ptr;
use std::sync::Once;

use ffmpeg::ffi as sys;
use ffmpeg::format::{Pixel, input};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{Context as Scaler, Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use slint::wgpu_28::wgpu;
use thiserror::Error;

pub use ffmpeg::Rational;
pub use ffmpeg_next as ffmpeg_reexport;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod hw_textures;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use hw_textures::HwFrameTextures;

/// Texture format the decoder writes. Matches `compositor::TEXTURE_FORMAT`.
pub const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Which hardware acceleration backend to try when opening a stream.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum HwAccel {
    /// Force software (CPU) decode.
    None,
    /// Pick a sensible hwaccel for the current host
    /// (VideoToolbox on macOS/iOS, otherwise software).
    #[default]
    Auto,
    /// Apple VideoToolbox.
    VideoToolbox,
}

/// The hwaccel actually negotiated for an open decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveHwAccel {
    None,
    VideoToolbox,
}

/// All errors that can come out of a decoder.
#[derive(Debug, Error)]
pub enum DecoderError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
    #[error("no video stream found in input")]
    NoVideoStream,
    #[error(
        "target wgpu texture has unsupported format {format:?}; expected Rgba8Unorm or Rgba8UnormSrgb"
    )]
    UnsupportedTextureFormat { format: wgpu::TextureFormat },
    #[error("requested hwaccel {requested:?} is not supported by codec {codec:?}")]
    HwAccelUnsupported {
        requested: HwAccel,
        codec: ffmpeg::codec::Id,
    },
    #[error("hwaccel device init failed (averror {0})")]
    HwDeviceInit(c_int),
    #[error("hwaccel frame transfer failed (averror {0})")]
    HwTransfer(c_int),
    #[error("decode_next_hw called without VideoToolbox hwaccel active (current = {0:?})")]
    HwAccelNotActive(ActiveHwAccel),
    #[error("hardware frame did not carry a CVPixelBuffer in data[3]")]
    HwFrameMissingPixelBuffer,
    #[error("hardware frame's CVPixelBuffer has no backing IOSurface")]
    HwFrameMissingIoSurface,
    #[error(
        "hardware frame pixel format {pixel_format:#010x} has {plane_count} planes; need 2 (NV12)"
    )]
    HwFrameUnsupportedPixelFormat {
        pixel_format: u32,
        plane_count: usize,
    },
    #[error("MTLDevice.newTextureWithDescriptor:iosurface:plane: returned nil for plane {plane}")]
    MtlTextureFromIosurfaceFailed { plane: usize },
    #[error("wgpu device is not running on the Metal backend; cannot wrap IOSurface textures")]
    NotMetalDevice,
}

/// Static metadata about the opened video stream.
#[derive(Debug, Clone, Copy)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub time_base: Rational,
    pub frame_rate: Option<Rational>,
    pub hwaccel: ActiveHwAccel,
}

/// Result of [`VideoDecoder::decode_next`].
#[derive(Debug, Clone, Copy)]
pub enum DecodeOutcome {
    /// A new frame was decoded and uploaded to the target texture.
    Frame {
        /// Presentation timestamp in `VideoInfo::time_base` units, if available.
        pts: Option<i64>,
    },
    /// The input stream has been fully drained.
    Eof,
}

/// Result of [`VideoDecoder::decode_next_hw`]. Owns the underlying
/// `CVPixelBuffer` for the lifetime of the wgpu textures inside.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Debug)]
pub enum HwDecodeOutcome {
    Frame {
        textures: HwFrameTextures,
        /// Presentation timestamp in `VideoInfo::time_base` units, if available.
        pts: Option<i64>,
    },
    Eof,
}

/// A pull-style video decoder that decodes one frame per call and uploads it
/// straight into a shared `wgpu::Texture`.
pub struct VideoDecoder {
    input: ffmpeg::format::context::Input,
    stream_index: usize,
    decoder: ffmpeg::decoder::Video,
    /// Owns the `AVBufferRef` for the hardware device, if any.
    /// Field is kept so it outlives the codec context.
    _hw_device: Option<HwDeviceCtx>,
    active_hwaccel: ActiveHwAccel,
    sws: Option<SwsState>,
    transfer_frame: Video,
    rgba_frame: Video,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    time_base: Rational,
    frame_rate: Option<Rational>,
    flushed: bool,
    drained: bool,
    upload_scratch: Vec<u8>,
}

struct SwsState {
    scaler: Scaler,
    src_format: Pixel,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
}

impl VideoDecoder {
    /// Open a video file and prepare to decode its best video stream into
    /// frames of `dst_width` x `dst_height` RGBA pixels.
    ///
    /// Uses [`HwAccel::Auto`] — i.e. VideoToolbox on macOS, software elsewhere.
    pub fn open(
        path: impl AsRef<Path>,
        dst_width: u32,
        dst_height: u32,
    ) -> Result<Self, DecoderError> {
        Self::open_with(path, dst_width, dst_height, HwAccel::default())
    }

    /// Open with an explicit hwaccel selection.
    pub fn open_with(
        path: impl AsRef<Path>,
        dst_width: u32,
        dst_height: u32,
        hwaccel: HwAccel,
    ) -> Result<Self, DecoderError> {
        ensure_ffmpeg_initialized();
        let path = path.as_ref();
        let input = input(&path)?;
        Self::from_input(input, dst_width, dst_height, hwaccel)
    }

    fn from_input(
        input: ffmpeg::format::context::Input,
        dst_width: u32,
        dst_height: u32,
        hwaccel: HwAccel,
    ) -> Result<Self, DecoderError> {
        let dst_width = dst_width.max(1);
        let dst_height = dst_height.max(1);

        let stream = input
            .streams()
            .best(Type::Video)
            .ok_or(DecoderError::NoVideoStream)?;
        let stream_index = stream.index();
        let time_base = stream.time_base();
        let frame_rate_raw = stream.avg_frame_rate();
        let frame_rate = if frame_rate_raw.numerator() == 0 {
            None
        } else {
            Some(frame_rate_raw)
        };

        let mut codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;

        let resolved = resolve_hwaccel(hwaccel);
        let (active_hwaccel, hw_device) = setup_hwaccel(&mut codec_ctx, hwaccel, resolved)?;

        let decoder = codec_ctx.decoder().video()?;
        let src_width = decoder.width();
        let src_height = decoder.height();

        Ok(Self {
            input,
            stream_index,
            decoder,
            _hw_device: hw_device,
            active_hwaccel,
            sws: None,
            transfer_frame: Video::empty(),
            rgba_frame: Video::empty(),
            src_width,
            src_height,
            dst_width,
            dst_height,
            time_base,
            frame_rate,
            flushed: false,
            drained: false,
            upload_scratch: Vec::new(),
        })
    }

    pub fn info(&self) -> VideoInfo {
        VideoInfo {
            width: self.dst_width,
            height: self.dst_height,
            time_base: self.time_base,
            frame_rate: self.frame_rate,
            hwaccel: self.active_hwaccel,
        }
    }

    /// Source (pre-scaling) frame dimensions reported by the codec.
    pub fn source_size(&self) -> (u32, u32) {
        (self.src_width, self.src_height)
    }

    pub fn hwaccel(&self) -> ActiveHwAccel {
        self.active_hwaccel
    }

    /// Decode the next frame and upload it to `texture`.
    ///
    /// If the texture has been resized the internal `sws` scaler is rebuilt
    /// on the next call so the caller can resize the shared compositor target
    /// freely.
    pub fn decode_next(
        &mut self,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<DecodeOutcome, DecoderError> {
        validate_target(texture)?;

        let size = texture.size();
        if size.width != self.dst_width || size.height != self.dst_height {
            self.dst_width = size.width;
            self.dst_height = size.height;
            self.sws = None;
            self.rgba_frame = Video::empty();
        }

        loop {
            let mut decoded = Video::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let pts = decoded.pts();
                    self.scale_and_upload(decoded, queue, texture)?;
                    return Ok(DecodeOutcome::Frame { pts });
                }
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => {
                    // need more input — fall through to packet pump
                }
                Err(ffmpeg::Error::Eof) => {
                    self.drained = true;
                    return Ok(DecodeOutcome::Eof);
                }
                Err(err) => return Err(err.into()),
            }

            if self.drained {
                return Ok(DecodeOutcome::Eof);
            }

            self.pump_packet()?;
        }
    }

    /// Decode the next frame and return a zero-copy pair of NV12 `wgpu::Texture`s
    /// aliasing the VideoToolbox IOSurface in place.
    ///
    /// Only usable on macOS/iOS, and only when the decoder was opened with a
    /// VideoToolbox-capable hwaccel (the default). The `device` *must* be the
    /// same wgpu device that will eventually sample the returned textures —
    /// typically the one Slint hands you in `RenderingState::RenderingSetup`.
    ///
    /// The returned [`HwDecodeOutcome::Frame`] owns the underlying
    /// `CVPixelBuffer` retain, so the textures are valid until the value is
    /// dropped. Drop the previous frame before requesting the next one to
    /// release the slot back to VideoToolbox's pool.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn decode_next_hw(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<HwDecodeOutcome, DecoderError> {
        if self.active_hwaccel != ActiveHwAccel::VideoToolbox {
            return Err(DecoderError::HwAccelNotActive(self.active_hwaccel));
        }

        loop {
            let mut decoded = Video::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    if pix_fmt_of(&decoded) != Pixel::VIDEOTOOLBOX {
                        // The decoder unexpectedly produced a software frame
                        // even though we asked for VT. Surface the same error
                        // the negotiation code raises so callers can react.
                        return Err(DecoderError::HwAccelNotActive(ActiveHwAccel::None));
                    }
                    let pts = decoded.pts();
                    let cv_ptr = unsafe { (*decoded.as_ptr()).data[3] } as *mut std::ffi::c_void;
                    let textures = hw_textures::extract_nv12_textures(cv_ptr, device)?;
                    return Ok(HwDecodeOutcome::Frame { textures, pts });
                }
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => {
                    // need more input — fall through to packet pump
                }
                Err(ffmpeg::Error::Eof) => {
                    self.drained = true;
                    return Ok(HwDecodeOutcome::Eof);
                }
                Err(err) => return Err(err.into()),
            }

            if self.drained {
                return Ok(HwDecodeOutcome::Eof);
            }

            self.pump_packet()?;
        }
    }

    fn scale_and_upload(
        &mut self,
        decoded: Video,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<(), DecoderError> {
        let from_hw = self.active_hwaccel == ActiveHwAccel::VideoToolbox
            && pix_fmt_of(&decoded) == Pixel::VIDEOTOOLBOX;

        if from_hw {
            // Reset destination so av_hwframe_transfer_data can derive a fresh
            // buffer from the hw_frames_ctx pool.
            self.transfer_frame = Video::empty();
            let err = unsafe {
                sys::av_hwframe_transfer_data(self.transfer_frame.as_mut_ptr(), decoded.as_ptr(), 0)
            };
            if err < 0 {
                return Err(DecoderError::HwTransfer(err));
            }
            unsafe {
                (*self.transfer_frame.as_mut_ptr()).pts = (*decoded.as_ptr()).pts;
            }
        }

        let (src_format, src_w, src_h) = if from_hw {
            (
                pix_fmt_of(&self.transfer_frame),
                self.transfer_frame.width(),
                self.transfer_frame.height(),
            )
        } else {
            (pix_fmt_of(&decoded), decoded.width(), decoded.height())
        };

        self.ensure_scaler(src_format, src_w, src_h)?;

        let Self {
            sws,
            transfer_frame,
            rgba_frame,
            ..
        } = self;
        let sws = sws.as_mut().expect("ensure_scaler initializes self.sws");
        let src_ref: &Video = if from_hw { transfer_frame } else { &decoded };
        sws.scaler.run(src_ref, rgba_frame)?;

        upload_rgba(
            queue,
            texture,
            &self.rgba_frame,
            self.dst_width,
            self.dst_height,
            &mut self.upload_scratch,
        );
        Ok(())
    }

    fn ensure_scaler(
        &mut self,
        src_format: Pixel,
        src_w: u32,
        src_h: u32,
    ) -> Result<(), DecoderError> {
        let needs_rebuild = match &self.sws {
            None => true,
            Some(s) => {
                s.src_format != src_format
                    || s.src_width != src_w
                    || s.src_height != src_h
                    || s.dst_width != self.dst_width
                    || s.dst_height != self.dst_height
            }
        };
        if !needs_rebuild {
            return Ok(());
        }
        let scaler = Scaler::get(
            src_format,
            src_w,
            src_h,
            Pixel::RGBA,
            self.dst_width,
            self.dst_height,
            Flags::BILINEAR,
        )?;
        self.sws = Some(SwsState {
            scaler,
            src_format,
            src_width: src_w,
            src_height: src_h,
            dst_width: self.dst_width,
            dst_height: self.dst_height,
        });
        // Force rgba_frame to be reallocated with the new dimensions.
        self.rgba_frame = Video::empty();
        Ok(())
    }

    fn pump_packet(&mut self) -> Result<(), DecoderError> {
        if self.flushed {
            return Ok(());
        }
        loop {
            let mut packet = ffmpeg::packet::Packet::empty();
            match packet.read(&mut self.input) {
                Ok(()) => {
                    if packet.stream() == self.stream_index {
                        self.decoder.send_packet(&packet)?;
                        return Ok(());
                    }
                    // Packet belongs to another stream; keep reading.
                }
                Err(ffmpeg::Error::Eof) => {
                    self.decoder.send_eof()?;
                    self.flushed = true;
                    return Ok(());
                }
                Err(err) => return Err(err.into()),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Hwaccel plumbing
// ---------------------------------------------------------------------------

/// RAII wrapper around an `AVBufferRef` that owns a hardware device context.
struct HwDeviceCtx {
    ptr: *mut sys::AVBufferRef,
}

// The AVBufferRef itself is internally refcounted and FFmpeg is fine with
// being created on one thread and freed on another.
unsafe impl Send for HwDeviceCtx {}
unsafe impl Sync for HwDeviceCtx {}

impl HwDeviceCtx {
    fn create(device_type: sys::AVHWDeviceType) -> Result<Self, DecoderError> {
        let mut ptr: *mut sys::AVBufferRef = ptr::null_mut();
        let err = unsafe {
            sys::av_hwdevice_ctx_create(&mut ptr, device_type, ptr::null(), ptr::null_mut(), 0)
        };
        if err < 0 || ptr.is_null() {
            return Err(DecoderError::HwDeviceInit(err));
        }
        Ok(Self { ptr })
    }
}

impl Drop for HwDeviceCtx {
    fn drop(&mut self) {
        unsafe {
            sys::av_buffer_unref(&mut self.ptr);
        }
    }
}

fn resolve_hwaccel(requested: HwAccel) -> HwAccel {
    match requested {
        HwAccel::Auto => {
            if cfg!(any(target_os = "macos", target_os = "ios")) {
                HwAccel::VideoToolbox
            } else {
                HwAccel::None
            }
        }
        other => other,
    }
}

fn setup_hwaccel(
    ctx: &mut ffmpeg::codec::context::Context,
    original_request: HwAccel,
    resolved: HwAccel,
) -> Result<(ActiveHwAccel, Option<HwDeviceCtx>), DecoderError> {
    let device_type = match resolved {
        HwAccel::None => return Ok((ActiveHwAccel::None, None)),
        HwAccel::Auto => unreachable!("Auto is resolved before this call"),
        HwAccel::VideoToolbox => sys::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
    };

    // `Context::from_parameters` only fills `codec_id`; the actual `*const AVCodec`
    // isn't bound until `avcodec_open2` runs inside `decoder().video()`. Look up
    // the decoder ourselves so we can probe its hwaccel configs *before* opening.
    let codec_id = ctx.id();
    let raw_codec_id = unsafe { (*ctx.as_mut_ptr()).codec_id };
    let codec_ptr = unsafe { sys::avcodec_find_decoder(raw_codec_id) };
    if codec_ptr.is_null() {
        return fallback_or_error(original_request, codec_id, "no decoder found for stream");
    }
    if !unsafe { codec_supports_hwaccel(codec_ptr, device_type) } {
        return fallback_or_error(
            original_request,
            codec_id,
            "codec does not advertise the requested hwaccel device",
        );
    }

    let hw = match HwDeviceCtx::create(device_type) {
        Ok(h) => h,
        Err(err) => match original_request {
            HwAccel::Auto => {
                eprintln!(
                    "decoder: hwaccel auto-fallback to software (av_hwdevice_ctx_create failed: {err})"
                );
                return Ok((ActiveHwAccel::None, None));
            }
            _ => return Err(err),
        },
    };

    unsafe {
        let raw = ctx.as_mut_ptr();
        (*raw).hw_device_ctx = sys::av_buffer_ref(hw.ptr);
        (*raw).get_format = Some(get_videotoolbox_format);
    }

    Ok((ActiveHwAccel::VideoToolbox, Some(hw)))
}

fn fallback_or_error(
    original_request: HwAccel,
    codec: ffmpeg::codec::Id,
    reason: &str,
) -> Result<(ActiveHwAccel, Option<HwDeviceCtx>), DecoderError> {
    match original_request {
        HwAccel::Auto => {
            eprintln!("decoder: hwaccel auto-fallback to software ({reason})");
            Ok((ActiveHwAccel::None, None))
        }
        _ => Err(DecoderError::HwAccelUnsupported {
            requested: original_request,
            codec,
        }),
    }
}

unsafe fn codec_supports_hwaccel(
    codec: *const sys::AVCodec,
    device_type: sys::AVHWDeviceType,
) -> bool {
    let mut i = 0;
    loop {
        let config = unsafe { sys::avcodec_get_hw_config(codec, i) };
        if config.is_null() {
            return false;
        }
        let methods = unsafe { (*config).methods };
        let dt = unsafe { (*config).device_type };
        if (methods & sys::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as c_int) != 0
            && dt == device_type
        {
            return true;
        }
        i += 1;
    }
}

unsafe extern "C" fn get_videotoolbox_format(
    _avctx: *mut sys::AVCodecContext,
    mut pix_fmts: *const sys::AVPixelFormat,
) -> sys::AVPixelFormat {
    unsafe {
        while *pix_fmts != sys::AVPixelFormat::AV_PIX_FMT_NONE {
            if *pix_fmts == sys::AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX {
                return sys::AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX;
            }
            pix_fmts = pix_fmts.add(1);
        }
        // No hwaccel format on offer — let FFmpeg pick whatever software
        // fallback it prefers (returning NONE would abort decoding).
        *pix_fmts
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ensure_ffmpeg_initialized() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if let Err(err) = ffmpeg::init() {
            eprintln!("ffmpeg::init failed: {err}");
        }
    });
}

fn validate_target(texture: &wgpu::Texture) -> Result<(), DecoderError> {
    let format = texture.format();
    match format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => Ok(()),
        other => Err(DecoderError::UnsupportedTextureFormat { format: other }),
    }
}

fn pix_fmt_of(frame: &Video) -> Pixel {
    let raw = unsafe { (*frame.as_ptr()).format };
    let av_fmt: sys::AVPixelFormat =
        unsafe { std::mem::transmute::<c_int, sys::AVPixelFormat>(raw) };
    Pixel::from(av_fmt)
}

fn upload_rgba(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    frame: &Video,
    width: u32,
    height: u32,
    scratch: &mut Vec<u8>,
) {
    let packed_bytes_per_row = width * 4;
    let frame_stride = frame.stride(0);
    let plane = frame.data(0);

    let (data, bytes_per_row) = if frame_stride == packed_bytes_per_row as usize {
        (plane, packed_bytes_per_row)
    } else {
        let needed = (packed_bytes_per_row as usize) * (height as usize);
        if scratch.len() != needed {
            scratch.resize(needed, 0);
        }
        for row in 0..height as usize {
            let src_start = row * frame_stride;
            let src_end = src_start + packed_bytes_per_row as usize;
            let dst_start = row * packed_bytes_per_row as usize;
            let dst_end = dst_start + packed_bytes_per_row as usize;
            scratch[dst_start..dst_end].copy_from_slice(&plane[src_start..src_end]);
        }
        (&scratch[..], packed_bytes_per_row)
    };

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Convenience: probe a video file's source dimensions without keeping a
/// decoder open.
pub fn probe_dimensions(path: impl AsRef<Path>) -> Result<(u32, u32), DecoderError> {
    ensure_ffmpeg_initialized();
    let input = input(&path.as_ref())?;
    let stream = input
        .streams()
        .best(Type::Video)
        .ok_or(DecoderError::NoVideoStream)?;
    let codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
    let decoder = codec_ctx.decoder().video()?;
    Ok((decoder.width(), decoder.height()))
}
