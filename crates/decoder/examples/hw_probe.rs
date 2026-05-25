//! Validates the zero-copy IOSurface → wgpu::Texture path end-to-end.
//!
//! Spins up a headless wgpu Metal device, opens a video, and decodes a
//! handful of frames via `decode_next_hw`. Prints, per frame, the IOSurface
//! pixel format, the resolved plane sizes and a tiny GPU readback from the
//! Y plane so we can sanity-check we're really sampling decoded pixels and
//! not (say) leftover heap garbage.
//!
//! Run: `cargo run -p decoder --example hw_probe -- assets/<file>.mp4`

use std::env;
use std::future::Future;
use std::path::PathBuf;
use std::pin::pin;
use std::task::{Context, Poll, Waker};
use std::thread;

use decoder::{HwAccel, HwDecodeOutcome, VideoDecoder};
use slint::wgpu_28::wgpu;

const FRAMES_TO_DECODE: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/15881269_3840_2160_60fps_720p_proxy.mp4")
    });

    println!("opening {}", path.display());

    let (src_w, src_h) = decoder::probe_dimensions(&path)?;
    println!("source size: {src_w}x{src_h}");

    let mut dec = VideoDecoder::open_with(&path, src_w, src_h, HwAccel::Auto)?;
    let info = dec.info();
    println!(
        "decoder ready: {}x{} @ {:?} fps, hwaccel = {:?}",
        info.width, info.height, info.frame_rate, info.hwaccel
    );

    let (device, queue) = init_wgpu()?;

    let mut frame_idx = 0;
    while frame_idx < FRAMES_TO_DECODE {
        match dec.decode_next_hw(&device)? {
            HwDecodeOutcome::Frame { textures, pts } => {
                let y_size = textures.y_texture.size();
                let cbcr_size = textures.cbcr_texture.size();
                println!(
                    "frame {frame_idx:>3}: pts = {pts:?} ; fmt = {:#010x} ; y = {}x{} ; cbcr = {}x{}",
                    textures.pixel_format,
                    y_size.width,
                    y_size.height,
                    cbcr_size.width,
                    cbcr_size.height,
                );

                // Read back a single Y row to confirm we can actually sample
                // from the IOSurface-backed wgpu texture.
                let sample = readback_y_row(&device, &queue, &textures.y_texture)?;
                let nonzero = sample.iter().filter(|b| **b != 0).count();
                println!(
                    "  └ y readback: {} non-zero samples / {}; first 8 = {:?}",
                    nonzero,
                    sample.len(),
                    &sample[..8.min(sample.len())]
                );

                frame_idx += 1;
            }
            HwDecodeOutcome::Eof => {
                println!("eof after {frame_idx} frames");
                break;
            }
        }
    }

    Ok(())
}

fn init_wgpu() -> Result<(wgpu::Device, wgpu::Queue), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))?;
    let info = adapter.get_info();
    println!(
        "wgpu adapter: {} ({:?} / {:?})",
        info.name, info.backend, info.device_type
    );
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("decoder.hwprobe.device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }))?;
    Ok((device, queue))
}

fn readback_y_row(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    y: &wgpu::Texture,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let size = y.size();
    // Round up bytes_per_row to wgpu's COPY_BYTES_PER_ROW_ALIGNMENT (256).
    const ALIGN: u32 = 256;
    let unpadded_row = size.width;
    let row_stride = unpadded_row.div_ceil(ALIGN) * ALIGN;
    let total = (row_stride * size.height) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hwprobe.staging"),
        size: total,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hwprobe.copy"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: y,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_stride),
                rows_per_image: Some(size.height),
            },
        },
        wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        tx.send(res).ok();
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;

    let data = slice.get_mapped_range();
    let row = data[..unpadded_row as usize].to_vec();
    drop(data);
    staging.unmap();
    Ok(row)
}

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => thread::yield_now(),
        }
    }
}
