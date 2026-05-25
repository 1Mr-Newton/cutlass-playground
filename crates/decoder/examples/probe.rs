//! Smoke test for the decoder crate's hwaccel path.
//!
//! Opens a video file (path from argv, defaults to one of the bundled
//! assets), spins up a headless wgpu device + queue, allocates a small
//! Rgba8Unorm texture, and decodes a handful of frames into it. Prints the
//! active hwaccel backend and per-frame PTS so we can confirm the
//! VideoToolbox path actually fires.
//!
//! Run with: `cargo run -p decoder --example probe -- assets/<file>.mp4`

use std::env;
use std::future::Future;
use std::path::PathBuf;
use std::pin::pin;
use std::task::{Context, Poll, Waker};
use std::thread;

use decoder::{DecodeOutcome, HwAccel, VideoDecoder};
use slint::wgpu_28::wgpu;

const FRAMES_TO_DECODE: usize = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/15622413_3840_2160_60fps.mp4")
        });

    println!("opening {}", path.display());

    let (width, height) = decoder::probe_dimensions(&path)?;
    println!("source size: {width}x{height}");

    // Decode at half-size for the probe so we don't allocate 64 MB textures.
    let dst_width = (width / 2).max(64);
    let dst_height = (height / 2).max(64);

    let mut dec = VideoDecoder::open_with(&path, dst_width, dst_height, HwAccel::Auto)?;
    let info = dec.info();
    println!(
        "decoder ready: {}x{} @ {:?} fps, hwaccel = {:?}",
        info.width, info.height, info.frame_rate, info.hwaccel
    );

    let (device, queue) = init_wgpu()?;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("decoder.probe.target"),
        size: wgpu::Extent3d {
            width: dst_width,
            height: dst_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: decoder::TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    for i in 0..FRAMES_TO_DECODE {
        match dec.decode_next(&queue, &texture)? {
            DecodeOutcome::Frame { pts } => {
                println!("frame {i:>3}: pts = {pts:?}");
            }
            DecodeOutcome::Eof => {
                println!("eof after {i} frames");
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
        label: Some("decoder.probe.device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }))?;
    Ok((device, queue))
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
