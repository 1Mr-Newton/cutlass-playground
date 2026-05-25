//! Glue: open a sample video, decode with VideoToolbox into NV12 IOSurfaces,
//! composite to RGB with our compositor, hand the texture to Slint.

use std::env;
use std::path::PathBuf;

use compositor::{Nv12Planes, VideoCompositor};
use decoder::{HwAccel, HwDecodeOutcome, VideoDecoder};
use slint::{ComponentHandle, GraphicsAPI, RenderingState};

slint::include_modules!();

const DEFAULT_VIDEO: &str = "assets/13232364_3840_2160_24fps.mp4";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let video_path = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(DEFAULT_VIDEO)
    });

    if !video_path.exists() {
        eprintln!(
            "video file not found: {}\nusage: cargo run -- /path/to/video.mp4",
            video_path.display()
        );
        std::process::exit(1);
    }

    slint::BackendSelector::new()
        .require_wgpu_28(slint::wgpu_28::WGPUConfiguration::default())
        .select()?;

    let app = AppWindow::new()?;
    let app_weak = app.as_weak();

    let mut state: Option<RuntimeState> = None;

    app.window().set_rendering_notifier({
        let video_path = video_path.clone();
        move |render_state, graphics_api| match render_state {
            RenderingState::RenderingSetup => {
                if let GraphicsAPI::WGPU28 { device, queue, .. } = graphics_api {
                    match RuntimeState::initialize(device, queue, &video_path) {
                        Ok(s) => {
                            if let Some(app) = app_weak.upgrade() {
                                app.set_status(slint::SharedString::from(format!(
                                    "{}  ·  {}x{}  ·  hwaccel = {:?}",
                                    video_path
                                        .file_name()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("?"),
                                    s.info.width,
                                    s.info.height,
                                    s.info.hwaccel
                                )));
                            }
                            state = Some(s);
                        }
                        Err(err) => {
                            eprintln!("failed to initialize decoder/compositor: {err}");
                            if let Some(app) = app_weak.upgrade() {
                                app.set_status(slint::SharedString::from(format!(
                                    "initialization failed: {err}"
                                )));
                            }
                        }
                    }
                }
            }
            RenderingState::BeforeRendering => {
                let (Some(state), Some(app)) = (state.as_mut(), app_weak.upgrade()) else {
                    return;
                };
                state.tick(&app);
            }
            RenderingState::RenderingTeardown => {
                drop(state.take());
            }
            _ => {}
        }
    })?;

    // 60 Hz redraw timer — independent of the actual video frame rate, but
    // ensures we keep pulling new frames out of the decoder.
    let app_weak = app.as_weak();
    let timer = slint::Timer::default();
    timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(16),
        move || {
            if let Some(app) = app_weak.upgrade() {
                app.window().request_redraw();
            }
        },
    );

    app.run()?;

    Ok(())
}

struct RuntimeState {
    device: slint::wgpu_28::wgpu::Device,
    decoder: VideoDecoder,
    compositor: VideoCompositor,
    info: decoder::VideoInfo,
    /// Last decoded frame. Hold it across calls so the IOSurface stays
    /// alive between decode and composite (and so we re-render the same
    /// frame if the decoder isn't ready with a new one yet).
    current_frame: Option<decoder::HwFrameTextures>,
    eof_reported: bool,
    frames_decoded: u64,
    composites: u64,
    last_log: std::time::Instant,
}

impl RuntimeState {
    fn initialize(
        device: &slint::wgpu_28::wgpu::Device,
        queue: &slint::wgpu_28::wgpu::Queue,
        path: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (src_w, src_h) = decoder::probe_dimensions(path)?;
        // dst size for `decode_next` only — `decode_next_hw` ignores it and
        // hands back textures at the IOSurface's native resolution.
        let decoder = VideoDecoder::open_with(path, src_w, src_h, HwAccel::Auto)?;
        let info = decoder.info();
        eprintln!(
            "app: opened {} — {}x{} @ {:?} fps, hwaccel = {:?}",
            path.display(),
            info.width,
            info.height,
            info.frame_rate,
            info.hwaccel
        );

        let compositor = VideoCompositor::new(device, queue);

        Ok(Self {
            device: device.clone(),
            decoder,
            compositor,
            info,
            current_frame: None,
            eof_reported: false,
            frames_decoded: 0,
            composites: 0,
            last_log: std::time::Instant::now(),
        })
    }

    fn tick(&mut self, app: &AppWindow) {
        if !self.eof_reported {
            match self.decoder.decode_next_hw(&self.device) {
                Ok(HwDecodeOutcome::Frame { textures, pts: _ }) => {
                    self.current_frame = Some(textures);
                    self.frames_decoded += 1;
                }
                Ok(HwDecodeOutcome::Eof) => {
                    eprintln!("app: end of stream");
                    self.eof_reported = true;
                }
                Err(err) => {
                    eprintln!("app: decode_next_hw error: {err}");
                    self.eof_reported = true;
                }
            }
        }

        let Some(frame) = self.current_frame.as_ref() else {
            return;
        };

        let texture = self.compositor.render(
            Nv12Planes {
                y: &frame.y_texture,
                cbcr: &frame.cbcr_texture,
            },
            frame.width,
            frame.height,
        );

        match slint::Image::try_from(texture) {
            Ok(image) => app.set_video_texture(image),
            Err(err) => eprintln!("app: failed to import wgpu texture: {err}"),
        }

        self.composites += 1;
        let now = std::time::Instant::now();
        if now.duration_since(self.last_log) >= std::time::Duration::from_secs(1) {
            eprintln!(
                "app: decoded {} frames, composited {} times in the last second",
                self.frames_decoded, self.composites
            );
            self.frames_decoded = 0;
            self.composites = 0;
            self.last_log = now;
        }
    }
}
