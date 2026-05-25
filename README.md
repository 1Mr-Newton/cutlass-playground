# Cutlass Playground

Experimental workspace for **Cutlass** — an open-source, AI-first, prompt-to-edit video editor. This repo is where the core engine pieces are built and tested before they land in the full editor.

## What this is

Cutlass aims to be a CapCut-inspired editor where users describe edits in natural language and an AI agent turns that into structured, executable timeline commands. The playground focuses on the Rust engine underneath: decoding video, compositing on the GPU, and displaying frames in a Slint UI.

## Stack

- **Slint** — UI, layout, and declarative UX
- **WGPU** — GPU preview and compositing
- **FFmpeg** — video decode (with Apple VideoToolbox hardware acceleration on macOS)

## Crates

| Crate | Purpose |
|-------|---------|
| `app` | Slint app that decodes a video, composites NV12 → RGB, and displays the result |
| `decoder` | FFmpeg-backed decoder with optional zero-copy hwaccel (IOSurface → Metal → wgpu) |
| `compositor` | wgpu compositors, including an NV12 → RGB full-screen pass |

## Prerequisites

- Rust (2024 edition)
- FFmpeg development libraries (`ffmpeg-next` is linked at build time)
- macOS for the zero-copy VideoToolbox decode path (software decode works elsewhere)

## Running

Place a test video at `assets/13232364_3840_2160_24fps.mp4`, or pass a path explicitly:

```bash
cargo run -p app -- /path/to/video.mp4
```

The window shows the decoded frame and status text (resolution, hwaccel backend).

## Decoder examples

Probe decode behavior without the full UI:

```bash
cargo run -p decoder --example probe -- /path/to/video.mp4
cargo run -p decoder --example hw_probe -- /path/to/video.mp4   # macOS hwaccel
```

## Status

This is early infrastructure — a working decode → composite → display pipeline, not a full editor yet. Timeline editing, project state, and the AI agent layer come next.
