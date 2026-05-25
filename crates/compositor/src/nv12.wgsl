// Full-screen pass that samples an NV12 frame (Y plane + interleaved CbCr
// plane) coming from a VideoToolbox IOSurface and converts to linear sRGB.
//
// BT.709 with limited (video) range — the most common output format for
// modern HD/UHD video. The matrix is from ITU-R BT.709-6.

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// A single triangle that covers the entire viewport. UVs run 0..1 across
// the triangle's bounding box (= the screen-space quad we care about).
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    out.uv = vec2<f32>(x, y);
    out.clip_pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var cbcr_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

fn yuv709_video_to_rgb(y: f32, cb: f32, cr: f32) -> vec3<f32> {
    // BT.709 limited range: Y in [16/255, 235/255], CbCr in [16/255, 240/255]
    let y_norm  = (y  - 16.0 / 255.0) * (255.0 / 219.0);
    let cb_norm = (cb - 128.0 / 255.0) * (255.0 / 224.0);
    let cr_norm = (cr - 128.0 / 255.0) * (255.0 / 224.0);

    let r = y_norm + 1.5748 * cr_norm;
    let g = y_norm - 0.1873 * cb_norm - 0.4681 * cr_norm;
    let b = y_norm + 1.8556 * cb_norm;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let y = textureSample(y_tex, samp, in.uv).r;
    let cbcr = textureSample(cbcr_tex, samp, in.uv).rg;
    let rgb = yuv709_video_to_rgb(y, cbcr.x, cbcr.y);
    return vec4<f32>(rgb, 1.0);
}
