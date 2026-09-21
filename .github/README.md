# Blender 5.2 Optimized Viewport

**Development notes, and a way to support the work if the add-on is useful to you:** 🤔😄🤗 [patreon.com/YGLabs](https://www.patreon.com/c/YGLabs).🤗

A fork of Blender 5.2.0 that denoises the Cycles viewport with NVIDIA DLSS Ray Reconstruction
over NGX, generates intermediate frames with DLSS Frame Generation, and carries the viewport
work those two turned out to need.

Branched from the upstream release commit `fbe6228777e7d9afefcd61a413844e790ae75db7`
("Release: Bump to 5.2.0 release"), so a comparison against that commit is exactly this work
and nothing else. Windows x64. The requirements below say what needs Vulkan and what does
not.

This is a personal experimental fork. It is not affiliated with the Blender Foundation or with
NVIDIA, and it is not proposed for inclusion upstream.

## What it adds

**Ray Reconstruction** builds the viewport image from one sample per frame rather than
filtering a converged one, so the picture stays usable while the camera moves. On Vulkan the
model runs on the display side, on images shared with the render device; there is also a CUDA
path that evaluates it on the render device itself. Cycles fills its guides from the passes it
already has: depth, diffuse and specular albedo, normal and roughness, and motion.
Volumes get a motion vector taken from where the ray entered the medium rather than from the
camera, which is what keeps fog from smearing.

**Frame Generation** inserts one generated frame between two real ones while the viewport moves.
Its guides are built in a compute shader from linear depth and screen motion, and the presentation
queue holds only the newest pair: when a newer evaluation arrives, it replaces a pair still
waiting rather than queuing behind it, so the picture follows the camera instead of trailing a
backlog.

**Voxelized volumes.** Path tracing a medium one sample at a time is what makes fog expensive to
look at while the camera moves, and the reconstruction has no guide for it. So the medium is
voxelized into a grid aligned with the camera frustum: screen-space tiles across, 64 slices in
depth, each cell holding the light scattered toward the camera and the transmittance through it.
A ray that enters the fog reads the grid instead of marching the volume, which turns a per-sample
cost into a lookup.

Lighting the grid is where the work went. Emitters outside the volume's hull reach it; shadows
are marched through an octree of the density rather than assumed; slices are weighted by how much
of the cell they actually cover; multiple scattering is carried by an analytic return term taken
from the medium's own albedo. The grid runs while the view moves and fades back to the honest
path trace once it settles, so what you set up a shot against is the real thing.

**Faster frame-to-frame updates.** This is what decides whether an animation is watchable in
a rendered viewport at all. Stepping a frame used to hand Cycles' device buffers most of the
scene again, even when a single object had moved. Now the geometry arrays are laid out so that
a changed object rewrites only its own slice, attribute tables keep their allocations while the
layout holds still, and the light tree, the attribute map, and the per-object offsets are not
uploaded at all when nothing touched them. What has to be read back out of Blender — vertex
arrays, triangles, corner normals — is copied in parallel, and the sync maps are marked with an
epoch stamp rather than rebuilt. Motion history settles when the scene comes to rest, rather
than between every pair of frames.

**A VSync preference** beside the GPU backend: on, off, or strict FIFO on Vulkan, applied to a
live swapchain without a restart.

## What it does not contain

No NVIDIA SDK sources, headers, or binaries, and no NGX runtimes. The runtime libraries
(`nvngx_dlssd.dll`, `nvngx_dlssg.dll`) are loaded at run time from the ones the NVIDIA driver
installs on the machine. NVIDIA's license does not permit redistributing them here, and a build
made from this tree must not ship them either.

## Requirements

| | |
|---|---|
| Platform | Windows x64 |
| GPU backend | Vulkan for Frame Generation and for running the model on images shared with the render device; Ray Reconstruction also has a CUDA path |
| Ray Reconstruction | an RTX GPU and a driver that installs the DLSS Ray Reconstruction feature |
| Frame Generation | an RTX 40 or RTX 50 card: the check matches those names specifically, so a later generation is not accepted until it is added |
| DLSS SDK | headers only, located through `DLSS_SDK_ROOT` |
| Toolchain | whatever upstream Blender 5.2 requires on Windows, plus a CUDA toolkit for the Cycles CUDA and OptiX devices |

Everything degrades rather than fails: with no runtime or no suitable GPU the DLSS features stay
switched off, and a failure inside NGX once a feature is running is reported and the frame is left
as the path tracer made it. The rest of the work is not conditional on DLSS — the volume grid is
on by default and the viewport changes apply either way — so a build from this tree does not
render identically to upstream even with every DLSS feature off.

## Building

Follow the upstream Blender build instructions for Windows, adding the configuration file this
fork provides and the path to the DLSS SDK:

```
cmake -S . -B ../build_dlss ^
  -C build_files/cmake/config/blender_dlss_universal.cmake ^
  -DDLSS_SDK_ROOT=C:/path/to/DLSS
cmake --build ../build_dlss --config Release --target install
```

The configuration turns on `WITH_DLSS` and `WITH_DLSS_FRAME_GENERATION` and builds Cycles with
the CUDA and OptiX devices. It compiles CUDA binaries for `sm_89` and `sm_120` (Ada and
Blackwell); change `CYCLES_CUDA_BINARIES_ARCH` for other cards, or the first render will fall
back to a slow JIT compile.

`build_files/cmake/Modules/FindDLSS.cmake` locates the SDK headers through `DLSS_SDK_ROOT`,
given either as a CMake variable or as an environment variable.

## Diagnostics

Much of this work was driven by measurement, and the instrumentation is still in the tree
behind environment variables. They are all off unless set, and the build is silent without them.
`CYCLES_DEBUG_VIEWPORT_PHASES=1` reports where a viewport frame's time goes;
`CYCLES_DEBUG_POSE_TIME=1` reports what one accumulation costs. The `CYCLES_FROXEL_*` variables
override the volume grid for measurement: `CYCLES_FROXEL_FIX` is a bitmask that turns its
individual lighting corrections off one at a time, which is how their contributions were
attributed, and the others override distance, fade, light samples, and the numerical parameters.

## License

Blender is GPL-2.0-or-later; files added under `intern/cycles` follow Cycles' Apache-2.0, and
the rest follow the license of the area they sit in. Each added source and configuration file
carries an SPDX header.

Publishing these sources is not by itself permission to distribute a compiled build together
with NVIDIA's runtimes. If you build from this tree and hand the result to someone else, read
NVIDIA's license for the SDK and runtimes first.

## Upstream Blender

Everything else in this tree is Blender, unchanged. Blender is the free and open source 3D
creation suite: modeling, rigging, animation, simulation, rendering, compositing, motion
tracking and video editing.

- [Main website](https://www.blender.org)
- [Reference manual](https://docs.blender.org/manual/en/latest/index.html)
- [User community](https://www.blender.org/community/)
- [Build instructions](https://developer.blender.org/docs/handbook/building_blender/)
- [Developer documentation](https://developer.blender.org/docs/)
- Blender's own README, kept as it was: [README.md](../README.md)

Blender as a whole is licensed under the GNU General Public License, Version 3. Individual files
may have a different but compatible license. See [blender.org/about/license](
https://www.blender.org/about/license/).

Cloning this repository may cause Git LFS errors, as it does with Blender's own GitHub mirror.
Use `GIT_LFS_SKIP_SMUDGE=1` for the initial clone.
