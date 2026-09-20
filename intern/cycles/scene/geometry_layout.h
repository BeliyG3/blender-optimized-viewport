/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/vector.h"

CCL_NAMESPACE_BEGIN

class Geometry;

/* Where each geometry lives inside the scene-wide device arrays.
 *
 * Those arrays hold every geometry in the scene end to end, so changing one geometry marks the whole
 * array for reallocation and the entire scene is re-uploaded. On a scene whose geometry is driven by
 * modifiers - Blender reports it re-evaluated on every frame even when the result is identical - that
 * measured 180 ms per frame for the triangle arrays alone, at 40.5 M triangles.
 *
 * A snapshot of the layout that is actually resident on the device lets an update tell "the same
 * geometries in the same places, some of them rewritten" from "the layout moved". The former can keep
 * the allocation and patch ranges; only the latter needs a reallocation.
 *
 * Kept in its own header, free of scene and device dependencies, so the comparison can be unit
 * tested: a false positive here does not produce a visible glitch but stale triangles read against a
 * fresh acceleration structure, wrong object ids, or a read past the end of an array. */
struct DeviceGeometrySpan {
  /* Identity only - never dereferenced by the comparison. */
  Geometry *geometry;
  /* Primary array slice: triangles for a mesh, curves for hair, points for a point cloud. */
  size_t offset;
  size_t count;
  /* Secondary slice, only used by hair for its curve segments. */
  size_t aux_offset;
  size_t aux_count;
};

struct DeviceGeometryLayout {
  vector<DeviceGeometrySpan> spans;
  size_t triangles = 0;
  size_t curves = 0;
  size_t curve_segments = 0;
  size_t points = 0;
  /* False until an update has completed and recorded what the device holds. */
  bool valid = false;

  void clear()
  {
    spans.clear();
    triangles = curves = curve_segments = points = 0;
    valid = false;
  }
};

/* Whether `now` puts every geometry in the same place as `resident`.
 *
 * A matching layout is necessary but not sufficient: the caller must also check that the arrays are
 * resident at that size and that nothing else already required a reallocation.
 *
 * `reason`, when given, receives a short explanation of a rejection, for diagnostics. */
bool geometry_layout_matches(const DeviceGeometryLayout &resident,
                             const DeviceGeometryLayout &now,
                             const char **reason);

CCL_NAMESPACE_END
