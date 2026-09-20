/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/geometry_layout.h"

CCL_NAMESPACE_BEGIN

bool geometry_layout_matches(const DeviceGeometryLayout &resident,
                             const DeviceGeometryLayout &now,
                             const char **reason)
{
  const auto reject = [reason](const char *why) {
    if (reason) {
      *reason = why;
    }
    return false;
  };

  if (!resident.valid) {
    return reject("no resident layout yet");
  }

  if (now.spans.size() != resident.spans.size()) {
    return reject("geometry count changed");
  }

  if (now.triangles != resident.triangles || now.curves != resident.curves ||
      now.curve_segments != resident.curve_segments || now.points != resident.points)
  {
    return reject("total size changed");
  }

  for (size_t i = 0; i < now.spans.size(); i++) {
    const DeviceGeometrySpan &a = now.spans[i];
    const DeviceGeometrySpan &b = resident.spans[i];

    /* Identity and order both matter. Without this, a geometry removed and another of the same size
     * added in its place would look untouched - a fresh geometry starts at offset zero - and
     * `erase_by_swap` would reorder the list without changing any total. */
    if (a.geometry != b.geometry) {
      return reject("geometry order changed");
    }

    /* Hair occupies two independent arrays: the number of curves can change while the number of
     * segments does not, so both slices are compared. */
    if (a.offset != b.offset || a.count != b.count || a.aux_offset != b.aux_offset ||
        a.aux_count != b.aux_count)
    {
      return reject("geometry moved or resized");
    }
  }

  if (reason) {
    *reason = "";
  }
  return true;
}

CCL_NAMESPACE_END
