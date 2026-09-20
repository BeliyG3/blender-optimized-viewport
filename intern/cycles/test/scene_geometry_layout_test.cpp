/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "testing/testing.h"

#include "scene/geometry_layout.h"

CCL_NAMESPACE_BEGIN

/* The guarded allocator reports every allocation to these hooks, which live in the Cycles util
 * library. This suite is built standalone, so it supplies no-ops rather than pulling that library
 * and its dependencies in. */
void util_guarded_mem_alloc(size_t /*n*/) {}
void util_guarded_mem_free(size_t /*n*/) {}

/* The device arrays hold every geometry in the scene end to end. `geometry_layout_matches` decides whether an
 * update left every geometry in the same place, which is what allows the allocation to be kept and
 * only the rewritten slices uploaded. A false positive here does not produce a visible glitch - it
 * produces stale triangles read against a fresh acceleration structure, wrong object ids, or a read
 * past the end of an array. So the cases below are all the ways the layout can look unchanged while
 * it is not. */

using Layout = DeviceGeometryLayout;
using Span = DeviceGeometrySpan;

namespace {

/* Stand-ins for geometry identity. `layout_matches` only ever compares the pointers, never
 * dereferences them, so distinct addresses are all the test needs. */
Geometry *fake_geometry(const int index)
{
  static char storage[8];
  return reinterpret_cast<Geometry *>(storage + index);
}

Layout mesh_layout(const std::initializer_list<size_t> triangle_counts)
{
  Layout layout;
  int index = 1;
  for (const size_t count : triangle_counts) {
    Span span = {fake_geometry(index), layout.triangles, count, 0, 0};
    layout.triangles += count;
    layout.spans.push_back(span);
    index++;
  }
  layout.valid = true;
  return layout;
}

const char *match_reason(const Layout &resident, const Layout &now)
{
  const char *reason = "unset";
  geometry_layout_matches(resident, now, &reason);
  return reason;
}

}  // namespace

TEST(GeometryLayout, UnchangedLayoutMatches)
{
  const Layout resident = mesh_layout({100, 200, 300});
  const Layout now = mesh_layout({100, 200, 300});

  const char *reason = "unset";
  EXPECT_TRUE(geometry_layout_matches(resident, now, &reason));
  EXPECT_STREQ(reason, "");
}

TEST(GeometryLayout, RejectsWhenNothingIsResidentYet)
{
  Layout resident;
  resident.valid = false;
  const Layout now = mesh_layout({100});

  EXPECT_FALSE(geometry_layout_matches(resident, now, nullptr));
  EXPECT_STREQ(match_reason(resident, now), "no resident layout yet");
}

TEST(GeometryLayout, RejectsGeometryAdded)
{
  const Layout resident = mesh_layout({100, 200});
  const Layout now = mesh_layout({100, 200, 50});

  EXPECT_FALSE(geometry_layout_matches(resident, now, nullptr));
  EXPECT_STREQ(match_reason(resident, now), "geometry count changed");
}

/* The dangerous one: the totals agree, but triangles moved from one geometry to another, so every
 * offset after the first of them shifted. */
TEST(GeometryLayout, RejectsTrianglesMovedBetweenGeometries)
{
  const Layout resident = mesh_layout({100, 200, 300});
  const Layout now = mesh_layout({110, 190, 300});

  EXPECT_EQ(resident.triangles, now.triangles);
  EXPECT_FALSE(geometry_layout_matches(resident, now, nullptr));
  EXPECT_STREQ(match_reason(resident, now), "geometry moved or resized");
}

/* A removal plus an addition of the same size. The new geometry is a different object, and
 * `erase_by_swap` can also reorder the list, so identity has to be part of the comparison - the
 * offsets and totals alone would look untouched. */
TEST(GeometryLayout, RejectsReplacedGeometryOfIdenticalSize)
{
  const Layout resident = mesh_layout({100, 200});

  Layout now = resident;
  now.spans[1].geometry = fake_geometry(7);

  EXPECT_EQ(resident.triangles, now.triangles);
  EXPECT_FALSE(geometry_layout_matches(resident, now, nullptr));
  EXPECT_STREQ(match_reason(resident, now), "geometry order changed");
}

TEST(GeometryLayout, RejectsReorderedGeometries)
{
  const Layout resident = mesh_layout({100, 100});

  Layout now = resident;
  std::swap(now.spans[0].geometry, now.spans[1].geometry);

  EXPECT_FALSE(geometry_layout_matches(resident, now, nullptr));
  EXPECT_STREQ(match_reason(resident, now), "geometry order changed");
}

TEST(GeometryLayout, RejectsChangedTotal)
{
  const Layout resident = mesh_layout({100});
  const Layout now = mesh_layout({101});

  EXPECT_FALSE(geometry_layout_matches(resident, now, nullptr));
  EXPECT_STREQ(match_reason(resident, now), "total size changed");
}

/* Hair occupies two independent arrays. The number of curves can change while the number of
 * segments does not, so comparing only one of them would let a moved curve array through. */
TEST(GeometryLayout, RejectsHairWithSameSegmentsButDifferentCurves)
{
  Layout resident;
  resident.spans.push_back({fake_geometry(1), 0, 10, 0, 40});
  resident.spans.push_back({fake_geometry(2), 10, 10, 40, 40});
  resident.curves = 20;
  resident.curve_segments = 80;
  resident.valid = true;

  /* Same segment layout, but the first geometry now holds one more curve and the second one less. */
  Layout now = resident;
  now.spans[0].count = 11;
  now.spans[1].offset = 11;
  now.spans[1].count = 9;

  EXPECT_EQ(resident.curves, now.curves);
  EXPECT_EQ(resident.curve_segments, now.curve_segments);
  EXPECT_FALSE(geometry_layout_matches(resident, now, nullptr));
  EXPECT_STREQ(match_reason(resident, now), "geometry moved or resized");
}

TEST(GeometryLayout, RejectsHairWithSameCurvesButDifferentSegments)
{
  Layout resident;
  resident.spans.push_back({fake_geometry(1), 0, 10, 0, 40});
  resident.spans.push_back({fake_geometry(2), 10, 10, 40, 40});
  resident.curves = 20;
  resident.curve_segments = 80;
  resident.valid = true;

  Layout now = resident;
  now.spans[0].aux_count = 41;
  now.spans[1].aux_offset = 41;
  now.spans[1].aux_count = 39;

  EXPECT_FALSE(geometry_layout_matches(resident, now, nullptr));
  EXPECT_STREQ(match_reason(resident, now), "geometry moved or resized");
}

TEST(GeometryLayout, EmptySceneMatchesEmptyScene)
{
  Layout resident;
  resident.valid = true;
  Layout now;
  now.valid = true;

  EXPECT_TRUE(geometry_layout_matches(resident, now, nullptr));
}

TEST(GeometryLayout, ZeroSizedGeometryStillCompared)
{
  const Layout resident = mesh_layout({0, 100});
  const Layout now = mesh_layout({100, 0});

  EXPECT_EQ(resident.triangles, now.triangles);
  EXPECT_FALSE(geometry_layout_matches(resident, now, nullptr));
  EXPECT_STREQ(match_reason(resident, now), "geometry moved or resized");
}

/* `clear()` is what the RAII guard calls when an update does not run to completion, and an
 * invalidated snapshot must never match. */
TEST(GeometryLayout, ClearedSnapshotNeverMatches)
{
  Layout resident = mesh_layout({100, 200});
  const Layout now = mesh_layout({100, 200});

  resident.clear();

  EXPECT_FALSE(geometry_layout_matches(resident, now, nullptr));
  EXPECT_STREQ(match_reason(resident, now), "no resident layout yet");
}

CCL_NAMESPACE_END
