/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <cstdint>

#include "util/types.h"

CCL_NAMESPACE_BEGIN

class ViewportFrameGenerationScheduler {
 public:
  struct Decision {
    bool evaluate = false;
    bool reset = false;
    bool present_generated = false;
  };

  Decision begin_real_frame(uint64_t revision);
  void evaluation_succeeded(uint64_t revision, const Decision &decision);
  void evaluation_failed();

  void request_reset();
  bool consume_generated();

  uint64_t last_evaluated_revision() const
  {
    return last_evaluated_revision_;
  }

  bool has_evaluated_revision() const
  {
    return have_evaluated_revision_;
  }

 private:
  uint64_t last_evaluated_revision_ = 0;
  bool have_evaluated_revision_ = false;
  bool reset_requested_ = true;
  bool have_previous_real_ = false;
  bool generated_pending_ = false;
};

CCL_NAMESPACE_END
