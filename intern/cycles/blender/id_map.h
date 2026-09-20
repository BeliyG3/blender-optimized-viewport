/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <cstring>

#include "scene/geometry.h"
/* The sweep reads `used_stamp` off the node, so every type this map is instantiated with has to be
 * complete here - a forward declaration was enough only while liveness lived in a separate set. */
#include "scene/procedural.h"
#include "scene/scene.h"

#include "util/map.h"
#include "util/set.h"

namespace blender {
struct ID;
}

CCL_NAMESPACE_BEGIN

/* ID Map
 *
 * Utility class to map between Blender datablocks and Cycles data structures,
 * and keep track of recalc tags from the dependency graph. */

template<typename K, typename T, typename Flags = uint> class id_map {
 public:
  id_map(Scene *scene_) : scene(scene_) {}

  ~id_map()
  {
    set<T *> nodes;

    typename map<K, T *>::iterator jt;
    for (jt = b_map.begin(); jt != b_map.end(); jt++) {
      nodes.insert(jt->second);
    }

    scene->delete_nodes(nodes);
  }

  T *find(const K &key)
  {
    /* One search, not two. `b_map` is a tree, and this map holds an entry per instance - 33858 of
     * them on a scene driven by Geometry Nodes - so the second lookup walked another fifteen
     * comparisons for a value the first one had already reached. */
    const auto it = b_map.find(key);
    return (it != b_map.end()) ? it->second : nullptr;
  }

  void set_recalc(void *id_ptr)
  {
    b_recalc.insert(id_ptr);
  }

  bool check_recalc(const blender::ID *id)
  {
    return id && b_recalc.contains(id);
  }

  bool has_recalc()
  {
    return !(b_recalc.empty());
  }

  void pre_sync()
  {
    /* A new epoch invalidates every stamp at once, so nothing has to be cleared. The counter is
     * what `post_sync` compares against; `used_this_sync` is how it learns, without walking
     * anything, whether every entry was seen. */
    used_epoch++;
    used_this_sync = 0;

    if (used_epoch == 0) {
      /* Wrapped. Only reachable after four billion syncs, but a stamp left over from epoch 0 would
       * read as current, so they are cleared once and the epoch restarts past the initial value. */
      for (const auto &entry : b_map) {
        entry.second->used_stamp = 0;
      }
      used_epoch = 1;
    }
  }

  /* Nodes this sync created and destroyed. Not answerable from the `recalc` that `add_or_update`
   * returns: that is true both for a node that was just created and for one that merely needs
   * recomputing, so it cannot tell "recreated" from "changed".
   *
   * Worth counting because the cost is out of all proportion to the number. A single created
   * object makes the object manager reallocate every scene array and drops the attribute layout
   * reuse - that is, one recreated object costs exactly what the layout snapshot exists to avoid,
   * no matter that the scene has 33858 of them and only one moved. */
  int stats_created = 0;
  int stats_deleted = 0;

  void clear_stats()
  {
    stats_created = 0;
    stats_deleted = 0;
  }

  /* Add new data. */
  void add(const K &key, T *data)
  {
    assert(find(key) == nullptr);
    b_map[key] = data;
    used(data);
  }

  /* Update existing data. */
  bool update(T *data, const blender::ID *id)
  {
    return update(data, id, id);
  }
  bool update(T *data, const blender::ID *id, const blender::ID *parent)
  {
    bool recalc = (b_recalc.contains(id));
    if (parent && parent != id) {
      recalc = recalc || (b_recalc.contains(parent));
    }
    used(data);
    return recalc;
  }

  /* Combined add and update as needed. */
  bool add_or_update(T **r_data, const blender::ID *id)
  {
    return add_or_update(r_data, id, id, id);
  }
  bool add_or_update(T **r_data, const blender::ID *id, const K &key)
  {
    return add_or_update(r_data, id, id, key);
  }
  bool add_or_update(T **r_data, const blender::ID *id, const blender::ID *parent, const K &key)
  {
    T *data = find(key);
    bool recalc;

    if (!data) {
      /* Add data if it didn't exist yet. */
      data = scene->create_node<T>();
      add(key, data);
      recalc = true;
      stats_created++;
    }
    else {
      /* check if updated needed. */
      recalc = update(data, id, parent);
    }

    *r_data = data;
    return recalc;
  }

  /* Combined add or update for convenience. */

  bool is_used(const K &key)
  {
    T *data = find(key);
    return (data) ? data->used_stamp == used_epoch : false;
  }

  void used(T *data)
  {
    /* tag data as still in use */
    if (data->used_stamp != used_epoch) {
      data->used_stamp = used_epoch;
      used_this_sync++;
    }
  }

  void set_default(T *data)
  {
    b_map[nullptr] = data;
  }

  void post_sync(bool do_delete = true)
  {
    /* Nothing died, so there is nothing to sweep. `used_this_sync` counts entries marked for the
     * first time this epoch, and every mark goes through a node already in the map, so equality
     * with the map's size means every entry was seen. On a scene of 33858 instances this turns the
     * sweep - a walk of the whole tree - into a comparison, and playback never loses an entry.
     *
     * Compared with `!=` rather than `<` so that anything unexpected still falls through to the
     * full walk instead of silently skipping it. */
    if (do_delete && used_this_sync != b_map.size()) {
      /* Erases in place. Rebuilding the map meant inserting every surviving entry into a fresh tree
       * and then assigning it over the old one - two allocations and a full re-sort per frame, for
       * a map that usually loses nothing at all. */
      set<T *> nodes_to_delete;

      for (auto it = b_map.begin(); it != b_map.end();) {
        if (it->second->used_stamp != used_epoch) {
          flags.erase(it->second);
          nodes_to_delete.insert(it->second);
          it = b_map.erase(it);
          stats_deleted++;
        }
        else {
          ++it;
        }
      }

      if (!nodes_to_delete.empty()) {
        scene->delete_nodes(nodes_to_delete);
      }
    }

    b_recalc.clear();
  }

  const map<K, T *> &key_to_scene_data()
  {
    return b_map;
  }

  bool test_flag(T *data, Flags val)
  {
    typename map<T *, uint>::iterator it = flags.find(data);
    return it != flags.end() && (it->second & (1 << val)) != 0;
  }

  void set_flag(T *data, Flags val)
  {
    flags[data] |= (1 << val);
  }

  void clear_flag(T *data, Flags val)
  {
    typename map<T *, uint>::iterator it = flags.find(data);
    if (it != flags.end()) {
      it->second &= ~(1 << val);

      if (it->second == 0) {
        flags.erase(it);
      }
    }
  }

 protected:
  map<K, T *> b_map;
  /* Liveness, as an epoch rather than a container. This used to be a hash set of node pointers
   * rebuilt every sync: on a scene where Geometry Nodes turn 837 objects into 33858 instances that
   * is 33858 node allocations per frame, as many frees on the next, and a hash miss per entry when
   * the map is swept - 1.9 ms on a 13900K, and 2.5 ms of the measured tail on a 10600K. A stamp on
   * the node answers the same question by comparison and needs no memory at all. */
  uint32_t used_epoch = 0;
  size_t used_this_sync = 0;
  map<T *, uint> flags;
  set<const void *> b_recalc;
  Scene *scene;
};

/* Object Key
 *
 * To uniquely identify instances, we use the parent, object and persistent instance ID.
 * We also export separate object for a mesh and its particle hair. */

enum { OBJECT_PERSISTENT_ID_SIZE = 8 /* MAX_DUPLI_RECUR in Blender. */ };

struct ObjectKey {
  void *parent;
  int id[OBJECT_PERSISTENT_ID_SIZE];
  void *ob;
  bool use_particle_hair;

  ObjectKey(void *parent_,
            const int id_[OBJECT_PERSISTENT_ID_SIZE],
            void *ob_,
            bool use_particle_hair_)
      : parent(parent_), ob(ob_), use_particle_hair(use_particle_hair_)
  {
    if (id_) {
      memcpy(id, id_, sizeof(id));
    }
    else {
      memset(id, 0, sizeof(id));
    }
  }

  bool operator<(const ObjectKey &k) const
  {
    if (ob < k.ob) {
      return true;
    }
    if (ob == k.ob) {
      if (parent < k.parent) {
        return true;
      }
      if (parent == k.parent) {
        if (use_particle_hair < k.use_particle_hair) {
          return true;
        }
        if (use_particle_hair == k.use_particle_hair) {
          return memcmp(id, k.id, sizeof(id)) < 0;
        }
      }
    }

    return false;
  }
};

/* Geometry Key
 *
 * We export separate geometry for a mesh and its particle hair, so key needs to
 * distinguish between them. */

struct GeometryKey {
  void *id;
  Geometry::Type geometry_type;

  GeometryKey(void *id, Geometry::Type geometry_type) : id(id), geometry_type(geometry_type) {}

  bool operator<(const GeometryKey &k) const
  {
    if (id < k.id) {
      return true;
    }
    if (id == k.id) {
      if (geometry_type < k.geometry_type) {
        return true;
      }
    }

    return false;
  }
};

/* Particle System Key */

struct ParticleSystemKey {
  void *ob;
  int id[OBJECT_PERSISTENT_ID_SIZE];

  ParticleSystemKey(void *ob_, const int id_[OBJECT_PERSISTENT_ID_SIZE]) : ob(ob_)
  {
    if (id_) {
      memcpy(id, id_, sizeof(id));
    }
    else {
      memset(id, 0, sizeof(id));
    }
  }

  bool operator<(const ParticleSystemKey &k) const
  {
    /* first id is particle index, we don't compare that */
    if (ob < k.ob) {
      return true;
    }
    if (ob == k.ob) {
      return memcmp(id + 1, k.id + 1, sizeof(int) * (OBJECT_PERSISTENT_ID_SIZE - 1)) < 0;
    }

    return false;
  }
};

CCL_NAMESPACE_END
