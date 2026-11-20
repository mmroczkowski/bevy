use core::mem;

use bevy_ecs::resource::Resource;
use bevy_platform::collections::HashMap;
use bevy_render::render_resource::{BindGroup, BufferId};

/// The buffers and metadata offset a bin unpacking or uniform allocation bind group binds.
///
/// Buffer ids are never handed out twice, so equal inputs name the same buffers.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SceneUnpackingBindGroupInputs {
    BinUnpacking {
        metadata: BufferId,
        metadata_offset: u64,
        binned_mesh_instances: BufferId,
        work_items: BufferId,
        bin_metadata: BufferId,
        bin_index_to_bin_metadata_index: BufferId,
    },
    UniformAllocation {
        metadata: BufferId,
        metadata_offset: u64,
        bin_metadata: BufferId,
        indirect_parameters_metadata: BufferId,
        fan: BufferId,
    },
}

/// The bin unpacking and uniform allocation bind groups made in the last three frames, so that a
/// batch set whose buffers stayed the same keeps its bind groups rather than asking the render
/// device for them, one per batch set, view and phase every frame. Work item buffers are filled in
/// turns of two frames, so a bind group is often asked for again only every other frame.
#[derive(Resource, Default)]
pub struct SceneUnpackingBindGroupsMadeLately {
    this_frame: HashMap<SceneUnpackingBindGroupInputs, BindGroup>,
    previous_frame: HashMap<SceneUnpackingBindGroupInputs, BindGroup>,
    frame_before: HashMap<SceneUnpackingBindGroupInputs, BindGroup>,
}

impl SceneUnpackingBindGroupsMadeLately {
    pub(crate) fn begin_frame(&mut self) {
        let previous_frame = mem::take(&mut self.this_frame);
        self.frame_before = mem::replace(&mut self.previous_frame, previous_frame);
    }

    pub(crate) fn get_or_make(
        &mut self,
        inputs: SceneUnpackingBindGroupInputs,
        make: impl FnOnce() -> BindGroup,
    ) -> BindGroup {
        if let Some(bind_group) = self.this_frame.get(&inputs) {
            return bind_group.clone();
        }
        let bind_group = self
            .previous_frame
            .remove(&inputs)
            .or_else(|| self.frame_before.remove(&inputs))
            .unwrap_or_else(make);
        self.this_frame.insert(inputs, bind_group.clone());
        bind_group
    }
}
