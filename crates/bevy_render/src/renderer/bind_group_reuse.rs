use alloc::vec::Vec;
use core::mem;
use std::sync::Mutex;

use bevy_platform::collections::HashMap;
use wgpu::{BindGroupEntry, BindingResource, BufferAddress, BufferSize};

use crate::render_resource::BindGroup;

/// Bind groups made in the last three frames, so that a bind group asked for again with the same
/// layout and resources is the one made before rather than a new one. Batching fills some of its
/// buffers in turns, so many bind groups are asked for again only every other frame.
///
/// Keys hold the resources they name, so no resource they name can be freed and its address
/// taken by another while an entry stands.
#[derive(Default)]
pub(super) struct BindGroupsMadeLately(Mutex<BindGroupGenerations>);

#[derive(Default)]
struct BindGroupGenerations {
    this_frame: HashMap<BindGroupKey, BindGroup>,
    previous_frame: HashMap<BindGroupKey, BindGroup>,
    frame_before: HashMap<BindGroupKey, BindGroup>,
}

#[derive(PartialEq, Eq, Hash)]
pub(super) struct BindGroupKey {
    layout: wgpu::BindGroupLayout,
    entries: Vec<(u32, ResourceKey)>,
}

#[derive(PartialEq, Eq, Hash)]
enum ResourceKey {
    Buffer(BufferKey),
    BufferArray(Vec<BufferKey>),
    Sampler(wgpu::Sampler),
    SamplerArray(Vec<wgpu::Sampler>),
    TextureView(wgpu::TextureView),
    TextureViewArray(Vec<wgpu::TextureView>),
}

#[derive(PartialEq, Eq, Hash)]
struct BufferKey {
    buffer: wgpu::Buffer,
    offset: BufferAddress,
    size: Option<BufferSize>,
}

impl BufferKey {
    fn of(binding: &wgpu::BufferBinding<'_>) -> Self {
        Self {
            buffer: binding.buffer.clone(),
            offset: binding.offset,
            size: binding.size,
        }
    }
}

impl ResourceKey {
    fn of(resource: &BindingResource<'_>) -> Option<Self> {
        Some(match resource {
            BindingResource::Buffer(binding) => Self::Buffer(BufferKey::of(binding)),
            BindingResource::BufferArray(bindings) => {
                Self::BufferArray(bindings.iter().map(BufferKey::of).collect())
            }
            BindingResource::Sampler(sampler) => Self::Sampler((*sampler).clone()),
            BindingResource::SamplerArray(samplers) => {
                Self::SamplerArray(samplers.iter().map(|sampler| (*sampler).clone()).collect())
            }
            BindingResource::TextureView(view) => Self::TextureView((*view).clone()),
            BindingResource::TextureViewArray(views) => {
                Self::TextureViewArray(views.iter().map(|view| (*view).clone()).collect())
            }
            _ => return None,
        })
    }
}

impl BindGroupKey {
    /// No key names a bind group binding resources that are not kept, such as acceleration
    /// structures.
    pub(super) fn of(
        layout: &wgpu::BindGroupLayout,
        entries: &[BindGroupEntry<'_>],
    ) -> Option<Self> {
        let entries = entries
            .iter()
            .map(|entry| Some((entry.binding, ResourceKey::of(&entry.resource)?)))
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            layout: layout.clone(),
            entries,
        })
    }
}

impl BindGroupsMadeLately {
    pub(super) fn take(&self, key: &BindGroupKey) -> Option<BindGroup> {
        let mut generations = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(bind_group) = generations.this_frame.get(key) {
            return Some(bind_group.clone());
        }
        let (kept_key, bind_group) = generations
            .previous_frame
            .remove_entry(key)
            .or_else(|| generations.frame_before.remove_entry(key))?;
        generations.this_frame.insert(kept_key, bind_group.clone());
        Some(bind_group)
    }

    pub(super) fn keep(&self, key: BindGroupKey, bind_group: &BindGroup) {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .this_frame
            .insert(key, bind_group.clone());
    }

    pub(super) fn end_frame(&self) {
        let mut generations = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let this_frame = mem::take(&mut generations.this_frame);
        generations.frame_before = mem::replace(&mut generations.previous_frame, this_frame);
    }
}
