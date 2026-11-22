use alloc::{sync::Arc, vec::Vec};

use crate::render_resource::SurfaceTexture;
use crate::renderer::render_context::{
    finish_commands, finish_commands_sequential, PendingCommandBuffer,
};

#[cfg(not(target_arch = "wasm32"))]
use bevy_tasks::{TaskPool, TaskPoolBuilder};
#[cfg(not(target_arch = "wasm32"))]
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
};

#[cfg(feature = "trace")]
use bevy_log::info_span;

/// Tracks a single frame's GPU submission, allowing a caller to wait for that specific frame's
/// [`wgpu::SubmissionIndex`] without blocking on newer frames.
#[derive(Clone)]
pub struct FrameSubmissionHandle {
    #[cfg(not(target_arch = "wasm32"))]
    state: Arc<(Mutex<Option<Option<wgpu::SubmissionIndex>>>, Condvar)>,
    #[cfg(target_arch = "wasm32")]
    index: Option<wgpu::SubmissionIndex>,
}

impl FrameSubmissionHandle {
    #[cfg(not(target_arch = "wasm32"))]
    fn pending() -> Self {
        Self {
            state: Arc::new((Mutex::new(None), Condvar::new())),
        }
    }

    pub(crate) fn resolved(index: Option<wgpu::SubmissionIndex>) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                state: Arc::new((Mutex::new(Some(index)), Condvar::new())),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self { index }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn complete(&self, index: Option<wgpu::SubmissionIndex>) {
        let (lock, condvar) = &*self.state;
        let mut guard = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = Some(index);
        condvar.notify_all();
    }

    /// Waits until this frame's command buffers have been submitted to the GPU queue and returns
    /// its [`wgpu::SubmissionIndex`], or `None` if no command buffers were submitted.
    pub fn wait(&self) -> Option<wgpu::SubmissionIndex> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (lock, condvar) = &*self.state;
            let mut guard = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            loop {
                if let Some(index) = guard.as_ref() {
                    return index.clone();
                }
                guard = condvar
                    .wait(guard)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.index.clone()
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct SubmissionSyncState {
    in_flight: bool,
    panicked: bool,
    latest_submission: Option<FrameSubmissionHandle>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct SubmissionSync {
    in_flight: AtomicBool,
    state: Mutex<SubmissionSyncState>,
    condvar: Condvar,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for SubmissionSync {
    fn default() -> Self {
        Self {
            in_flight: AtomicBool::new(false),
            state: Mutex::new(SubmissionSyncState {
                in_flight: false,
                panicked: false,
                latest_submission: None,
            }),
            condvar: Condvar::new(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl SubmissionSync {
    #[inline]
    pub(crate) fn wait(&self) {
        if !self.in_flight.load(Ordering::Acquire) {
            return;
        }
        #[cfg(feature = "trace")]
        let _span = info_span!("wait_for_render_submission").entered();
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while guard.in_flight {
            guard = self
                .condvar
                .wait(guard)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        if guard.panicked {
            panic!("render submission thread panicked");
        }
    }

    pub(crate) fn last_submission(&self) -> Option<FrameSubmissionHandle> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .latest_submission
            .clone()
    }

    fn record_sync_submission(&self, handle: FrameSubmissionHandle) {
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.latest_submission = Some(handle);
    }

    fn begin_async_submission(&self, handle: FrameSubmissionHandle) {
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while guard.in_flight {
            guard = self
                .condvar
                .wait(guard)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        if guard.panicked {
            panic!("render submission thread panicked");
        }
        guard.in_flight = true;
        guard.latest_submission = Some(handle);
        self.in_flight.store(true, Ordering::Release);
    }

    fn finish_async_submission(&self) {
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.in_flight = false;
        self.in_flight.store(false, Ordering::Release);
        self.condvar.notify_all();
    }

    fn mark_panicked(&self) {
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.panicked = true;
        guard.in_flight = false;
        // Keep the fast-path atomic flag set so later `wait()` calls enter the slow path
        // and propagate the panic instead of silently continuing after the worker crashed.
        self.in_flight.store(true, Ordering::Release);
        self.condvar.notify_all();
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct SubmissionJob {
    commands: Vec<PendingCommandBuffer>,
    presents: Vec<SurfaceTexture>,
    handle: FrameSubmissionHandle,
}

#[cfg(not(target_arch = "wasm32"))]
struct SubmissionWorker {
    sender: Option<mpsc::SyncSender<SubmissionJob>>,
    thread: Option<JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for SubmissionWorker {
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct ActiveJobPanicGuard<'a> {
    sync: &'a SubmissionSync,
    handle: Option<&'a FrameSubmissionHandle>,
    armed: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for ActiveJobPanicGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            if let Some(handle) = self.handle {
                handle.complete(None);
            }
            self.sync.mark_panicked();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct SubmissionCoordinator {
    sync: Arc<SubmissionSync>,
    worker: Mutex<Option<SubmissionWorker>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for SubmissionCoordinator {
    fn default() -> Self {
        Self {
            sync: Arc::new(SubmissionSync::default()),
            worker: Mutex::new(None),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl SubmissionCoordinator {
    pub(crate) fn sync(&self) -> Arc<SubmissionSync> {
        Arc::clone(&self.sync)
    }

    #[inline]
    pub(crate) fn wait(&self) {
        self.sync.wait();
    }

    pub(crate) fn last_submission(&self) -> Option<FrameSubmissionHandle> {
        self.sync.last_submission()
    }

    fn ensure_worker_sender(&self, raw_queue: &wgpu::Queue) -> mpsc::SyncSender<SubmissionJob> {
        let mut worker = self
            .worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(existing) = worker.as_ref().and_then(|w| w.sender.as_ref()) {
            return existing.clone();
        }
        let (sender, receiver) = mpsc::sync_channel::<SubmissionJob>(1);
        let thread_queue = raw_queue.clone();
        let thread_sync = Arc::clone(&self.sync);
        let join_handle = thread::Builder::new()
            .name("Render submission thread".into())
            .spawn(move || {
                #[cfg(feature = "trace")]
                let _span = info_span!("render_submission_thread").entered();
                // On 4 or fewer logical CPUs, finishing encoders directly on the submission thread
                // avoids adding extra worker wakeups that contend with main and render threads.
                let finish_pool = thread::available_parallelism()
                    .ok()
                    .filter(|count| count.get() > 4)
                    .map(|_| {
                        TaskPoolBuilder::new()
                            .num_threads(2)
                            .thread_name("Render finish".into())
                            .build()
                    });
                while let Ok(job) = receiver.recv() {
                    let mut panic_guard = ActiveJobPanicGuard {
                        sync: &thread_sync,
                        handle: Some(&job.handle),
                        armed: true,
                    };
                    let submission_index = execute_submission(
                        &thread_queue,
                        job.commands,
                        job.presents,
                        match finish_pool.as_ref() {
                            Some(pool) => FinishMode::DedicatedPool(pool),
                            None => FinishMode::Sequential,
                        },
                    );
                    job.handle.complete(submission_index);
                    panic_guard.armed = false;
                    thread_sync.finish_async_submission();
                }
            })
            .expect("Failed to create render submission thread");
        *worker = Some(SubmissionWorker {
            sender: Some(sender.clone()),
            thread: Some(join_handle),
        });
        sender
    }

    pub(crate) fn submit_frame(
        &self,
        raw_queue: &wgpu::Queue,
        commands: Vec<PendingCommandBuffer>,
        presents: Vec<SurfaceTexture>,
        synchronous: bool,
    ) -> FrameSubmissionHandle {
        if commands.is_empty() && presents.is_empty() {
            let handle = FrameSubmissionHandle::resolved(None);
            self.sync.wait();
            self.sync.record_sync_submission(handle.clone());
            return handle;
        }

        if synchronous {
            self.sync.wait();
            let submission_index =
                execute_submission(raw_queue, commands, presents, FinishMode::ComputePool);
            let handle = FrameSubmissionHandle::resolved(submission_index);
            self.sync.record_sync_submission(handle.clone());
            return handle;
        }

        let handle = FrameSubmissionHandle::pending();
        let sender = self.ensure_worker_sender(raw_queue);
        self.sync.begin_async_submission(handle.clone());
        if sender
            .send(SubmissionJob {
                commands,
                presents,
                handle: handle.clone(),
            })
            .is_err()
        {
            handle.complete(None);
            self.sync.mark_panicked();
            panic!("render submission thread exited unexpectedly");
        }
        handle
    }
}

#[cfg(not(target_arch = "wasm32"))]
enum FinishMode<'a> {
    Sequential,
    DedicatedPool(&'a TaskPool),
    ComputePool,
}

fn execute_submission(
    raw_queue: &wgpu::Queue,
    commands: Vec<PendingCommandBuffer>,
    presents: Vec<SurfaceTexture>,
    #[cfg(not(target_arch = "wasm32"))] finish_mode: FinishMode<'_>,
) -> Option<wgpu::SubmissionIndex> {
    let submission_index = if commands.is_empty() {
        None
    } else {
        #[cfg(feature = "trace")]
        let buffer_count = commands.len();
        #[cfg(not(target_arch = "wasm32"))]
        let buffers = match finish_mode {
            FinishMode::Sequential => finish_commands_sequential(commands),
            FinishMode::DedicatedPool(pool) => finish_commands(commands, Some(pool)),
            FinishMode::ComputePool => finish_commands(commands, None),
        };
        #[cfg(target_arch = "wasm32")]
        let buffers = finish_commands(commands);
        if buffers.is_empty() {
            None
        } else {
            #[cfg(feature = "trace")]
            let _span = info_span!("queue_submit", count = buffer_count).entered();
            Some(raw_queue.submit(buffers))
        }
    };

    if !presents.is_empty() {
        #[cfg(feature = "trace")]
        let _span = info_span!("present_frames").entered();
        for surface_texture in presents {
            surface_texture.present_on_raw_queue(raw_queue);
        }
    }

    submission_index
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
pub(crate) struct SubmissionCoordinator {
    last_submission: core::cell::RefCell<Option<FrameSubmissionHandle>>,
}

#[cfg(target_arch = "wasm32")]
impl SubmissionCoordinator {
    #[inline]
    pub(crate) fn wait(&self) {}

    pub(crate) fn last_submission(&self) -> Option<FrameSubmissionHandle> {
        self.last_submission.borrow().clone()
    }

    pub(crate) fn submit_frame(
        &self,
        raw_queue: &wgpu::Queue,
        commands: Vec<PendingCommandBuffer>,
        presents: Vec<SurfaceTexture>,
        _synchronous: bool,
    ) -> FrameSubmissionHandle {
        let submission_index = execute_submission(raw_queue, commands, presents);
        let handle = FrameSubmissionHandle::resolved(submission_index);
        *self.last_submission.borrow_mut() = Some(handle.clone());
        handle
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn handle_unblocks_waiter_when_completed() {
        let handle = FrameSubmissionHandle::pending();
        let worker_handle = handle.clone();
        let worker = thread::spawn(move || {
            worker_handle.complete(None);
        });
        assert!(handle.wait().is_none());
        worker.join().unwrap();
    }

    #[test]
    fn panic_guard_unblocks_sync_and_handle_without_deadlock() {
        let sync = Arc::new(SubmissionSync::default());
        let handle = FrameSubmissionHandle::pending();
        sync.begin_async_submission(handle.clone());

        let worker_sync = Arc::clone(&sync);
        let worker_handle = handle.clone();
        let worker = thread::spawn(move || {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                let _guard = ActiveJobPanicGuard {
                    sync: &worker_sync,
                    handle: Some(&worker_handle),
                    armed: true,
                };
                panic!("simulated submission failure");
            }));
        });

        assert!(handle.wait().is_none());
        let wait_result = catch_unwind(AssertUnwindSafe(|| sync.wait()));
        assert!(wait_result.is_err());
        worker.join().unwrap();
    }
}
