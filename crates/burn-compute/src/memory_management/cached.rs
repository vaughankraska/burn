use crate::{
    memory_id_type,
    storage::{ComputeStorage, StorageHandle, StorageUtilization},
};
use alloc::vec::Vec;
use hashbrown::HashMap;
use std::sync::Arc;

#[cfg(all(not(target_family = "wasm"), feature = "std"))]
use std::time;
#[cfg(all(target_family = "wasm", feature = "std"))]
use web_time as time;

use super::{MemoryBinding, MemoryHandle, MemoryManagement};

// The ChunkId allows to keep track of how many references there are to a specific chunk.
memory_id_type!(ChunkId, ChunkHandle, ChunkBinding);
// The SliceId allows to keep track of how many references there are to a specific slice.
memory_id_type!(SliceId, SliceHandle, SliceBinding);

/// A tensor memory handle, referring to either a chunk or a slice.
#[derive(Debug, Clone)]
pub enum SimpleHandle {
    /// A whole chunk of memory.
    Chunk(ChunkHandle),
    /// A slice of a chunk of memory.
    Slice(SliceHandle),
}

/// Binding of the [simple handle](SimpleHandle).
#[derive(Debug, Clone)]
pub enum SimpleBinding {
    /// Binding of the [chunk handle](ChunkHandle).
    Chunk(ChunkBinding),
    /// Binding of the [slice handle](SliceHandle)
    Slice(SliceBinding),
}

const DEFAULT_ROUNDING: usize = 512;

/// maximum size we can allocate before moving on to the big pool
const SMALL_POOL_THRESHOLD: usize = 1024;

/// How should we round memory we ask the memory_management
/// and the size being requested
#[derive(Debug)]
pub enum RoundingStrategy {
    Default,
}

impl RoundingStrategy {
    pub fn round_size(&self, size: usize) -> usize {
        match self {
            RoundingStrategy::Default => {
                let remainder = size % DEFAULT_ROUNDING;

                if remainder == 0 {
                    size
                } else {
                    size + (DEFAULT_ROUNDING - remainder)
                }
            }
        }
    }
}

// When should coalescing occur
#[derive(Debug)]
pub enum MergingStrategy {
    MergeBeforeAllocations,
}

// Region of continous memory containing one or many slices. IMPORTANT: slices are assumed to be
// ordered by their offset.
#[derive(new)]
struct Chunk {
    storage: StorageHandle,
    handle: ChunkHandle,
    slices: Vec<SliceId>,
}

impl Chunk {
    fn merge_free_slices(&self, slices_pool: &mut HashMap<SliceId, Slice>) {
        let mut merged_slices: Vec<SliceId> = Vec::new();

        let mut start_slice_idx: usize = 0;
        let mut end_slice_idx: usize = 0;
        //double pointer technique, iterate from a start index and keep incrementing end_slice_idx
        //if the slices are free, if not start index takes the value of the index after
        //end_slice_idx
        while start_slice_idx < self.slices.len() {
            while end_slice_idx + 1 < self.slices.len()
                && slices_pool
                    .get(&self.slices[end_slice_idx + 1])
                    .expect("memory slice not in memory pool")
                    .handle
                    .is_free()
            {
                end_slice_idx += 1;
            }

            // if multiple contiguous slice, merge them into one bigger slice
            if start_slice_idx != end_slice_idx {
                let handle_slice = SliceHandle::new();
                let start_slice: StorageHandle = slices_pool
                    .get(&self.slices[start_slice_idx])
                    .expect("memory slice not in memory pool")
                    .storage;
                let end_slice: StorageHandle = slices_pool
                    .get(&self.slices[start_slice_idx])
                    .expect("memory slice not in memory pool")
                    .storage;
                let merged_slice_offset: usize = start_slice.offset();
                let merged_slice_size: usize =
                    (end_slice.offset() - start_slice.offset()) + end_slice.size();
                let storage = StorageHandle {
                    id: self.storage.id.clone(),
                    utilization: StorageUtilization::Slice {
                        offset: merged_slice_offset,
                        size: merged_slice_size,
                    },
                };

                //create new merged slice
                slices_pool.insert(
                    *handle_slice.id(),
                    Slice::new(storage, handle_slice.clone(), self.handle.clone()),
                );

                //add new slice to merged pool
                merged_slices.push(*handle_slice.id());

                //cleanup of old non merged slice
                for slice_idx in start_slice_idx..=end_slice_idx {
                    slices_pool.remove(&self.slices[slice_idx]);
                }
            } else {
                merged_slices.push(self.slices[start_slice_idx]);
            }

            start_slice_idx = end_slice_idx + 1;
        }
        self.slices = merged_slices;
    }
}

#[derive(new)]
struct Slice {
    storage: StorageHandle,
    handle: SliceHandle,
    // It is important to keep the chunk handle inside the slice, since it increases the ref count
    // on the chunk id and make the `is_free` method return false until the slice is freed.
    //
    // TL;DR we can't only store the chunk id.
    chunk: ChunkHandle,
}

enum SliceSize {
    Small { size: usize },
    Large { size: usize },
}

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct CachedMemoryMangament<Storage> {
    large_chunk_pool: HashMap<ChunkId, Chunk>,
    small_chunk_pool: HashMap<ChunkId, Chunk>,
    slices: HashMap<SliceId, Slice>,
    rounding_strategy: RoundingStrategy,
    merge_strategy: MergingStrategy,
    storage: Storage,
}

impl<Storage> core::fmt::Debug for CachedMemoryMangament<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            alloc::format!(
                "SimpleMemoryManagement {:?} - {:?} - {:?}",
                self.rounding_strategy,
                self.merge_strategy,
                core::any::type_name::<Storage>(),
            )
            .as_str(),
        )
    }
}

impl SliceSize {
    fn new(size: usize) -> SliceSize {
        if size < SMALL_POOL_THRESHOLD {
            SliceSize::Small { size }
        } else {
            SliceSize::Large { size }
        }
    }

    fn size(&self) -> usize {
        match self {
            SliceSize::Small { size } | SliceSize::Large { size } => *size,
        }
    }
}

impl MemoryBinding for SimpleBinding {}

impl MemoryHandle<SimpleBinding> for SimpleHandle {
    fn can_mut(&self) -> bool {
        match &self {
            SimpleHandle::Chunk(id) => id.can_mut(),
            SimpleHandle::Slice(id) => id.can_mut(),
        }
    }

    fn binding(self) -> SimpleBinding {
        match self {
            Self::Chunk(handle) => SimpleBinding::Chunk(handle.binding()),
            Self::Slice(handle) => SimpleBinding::Slice(handle.binding()),
        }
    }
}

impl<Storage: ComputeStorage> MemoryManagement<Storage> for CachedMemoryMangament<Storage> {
    type Handle = SimpleHandle;
    type Binding = SimpleBinding;

    /// Returns the resource from the storage, for the specified handle.
    fn get(&mut self, binding: Self::Binding) -> Storage::Resource {
        let storage = match binding {
            // Could be worth having small Chunk and Big Chunk instead.
            // Will just try looking in both pools instead
            SimpleBinding::Chunk(chunk) => {
                &self
                    .large_chunk_pool
                    .get(chunk.id())
                    .or_else(|| self.small_chunk_pool.get(chunk.id()))
                    .expect("Storage found for the given execution buffer handle")
                    .storage
            }
            SimpleBinding::Slice(slice) => {
                &self
                    .slices
                    .get(slice.id())
                    .expect("Storage found for the given execution buffer handle")
                    .storage
            }
        };

        self.storage.get(storage)
    }

    /// Reserves memory of specified size using the reserve algorithm, and return
    /// a handle to the reserved memory.
    ///
    /// Also clean ups, removing unused slices, and chunks if permitted by deallocation strategy.
    fn reserve(&mut self, size: usize) -> Self::Handle {
        let handle = self.reserve_algorithm(size);

        // TODO : Figure out if this is the right thing to do
        //if self.dealloc_strategy.should_dealloc() {
        //    self.cleanup_chunks();
        //}
        handle
    }

    fn alloc(&mut self, size: usize) -> Self::Handle {
        self.create_chunk(size)
    }

    fn dealloc(&mut self, binding: Self::Binding) {
        match binding {
            SimpleBinding::Chunk(chunk) => {
                if let Some(chunk) = self.large_chunk_pool.remove(chunk.id()) {
                    self.storage.dealloc(chunk.storage.id);
                } else if let Some(chunk) = self.small_chunk_pool.remove(chunk.id()) {
                    self.storage.dealloc(chunk.storage.id);
                }
            }
            SimpleBinding::Slice(_) => panic!("Can't dealloc slice manually"),
        }
    }

    fn storage(&mut self) -> &mut Storage {
        &mut self.storage
    }
}

impl<Storage: ComputeStorage> CachedMemoryMangament<Storage> {
    /// Creates a new instance using the given storage, deallocation strategy and slice strategy.
    pub fn new(storage: Storage) -> Self {
        Self {
            large_chunk_pool: HashMap::new(),
            small_chunk_pool: HashMap::new(),
            slices: HashMap::new(),
            rounding_strategy: RoundingStrategy::Default,
            merge_strategy: MergingStrategy::MergeBeforeAllocations,
            storage,
        }
    }

    fn merge_free_slices(&mut self) {
        // This might be wrong, but I think this is fine
        for pool in [&self.small_chunk_pool, &self.large_chunk_pool] {
            for (chunk_id, chunk) in pool.iter_mut() {
                chunk.merge_free_slices(&mut self.slices)
            }
        }
    }

    // Best fit algorithm, will find the smallest slice that can contain
    // the slice_size, return None if it can't
    fn find_free_slice(&self, slice_size: &SliceSize) -> Option<&Slice> {
        let mut size_diff_current = usize::MAX;
        let mut best_fit_slice = None;
        let &pool = match slice_size {
            SliceSize::Small { .. } => &self.small_chunk_pool,
            SliceSize::Large { .. } => &self.large_chunk_pool,
        };

        // Iterate over all the chunks in the right pool,
        // over all the slices inside those chunks
        for chunk in pool.values() {
            for slice_id in &chunk.slices {
                let slice: &Slice = &self
                    .slices
                    .get(slice_id)
                    .expect("existing slice is not in slice pool");

                // If slice is being used, do not choose it
                if !slice.handle.is_free() {
                    continue;
                }

                let storage_size = slice.storage.size();

                // If we find slice of the correct size return slice
                if slice_size.size() == storage_size {
                    best_fit_slice = Some(slice)
                }

                // Find the smallest large enough slice that can hold enough
                // of the given size
                let size_diff = storage_size - slice_size.size();
                if size_diff < size_diff_current {
                    best_fit_slice = Some(slice);
                    size_diff_current = size_diff;
                }
            }
        }

        best_fit_slice
    }

    fn reserve_algorithm(&mut self, size: usize) -> SimpleHandle {
        match self.merge_strategy {
            MergingStrategy::MergeBeforeAllocations => {
                self.merge_free_slices();
            }
        }

        size = self.rounding_strategy.round_size(size);
        let slice_size = SliceSize::new(size);
        let free_slice = self.find_free_slice(&slice_size);

        //
        match chunk {
            Some(chunk) => {
                if size == chunk.storage.size() {
                    // If there is one of exactly the same size, it reuses it.
                    SimpleHandle::Chunk(chunk.handle.clone())
                } else {
                    // Otherwise creates a slice of the right size upon it, always starting at zero.
                    self.create_slice(size, chunk.handle.clone())
                }
            }
            // If no chunk available, creates one of exactly the right size.
            None => self.create_chunk(size),
        }
    }

    /// Finds the smallest of the free and large enough chunks to fit `size`
    /// Returns the chunk's id and size.
    //    fn find_free_chunk(&self, size: usize) -> Option<&Chunk> {
    //        let mut size_diff_current = usize::MAX;
    //        let mut current = None;
    //
    //        for chunk in self.chunks.values() {
    //            // If chunk is already used, we do not choose it
    //            if !chunk.handle.is_free() {
    //                continue;
    //            }
    //
    //            let storage_size = chunk.storage.size();
    //
    //            // If we find a chunk of exactly the right size, we stop searching altogether
    //            if size == storage_size {
    //                current = Some(chunk);
    //                break;
    //            }
    //
    //            // Finds the smallest of the large enough chunks that can accept a slice
    //            // of the given size
    //            if self.slice_strategy.can_use_chunk(storage_size, size) {
    //                let size_diff = storage_size - size;
    //
    //                if size_diff < size_diff_current {
    //                    current = Some(chunk);
    //                    size_diff_current = size_diff;
    //                }
    //            }
    //        }
    //
    //        current
    //    }

    /// Creates a slice of size `size` upon the given chunk.
    ///
    /// For now slices must start at zero, therefore there can be only one per chunk
    //    fn create_slice(&mut self, size: usize, handle_chunk: ChunkHandle) -> SimpleHandle {
    //        let chunk = self.chunks.get_mut(handle_chunk.id()).unwrap();
    //        let handle_slice = SliceHandle::new();
    //
    //        let storage = StorageHandle {
    //            id: chunk.storage.id.clone(),
    //            utilization: StorageUtilization::Slice { offset: 0, size },
    //        };
    //
    //        if chunk.slices.is_empty() {
    //            self.slices.insert(
    //                *handle_slice.id(),
    //                Slice::new(storage, handle_slice.clone(), handle_chunk.clone()),
    //            );
    //        } else {
    //            panic!("Can't have more than 1 slice yet.");
    //        }
    //
    //        chunk.slices.push(*handle_slice.id());
    //
    //        SimpleHandle::Slice(handle_slice)
    //    }

    /// Creates a chunk of given size by allocating on the storage.
    fn create_chunk(&mut self, size: usize) -> SimpleHandle {
        let storage = self.storage.alloc(size);
        let handle = ChunkHandle::new();

        self.chunks.insert(
            *handle.id(),
            Chunk::new(storage, handle.clone(), Vec::new()),
        );

        SimpleHandle::Chunk(handle)
    }

    /// Deallocates free chunks and remove them from chunks map.
    fn cleanup_chunks(&mut self) {
        let mut ids_to_remove = Vec::new();

        self.chunks.iter().for_each(|(chunk_id, chunk)| {
            if chunk.handle.is_free() {
                ids_to_remove.push(*chunk_id);
            }
        });

        ids_to_remove
            .iter()
            .map(|chunk_id| self.chunks.remove(chunk_id).unwrap())
            .for_each(|chunk| {
                self.storage.dealloc(chunk.storage.id);
            });
    }

    /// Removes free slices from slice map and corresponding chunks.
    fn cleanup_slices(&mut self) {
        let mut ids_to_remove = Vec::new();

        self.slices.iter().for_each(|(slice_id, slice)| {
            if slice.handle.is_free() {
                ids_to_remove.push(*slice_id);
            }
        });

        ids_to_remove
            .iter()
            .map(|slice_id| self.slices.remove(slice_id).unwrap())
            .for_each(|slice| {
                let chunk = self.chunks.get_mut(slice.chunk.id()).unwrap();
                chunk.slices.retain(|id| id != slice.handle.id());
            });
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        memory_management::{MemoryHandle, MemoryManagement, SliceStrategy},
        storage::BytesStorage,
    };

    use super::{DeallocStrategy, SimpleMemoryManagement};

    #[test]
    fn can_mut_with_single_tensor_reference() {
        let mut memory_management = SimpleMemoryManagement::new(
            BytesStorage::default(),
            DeallocStrategy::Never,
            SliceStrategy::Never,
        );

        let chunk_size = 4;
        let simple_handle = memory_management.create_chunk(chunk_size);

        let x = simple_handle.clone();
        core::mem::drop(simple_handle);

        assert!(x.can_mut());
    }

    #[test]
    fn two_tensor_references_remove_mutability() {
        let mut memory_management = SimpleMemoryManagement::new(
            BytesStorage::default(),
            DeallocStrategy::Never,
            SliceStrategy::Never,
        );

        let chunk_size = 4;
        let simple_handle = memory_management.create_chunk(chunk_size);

        let x = simple_handle.clone();

        assert!(!simple_handle.can_mut());
        assert!(!x.can_mut())
    }

    #[test]
    fn when_non_empty_chunk_exists_and_other_one_created_there_should_be_two() {
        let mut memory_management = SimpleMemoryManagement::new(
            BytesStorage::default(),
            DeallocStrategy::Never,
            SliceStrategy::Never,
        );
        let chunk_size = 4;
        let _chunk_handle = memory_management.reserve(chunk_size);
        let _new_handle = memory_management.reserve(chunk_size);

        assert_eq!(memory_management.chunks.len(), 2);
    }

    #[test]
    fn when_empty_chunk_is_cleaned_upexists_it_disappears() {
        let mut memory_management = SimpleMemoryManagement::new(
            BytesStorage::default(),
            DeallocStrategy::Never,
            SliceStrategy::Never,
        );
        let chunk_size = 4;
        let chunk_handle = memory_management.reserve(chunk_size);
        drop(chunk_handle);
        memory_management.cleanup_chunks();

        assert_eq!(memory_management.chunks.len(), 0);
    }

    #[test]
    fn never_dealloc_strategy_never_deallocs() {
        let mut never_dealloc = DeallocStrategy::Never;
        for _ in 0..20 {
            assert!(!never_dealloc.should_dealloc())
        }
    }

    #[test]
    fn period_tick_dealloc_strategy_should_dealloc_after_period() {
        let period = 3;
        let mut period_tick_dealloc = DeallocStrategy::new_period_tick(period);

        for _ in 0..3 {
            for _ in 0..period - 1 {
                assert!(!period_tick_dealloc.should_dealloc());
            }
            assert!(period_tick_dealloc.should_dealloc());
        }
    }

    #[test]
    fn slice_strategy_minimum_bytes() {
        let strategy = SliceStrategy::MinimumSize(100);

        assert!(strategy.can_use_chunk(200, 101));
        assert!(!strategy.can_use_chunk(200, 99));
    }

    #[test]
    fn slice_strategy_maximum_bytes() {
        let strategy = SliceStrategy::MaximumSize(100);

        assert!(strategy.can_use_chunk(200, 99));
        assert!(!strategy.can_use_chunk(200, 101));
    }

    #[test]
    fn slice_strategy_ratio() {
        let strategy = SliceStrategy::Ratio(0.9);

        assert!(strategy.can_use_chunk(200, 180));
        assert!(!strategy.can_use_chunk(200, 179));
    }

    #[test]
    fn test_handle_mutability() {
        let mut memory_management = SimpleMemoryManagement::new(
            BytesStorage::default(),
            DeallocStrategy::Never,
            SliceStrategy::Ratio(0.5),
        );
        let handle = memory_management.reserve(10);

        let other_ref = handle.clone();

        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    #[test]
    fn test_slice_mutability() {
        let mut memory_management = SimpleMemoryManagement::new(
            BytesStorage::default(),
            DeallocStrategy::Never,
            SliceStrategy::Ratio(0.5),
        );
        let chunk = memory_management.reserve(10);

        if let super::SimpleHandle::Slice(_) = chunk {
            panic!("Should be a chunk.")
        }

        drop(chunk);

        let slice = memory_management.reserve(8);

        if let super::SimpleHandle::Chunk(_) = &slice {
            panic!("Should be a slice.")
        }

        if let super::SimpleHandle::Slice(slice) = slice {
            let other_ref = slice.clone();

            assert!(
                !slice.can_mut(),
                "Slice can't be mut when multiple ref to the same handle."
            );
            drop(other_ref);
            assert!(
                slice.can_mut(),
                "Slice should be mut when only one ref to the same handle."
            );
            assert!(
                !slice.is_free(),
                "Slice can't be reallocated when one ref still exist."
            );
        }
    }
}
