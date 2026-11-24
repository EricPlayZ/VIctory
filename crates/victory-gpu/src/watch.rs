//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// Whether every watch page of `[start, end)` is in one committed private
/// region. A failed query for a subrange then cannot have crossed a region
/// boundary or encountered an unmapped page elsewhere in the shadow.
pub(crate) fn private_region_covers(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let start = start / PAGE * PAGE;
    let Some(end) = end.checked_next_multiple_of(PAGE) else {
        return false;
    };
    // SAFETY: a query of the virtual address, without reading its memory.
    unsafe {
        let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        VirtualQuery(start as *const _, &mut info, size_of_val(&info)) != 0
            && info.State == MEM_COMMIT
            && info.Type == MEM_PRIVATE
            && (info.BaseAddress as usize)
                .checked_add(info.RegionSize)
                .is_some_and(|limit| limit >= end)
    }
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Pages found reserved but not committed (no data; placeholders the
    /// title maps physical pages into later): reported written once they
    /// hold memory.
    holes: BTreeSet<usize>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    holes: BTreeSet::new(),
});

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    // Not one write-watched allocation: a view is guarded whole, as before.
    // Otherwise the range is asked about page by page (allocations start on
    // 64 KiB): the title's buffers span many of its small allocations, some
    // reserved but not mapped yet, and copying such ranges whole was
    // ~4-5 GB a world frame, mostly of unmapped holes.
    let mut by_pages = Vec::new();
    let guarded = if !failed {
        Vec::new()
    } else if let Some(pages) = guarded.then(|| guard(first * PAGE, last * PAGE)).flatten() {
        count = 0;
        pages
    } else {
        count = 0;
        match by_page(j, first, last, guarded) {
            Ok(pages) => by_pages = pages,
            Err(pages) => {
                // Pages asked about before one that cannot be: their writes
                // are now cleared in the kernel, so the journal keeps them.
                if !pages.is_empty() {
                    j.asked += 1;
                    for page in pages {
                        j.written.insert(page, j.asked);
                    }
                }
                return None;
            }
        }
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    let pages = j.found[..count]
        .iter()
        .map(|&at| at / PAGE)
        .chain(guarded)
        .chain(by_pages);
    for page in pages.collect::<Vec<_>>() {
        j.written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// The written pages of pages `[first, last)`, asked about one at a time
/// (clearing what the kernel reports): a write-watched page as the kernel
/// says, a view's as its guard says, a reserved page with no memory as
/// unwritten (noted as a hole, written once it holds memory). `Err` with
/// the written pages found before a page none of these covers.
fn by_page(
    j: &mut Journal,
    first: usize,
    last: usize,
    guarded: bool,
) -> Result<Vec<usize>, Vec<usize>> {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_MAPPED, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut written = Vec::new();
    let mut answer = [0usize; PAGE / KERNEL_PAGE];
    for page in first..last {
        let mut count = answer.len();
        let mut granularity = 0;
        // SAFETY: the answer buffer holds `count` entries.
        let watched = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (page * PAGE) as *const _,
                PAGE,
                answer.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } == 0;
        if watched {
            if count > 0 || j.holes.remove(&page) {
                written.push(page);
            }
            continue;
        }
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return Err(written);
        }
        if info.State == MEM_RESERVE {
            j.holes.insert(page);
            continue;
        }
        let filled = j.holes.remove(&page);
        if guarded
            && info.Type == MEM_MAPPED
            && let Some(pages) = guard(page * PAGE, (page + 1) * PAGE)
        {
            if filled || !pages.is_empty() {
                written.push(page);
            }
            continue;
        }
        if filled {
            written.push(page);
        }
        return Err(written);
    }
    Ok(written)
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.Type != MEM_MAPPED {
        return None;
    }
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    for block in first..=last {
        let open = block < last && !g.clean.contains(&block);
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail (unmapped).
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        }
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        let whole = unsafe {
            VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old)
        } != 0;
        if !whole {
            // The block now spans views (the title maps at 4 KiB): one call
            // cannot cross them, so each view's part is opened on its own.
            open_views(block * BLOCK, (block + 1) * BLOCK);
        }
    }
    !blocks.is_empty()
}

/// Gives write access back to `[from, to)` view by view: the read-only
/// mapped regions in it, each with a call of its own.
fn open_views(from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualProtect,
        VirtualQuery,
    };
    let mut at = from;
    while at < to {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
            return;
        }
        let end = (info.BaseAddress as usize + info.RegionSize).min(to).max(at + 1);
        if info.Type == MEM_MAPPED && info.Protect == PAGE_READONLY {
            let mut old = 0;
            // SAFETY: title view memory inside one region; only its
            // protection changes.
            unsafe { VirtualProtect(at as *const _, end - at, PAGE_READWRITE, &mut old) };
        }
        at = end;
    }
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    open(&mut guards(), at, at + len);
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded.
pub fn title_memory_remapped(at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let mut g = guards();
    let end = at + len;
    let blocks: Vec<usize> = g
        .protected
        .range(at / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for block in blocks {
        // A remap covering only part of a guarded block (pages are 4 KiB,
        // blocks 64 KiB) leaves the rest of the block read-only under its old
        // mapping: give that part write access back before forgetting the
        // block, or the title's next write there faults with no guard to
        // take it.
        let (from, to) = (block * BLOCK, (block + 1) * BLOCK);
        let mut old = 0;
        if from < at {
            // SAFETY: title view memory we made read-only; only its
            // protection changes (an unmapped part just fails).
            unsafe { VirtualProtect(from as *const _, at - from, PAGE_READWRITE, &mut old) };
        }
        if end < to {
            // SAFETY: as above.
            unsafe { VirtualProtect(end as *const _, to - end, PAGE_READWRITE, &mut old) };
        }
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
    drop(g);
    // New contents: written, for every refresh that copied the pages before
    // (taken after the guards: `writes` takes the journal first).
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.asked += 1;
    let now = j.asked;
    for page in at / PAGE..end.div_ceil(PAGE) {
        j.written.insert(page, now);
        j.holes.remove(&page);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // Read-only view memory no guard claims: a guard that lost track
        // (blocks are 64 KiB, the title's views 4 KiB-aligned). The console
        // never write-protects title views, so the block is opened and noted
        // written (not clean), and the copies taken from it are made again.
        if readonly_view(at) {
            #[cfg(feature = "diagnostics")]
            unguarded_write(at);
            // `g` is still held (since the checks above): taking it again
            // would deadlock the faulting thread.
            let block = at / BLOCK;
            g.protected.remove(&block);
            g.clean.remove(&block);
            open_views(block * BLOCK, (block + 1) * BLOCK);
            if writable(at) {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Diagnostics: a write fault the guards did not take, at mapped (view)
/// memory that is read-only: logged with its protection, as a guard lost
/// track of would show.
#[cfg(feature = "diagnostics")]
fn unguarded_write(at: usize) {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if n != 0 {
        eprintln!(
            "VIctory: write fault not taken by the guards at {at:#x}: state {:#x} protect {:#x} type {:#x}, region {:#x}+{:#x}",
            info.State, info.Protect, info.Type, info.BaseAddress as usize, info.RegionSize
        );
    }
}

/// Whether `at` is committed, read-only mapped (view) memory.
fn readonly_view(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Type == MEM_MAPPED && info.Protect == PAGE_READONLY
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
        VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        // The journal outlives allocations: pages an earlier test wrote at
        // this address may be listed, with older askings than this test's.
        let (first, _) = writes(start, end, false).expect("watched");
        let since = |pages: Vec<(usize, u64)>| -> Vec<(usize, u64)> {
            pages.into_iter().filter(|&(_, w)| w > first).collect()
        };
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(since(after), [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(since(again), [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    pub(crate) struct Reservation(pub(crate) *mut std::ffi::c_void);

    impl Drop for Reservation {
        fn drop(&mut self) {
            // SAFETY: this is the base of a reservation owned by the test.
            assert_ne!(unsafe { VirtualFree(self.0, 0, MEM_RELEASE) }, 0);
        }
    }

    pub(crate) fn adjacent(watched: usize, unwatched: usize) -> (Reservation, Reservation) {
        for _ in 0..16 {
            // SAFETY: these independent test reservations use valid allocation flags.
            unsafe {
                let span = VirtualAlloc(
                    std::ptr::null(),
                    watched + unwatched,
                    MEM_RESERVE,
                    PAGE_NOACCESS,
                );
                assert!(!span.is_null());
                assert_ne!(VirtualFree(span, 0, MEM_RELEASE), 0);
                let first = VirtualAlloc(
                    span,
                    watched,
                    MEM_COMMIT | MEM_RESERVE | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                );
                if first.is_null() {
                    continue;
                }
                let first = Reservation(first);
                let second = VirtualAlloc(
                    span.cast::<u8>().wrapping_add(watched).cast(),
                    unwatched,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_READWRITE,
                );
                if !second.is_null() {
                    return (first, Reservation(second));
                }
            }
        }
        panic!("adjacent test reservations");
    }

    #[test]
    fn unwatched_subrange_does_not_disable_watch_elsewhere_in_shadow() {
        if crate::host::host().is_none() {
            return;
        }
        let watched = 16 << 20;
        let (first, second) = adjacent(watched, PAGE);
        let start = first.0 as usize;
        let len = (watched + PAGE) as u64;
        struct RegisteredBuffer(usize);
        impl Drop for RegisteredBuffer {
            fn drop(&mut self) {
                crate::shadow::unregister(self.0, self.0 as u64);
            }
        }
        crate::shadow::register(start, start as u64, len);
        let _registration = RegisteredBuffer(start);
        // SAFETY: both reservations are live. Populate demand-zero pages
        // before the first refresh, since touching them can trip write watch.
        unsafe {
            first.0.cast::<u8>().write_bytes(0, watched);
            first.0.cast::<u32>().write(13);
            second.0.cast::<u32>().write(27);
        }
        let shadow = crate::shadow::find(start as u64, len).unwrap();
        assert_eq!(shadow.update(start, watched), watched);
        assert_eq!(shadow.update(start, watched), 0);

        // A small view in the next allocation is copied as its whole page
        // (guarded there). A large conventional view on the same shadow can
        // still use write watching.
        assert_eq!(shadow.update(second.0 as usize, 4), PAGE);
        let refreshed: usize = (0..4).map(|_| shadow.update(start, watched)).sum();
        eprintln!(
            "4-byte unwatched refresh, then 4 clean watched requests copied {refreshed} bytes"
        );
        assert_eq!(refreshed, 0);

        // SAFETY: the first allocation remains live.
        unsafe { first.0.cast::<u32>().write(500) };
        assert_eq!(shadow.update(start, watched), PAGE);
        // SAFETY: the persistently mapped shadow owns the refreshed word.
        let mut data = std::ptr::null_mut();
        // SAFETY: the live buffer contains the refreshed word. No GPU work is in flight.
        unsafe {
            shadow.buffer.Map(0, None, Some(&mut data)).unwrap();
            let word = data.cast::<u32>().read_unaligned();
            shadow.buffer.Unmap(0, None);
            assert_eq!(word, 500);
        }

        // A later small refresh across the allocations' border refetches at
        // most the border page of the watched allocation, then nothing.
        assert!(shadow.update(start + watched - 4, 8) >= 8);
        assert!(shadow.update(start, watched) <= PAGE);
        assert_eq!(shadow.update(start, watched), 0);
    }

    /// Two adjacent allocations, the first write-watched, the second
    /// write-watched if `watched`, each one page; kept reserved (the journal
    /// remembers their pages).
    fn neighbours(watched: bool) -> usize {
        for _ in 0..16 {
            // SAFETY: plain test reservations.
            unsafe {
                let span = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS);
                assert!(!span.is_null());
                assert_ne!(VirtualFree(span, 0, MEM_RELEASE), 0);
                let kind = MEM_RESERVE | MEM_COMMIT;
                let a = VirtualAlloc(span, PAGE, kind | MEM_WRITE_WATCH, PAGE_READWRITE);
                if a.is_null() {
                    continue;
                }
                let second = span.cast::<u8>().add(PAGE).cast();
                let b = VirtualAlloc(second, PAGE, kind | if watched { MEM_WRITE_WATCH } else { 0 }, PAGE_READWRITE);
                if b.is_null() {
                    VirtualFree(a, 0, MEM_RELEASE);
                    continue;
                }
                return span as usize;
            }
        }
        panic!("no two adjacent allocations");
    }

    #[test]
    fn a_range_over_watched_allocations_is_asked_page_by_page() {
        let p = neighbours(true);
        let (start, end) = (p, p + 2 * PAGE);
        let (before, _) = writes(start, end, false).expect("watched, page by page");
        // SAFETY: committed pages of ours.
        unsafe {
            ((p + 8) as *mut u8).write(1);
            ((p + PAGE + 8) as *mut u8).write(1);
        }
        let (now, after) = writes(start, end, false).expect("watched, page by page");
        let after: Vec<_> = after.into_iter().filter(|&(_, w)| w > before).collect();
        assert_eq!(after, [(p / PAGE, now), (p / PAGE + 1, now)]);
    }

    #[test]
    fn a_write_found_before_an_unwatched_page_is_kept() {
        let p = neighbours(false);
        let (before, _) = writes(p, p + PAGE, false).expect("watched");
        // SAFETY: a committed page of ours.
        unsafe { ((p + 8) as *mut u8).write(1) };
        // The second allocation is not watched: refused, as before.
        assert!(writes(p, p + 2 * PAGE, false).is_none());
        // The first page's write was cleared in the kernel by that asking;
        // the journal still has it.
        let (_, again) = writes(p, p + PAGE, false).expect("watched");
        assert!(again.iter().any(|&(page, w)| page == p / PAGE && w > before));
    }

    /// Adjacent one-page allocations of the given `VirtualAlloc` kinds
    /// (read-write where committed); kept reserved.
    fn pages_of(kinds: &[u32]) -> usize {
        for _ in 0..16 {
            // SAFETY: plain test reservations.
            unsafe {
                let n = kinds.len() * PAGE;
                let span = VirtualAlloc(std::ptr::null(), n, MEM_RESERVE, PAGE_NOACCESS);
                assert!(!span.is_null());
                assert_ne!(VirtualFree(span, 0, MEM_RELEASE), 0);
                let mut made = Vec::new();
                for (i, &kind) in kinds.iter().enumerate() {
                    let at = span.cast::<u8>().add(i * PAGE).cast();
                    let protect = if kind & MEM_COMMIT != 0 { PAGE_READWRITE } else { PAGE_NOACCESS };
                    let p = VirtualAlloc(at, PAGE, kind, protect);
                    if p.is_null() {
                        break;
                    }
                    made.push(p);
                }
                if made.len() == kinds.len() {
                    return span as usize;
                }
                for p in made {
                    VirtualFree(p, 0, MEM_RELEASE);
                }
            }
        }
        panic!("no adjacent allocations");
    }

    const WATCHED: u32 = MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH;

    #[test]
    fn a_range_over_watched_pages_and_a_hole_is_tracked() {
        let p = pages_of(&[WATCHED, MEM_RESERVE, WATCHED]);
        let (start, end) = (p, p + 3 * PAGE);
        let (before, _) = writes(start, end, false).expect("watched pages and a hole");
        // SAFETY: committed pages of ours.
        unsafe {
            ((p + 8) as *mut u8).write(1);
            ((p + 2 * PAGE + 8) as *mut u8).write(1);
        }
        let (now, after) = writes(start, end, false).expect("watched pages and a hole");
        let after: Vec<_> = after.into_iter().filter(|&(_, w)| w > before).collect();
        assert_eq!(after, [(p / PAGE, now), (p / PAGE + 2, now)]);
    }

    #[test]
    fn a_remapped_page_is_written() {
        let p = pages_of(&[WATCHED, MEM_RESERVE]);
        let (before, _) = writes(p, p + 2 * PAGE, false).expect("tracked");
        title_memory_remapped(p + PAGE, PAGE);
        let (_, after) = writes(p, p + 2 * PAGE, false).expect("tracked");
        assert!(after.iter().any(|&(page, w)| page == p / PAGE + 1 && w > before));
    }

    #[test]
    fn a_hole_committed_later_is_not_taken_as_unchanged() {
        let p = pages_of(&[WATCHED, MEM_RESERVE]);
        assert!(writes(p, p + 2 * PAGE, false).is_some());
        // Committed in place (not watched): no longer answerable.
        // SAFETY: the reserved page of ours.
        let at = unsafe { VirtualAlloc((p + PAGE) as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!at.is_null());
        assert!(writes(p, p + 2 * PAGE, false).is_none());
    }

    #[test]
    fn small_refreshes_of_watched_memory_copy_whole_pages_once() {
        if crate::host::host().is_none() {
            return;
        }
        let size = 4 * PAGE;
        // SAFETY: a fresh write-watched allocation, kept reserved (below).
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                size,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        // SAFETY: committed pages of ours. Touched first: reading a
        // demand-zero page faults it in, which write watching reports.
        unsafe { (base as *mut u8).write_bytes(0, size) };
        crate::shadow::register(base, base as u64, size as u64);
        let shadow = crate::shadow::find(base as u64, size as u64).unwrap();
        // Small views (the world's directly indexed ones): eight in the
        // first page, one in the third, none covering a page whole.
        let views = [0, 64, 128, 1024, 4096, 8192, 16384, 32768, 2 * PAGE + 16];
        let refresh = || -> usize { views.iter().map(|&at| shadow.update(base + at, 4)).sum() };
        // First: the two pages, whole. They were "never copied" before, and
        // a page only partly copied stayed so, copied again at every refresh.
        assert_eq!(refresh(), 2 * PAGE);
        assert_eq!(refresh(), 0);
        assert_eq!(refresh(), 0);
        // A title write brings back only its page, with the new bytes.
        // SAFETY: a committed page of ours.
        unsafe { ((base + 64) as *mut u32).write(500) };
        assert_eq!(refresh(), PAGE);
        let mut data = std::ptr::null_mut();
        // SAFETY: the live shadow holds the refreshed word; no GPU work.
        unsafe {
            shadow.buffer.Map(0, None, Some(&mut data)).unwrap();
            let word = data.cast::<u8>().add(64).cast::<u32>().read_unaligned();
            shadow.buffer.Unmap(0, None);
            assert_eq!(word, 500);
        }
        crate::shadow::unregister(base, base as u64);
        // The allocation stays reserved: the write journal remembers its
        // pages, and another test's allocation there would inherit them.
    }
}
