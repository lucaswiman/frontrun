//! LD_PRELOAD / DYLD_INSERT_LIBRARIES library for frontrun I/O interception.
//!
//! Intercepts libc I/O syscall wrappers (`connect`, `send`, `recv`, `read`,
//! `write`, `close`, etc.) to maintain a file-descriptor → resource mapping.
//!
//! Events are transmitted to the Python side via one of two channels:
//!
//! 1. **Pipe (preferred):** When `FRONTRUN_IO_FD` is set to a file descriptor
//!    number, events are written directly to that fd (typically the write end
//!    of an `os.pipe()`).  The fd stays open for the process lifetime, so
//!    there is no open/close overhead per event.  The pipe's FIFO ordering
//!    provides a natural total order without timestamps.
//!
//! 2. **Log file (legacy fallback):** When only `FRONTRUN_IO_LOG` is set,
//!    events are appended to the named file (open + write + close per event).
//!
//! ## Platform differences
//!
//! **Linux:** The dynamic linker resolves symbols in LD_PRELOAD libraries first,
//! so defining `write()` etc. directly interposes them.  We forward to the real
//! libc functions via `dlsym(RTLD_NEXT, ...)`.
//!
//! **macOS:** Two-level namespaces prevent simple symbol overriding.  Instead we
//! define `frontrun_write()` etc. and register a `__DATA,__interpose` table that
//! tells dyld to replace calls to libSystem's functions with ours.  Forwarding
//! uses raw arm64 syscalls (`svc #0x80`) to avoid infinite recursion (since
//! `dlsym` itself calls `write`/`close` internally).

use libc::{c_int, c_void, iovec, msghdr, size_t, sockaddr, socklen_t, ssize_t, AF_INET, AF_INET6, AF_UNIX};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Mutex;

mod sql_extract;

#[cfg(not(target_os = "macos"))]
use libc::{c_char, RTLD_NEXT};
#[cfg(not(target_os = "macos"))]
use std::sync::atomic::AtomicUsize;

// Fail at compile time on unsupported macOS architectures.
#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
compile_error!("macOS I/O interception requires aarch64 (Apple Silicon)");

// ---------------------------------------------------------------------------
// Library readiness flag
// ---------------------------------------------------------------------------
//
// On macOS, our interposed functions are called during libSystem_initializer
// (before main), when TLS and the allocator are not yet available.  We skip
// interception until our constructor runs (after libSystem init completes).
// On Linux, LD_PRELOAD doesn't have this issue, so READY defaults to true.

#[cfg(target_os = "macos")]
static READY: AtomicBool = AtomicBool::new(false);

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
static READY: AtomicBool = AtomicBool::new(true);

/// macOS library constructor — registered via `__DATA,__mod_init_func`.
/// dyld calls this after libSystem is initialized but before main().
#[cfg(target_os = "macos")]
#[used]
#[link_section = "__DATA,__mod_init_func"]
static INIT_FN: unsafe extern "C" fn() = {
    unsafe extern "C" fn _frontrun_io_init() {
        READY.store(true, Ordering::Release);
    }
    _frontrun_io_init
};

// ---------------------------------------------------------------------------
// Real libc function pointers — Linux only (resolved lazily via dlsym)
// ---------------------------------------------------------------------------
//
// On macOS we use raw syscalls instead, so dlsym is not needed.

#[cfg(not(target_os = "macos"))]
macro_rules! resolve {
    ($name:ident, $ty:ty) => {{
        static ADDR: AtomicUsize = AtomicUsize::new(0);
        static RESOLVED: AtomicBool = AtomicBool::new(false);
        if !RESOLVED.load(Ordering::Acquire) {
            let sym = unsafe {
                libc::dlsym(
                    RTLD_NEXT,
                    concat!(stringify!($name), "\0").as_ptr() as *const c_char,
                )
            };
            if !sym.is_null() {
                ADDR.store(sym as usize, Ordering::Release);
            }
            RESOLVED.store(true, Ordering::Release);
        }
        let addr = ADDR.load(Ordering::Acquire);
        if addr != 0 {
            Some(unsafe { std::mem::transmute::<usize, $ty>(addr) })
        } else {
            None
        }
    }};
}

// ---------------------------------------------------------------------------
// Portable errno helpers
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
unsafe fn set_errno(val: c_int) {
    *libc::__errno_location() = val;
}

#[cfg(target_os = "macos")]
unsafe fn set_errno(val: c_int) {
    *libc::__error() = val;
}

#[cfg(target_os = "linux")]
unsafe fn get_errno() -> c_int {
    *libc::__errno_location()
}

#[cfg(target_os = "macos")]
unsafe fn get_errno() -> c_int {
    *libc::__error()
}

// ---------------------------------------------------------------------------
// Portable thread-id helper
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn get_tid() -> i64 {
    unsafe { libc::syscall(libc::SYS_gettid) as i64 }
}

#[cfg(target_os = "macos")]
fn get_tid() -> i64 {
    let mut tid: u64 = 0;
    unsafe {
        libc::pthread_threadid_np(0 as libc::pthread_t, &mut tid);
    }
    tid as i64
}

// ---------------------------------------------------------------------------
// Raw arm64 syscall wrappers — macOS only
// ---------------------------------------------------------------------------
//
// On macOS, `dlsym(RTLD_NEXT, ...)` causes infinite recursion when called
// from an interposed function because dlsym internally calls `write`/`close`.
// Using raw `svc #0x80` syscalls bypasses libc entirely and avoids this.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod raw_syscall {
    use libc::{c_char, c_int, c_void, iovec, msghdr, size_t, sockaddr, socklen_t, ssize_t};

    // macOS encodes the syscall class in the upper bits.  UNIX syscalls use
    // class 2 (0x2000000).
    const SYS_CLASS_UNIX: u64 = 0x2000000;

    const SYS_READ: u64 = SYS_CLASS_UNIX | 3;
    const SYS_WRITE: u64 = SYS_CLASS_UNIX | 4;
    const SYS_OPEN: u64 = SYS_CLASS_UNIX | 5;
    const SYS_CLOSE: u64 = SYS_CLASS_UNIX | 6;
    const SYS_GETPID: u64 = SYS_CLASS_UNIX | 20;
    const SYS_RECVMSG: u64 = SYS_CLASS_UNIX | 27;
    const SYS_SENDMSG: u64 = SYS_CLASS_UNIX | 28;
    const SYS_RECVFROM: u64 = SYS_CLASS_UNIX | 29;
    const SYS_GETPEERNAME: u64 = SYS_CLASS_UNIX | 31;
    const SYS_FCNTL: u64 = SYS_CLASS_UNIX | 92;
    const SYS_CONNECT: u64 = SYS_CLASS_UNIX | 98;
    const SYS_READV: u64 = SYS_CLASS_UNIX | 120;
    const SYS_WRITEV: u64 = SYS_CLASS_UNIX | 121;
    const SYS_SENDTO: u64 = SYS_CLASS_UNIX | 133;

    /// Execute a raw arm64 syscall with up to 6 arguments.
    ///
    /// On return, if the carry flag is set the call failed and x0 contains the
    /// errno value.  We store it via `set_errno` and return -1.
    #[inline(always)]
    unsafe fn syscall6(num: u64, a0: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64) -> i64 {
        let ret: u64;
        let err: u64;
        core::arch::asm!(
            "svc #0x80",
            "cset {err:w}, cs",
            inout("x16") num => _,
            inout("x0") a0 => ret,
            inout("x1") a1 => _,
            inout("x2") a2 => _,
            inout("x3") a3 => _,
            inout("x4") a4 => _,
            inout("x5") a5 => _,
            err = out(reg) err,
            options(nostack),
        );
        if err != 0 {
            super::set_errno(ret as c_int);
            -1
        } else {
            ret as i64
        }
    }

    pub unsafe fn write(fd: c_int, buf: *const c_void, count: size_t) -> ssize_t {
        syscall6(SYS_WRITE, fd as u64, buf as u64, count as u64, 0, 0, 0) as ssize_t
    }

    pub unsafe fn read(fd: c_int, buf: *mut c_void, count: size_t) -> ssize_t {
        syscall6(SYS_READ, fd as u64, buf as u64, count as u64, 0, 0, 0) as ssize_t
    }

    pub unsafe fn open(path: *const c_char, flags: c_int, mode: c_int) -> c_int {
        syscall6(SYS_OPEN, path as u64, flags as u64, mode as u64, 0, 0, 0) as c_int
    }

    pub unsafe fn close(fd: c_int) -> c_int {
        syscall6(SYS_CLOSE, fd as u64, 0, 0, 0, 0, 0) as c_int
    }

    pub unsafe fn connect(fd: c_int, addr: *const sockaddr, addrlen: socklen_t) -> c_int {
        syscall6(
            SYS_CONNECT,
            fd as u64,
            addr as u64,
            addrlen as u64,
            0,
            0,
            0,
        ) as c_int
    }

    pub unsafe fn sendto(
        fd: c_int,
        buf: *const c_void,
        len: size_t,
        flags: c_int,
        addr: *const sockaddr,
        addrlen: socklen_t,
    ) -> ssize_t {
        syscall6(
            SYS_SENDTO,
            fd as u64,
            buf as u64,
            len as u64,
            flags as u64,
            addr as u64,
            addrlen as u64,
        ) as ssize_t
    }

    /// `send()` has no dedicated macOS syscall — it is `sendto` with a NULL address.
    pub unsafe fn send(fd: c_int, buf: *const c_void, len: size_t, flags: c_int) -> ssize_t {
        sendto(fd, buf, len, flags, std::ptr::null(), 0)
    }

    pub unsafe fn sendmsg(fd: c_int, msg: *const msghdr, flags: c_int) -> ssize_t {
        syscall6(SYS_SENDMSG, fd as u64, msg as u64, flags as u64, 0, 0, 0) as ssize_t
    }

    pub unsafe fn recvfrom(
        fd: c_int,
        buf: *mut c_void,
        len: size_t,
        flags: c_int,
        addr: *mut sockaddr,
        addrlen: *mut socklen_t,
    ) -> ssize_t {
        syscall6(
            SYS_RECVFROM,
            fd as u64,
            buf as u64,
            len as u64,
            flags as u64,
            addr as u64,
            addrlen as u64,
        ) as ssize_t
    }

    /// `recv()` has no dedicated macOS syscall — it is `recvfrom` with NULL address/len.
    pub unsafe fn recv(fd: c_int, buf: *mut c_void, len: size_t, flags: c_int) -> ssize_t {
        recvfrom(fd, buf, len, flags, std::ptr::null_mut(), std::ptr::null_mut())
    }

    pub unsafe fn recvmsg(fd: c_int, msg: *mut msghdr, flags: c_int) -> ssize_t {
        syscall6(SYS_RECVMSG, fd as u64, msg as u64, flags as u64, 0, 0, 0) as ssize_t
    }

    pub unsafe fn writev(fd: c_int, iov: *const iovec, iovcnt: c_int) -> ssize_t {
        syscall6(
            SYS_WRITEV,
            fd as u64,
            iov as u64,
            iovcnt as u64,
            0,
            0,
            0,
        ) as ssize_t
    }

    pub unsafe fn readv(fd: c_int, iov: *const iovec, iovcnt: c_int) -> ssize_t {
        syscall6(
            SYS_READV,
            fd as u64,
            iov as u64,
            iovcnt as u64,
            0,
            0,
            0,
        ) as ssize_t
    }

    pub unsafe fn getpid() -> c_int {
        syscall6(SYS_GETPID, 0, 0, 0, 0, 0, 0) as c_int
    }

    pub unsafe fn getpeername(
        fd: c_int,
        addr: *mut sockaddr,
        addrlen: *mut socklen_t,
    ) -> c_int {
        syscall6(
            SYS_GETPEERNAME,
            fd as u64,
            addr as u64,
            addrlen as u64,
            0,
            0,
            0,
        ) as c_int
    }

    pub unsafe fn fcntl(fd: c_int, cmd: c_int, arg: *mut c_void) -> c_int {
        syscall6(SYS_FCNTL, fd as u64, cmd as u64, arg as u64, 0, 0, 0) as c_int
    }
}

// ---------------------------------------------------------------------------
// Per-thread reentrancy guard
// ---------------------------------------------------------------------------
//
// Prevents infinite recursion when our logging code calls write()/close()
// which would trigger our interceptors again.  On macOS this is belt-and-
// suspenders since raw syscalls already bypass interposition, but it's cheap
// insurance.

thread_local! {
    static IN_HOOK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct ReentrancyGuard;

impl ReentrancyGuard {
    fn enter() -> Option<Self> {
        IN_HOOK.with(|flag| {
            if flag.get() {
                None
            } else {
                flag.set(true);
                Some(ReentrancyGuard)
            }
        })
    }
}

impl Drop for ReentrancyGuard {
    fn drop(&mut self) {
        IN_HOOK.with(|flag| flag.set(false));
    }
}

// ---------------------------------------------------------------------------
// Global fd → resource map
// ---------------------------------------------------------------------------

struct FdMap {
    map: HashMap<c_int, String>,
}

impl FdMap {
    fn new() -> Self {
        FdMap {
            map: HashMap::new(),
        }
    }

    fn insert(&mut self, fd: c_int, resource: String) {
        self.map.insert(fd, resource);
    }

    fn get(&self, fd: c_int) -> Option<&String> {
        self.map.get(&fd)
    }

    fn remove(&mut self, fd: c_int) -> Option<String> {
        self.map.remove(&fd)
    }
}

static FD_MAP: once_cell::sync::Lazy<Mutex<FdMap>> =
    once_cell::sync::Lazy::new(|| Mutex::new(FdMap::new()));

// ---------------------------------------------------------------------------
// Pipe-based event transport (FRONTRUN_IO_FD)
// ---------------------------------------------------------------------------
//
// When set, events are written to this persistent fd (the write end of a
// pipe created by Python's IOEventDispatcher).  Much faster than file I/O
// because there is no open/close per event.

static PIPE_FD: AtomicI32 = AtomicI32::new(-1);
static PIPE_FD_CHECKED: AtomicBool = AtomicBool::new(false);

/// The read end of the event pipe.  Reads on this fd must bypass
/// interception entirely — otherwise the LD_PRELOAD `read()` hook
/// acquires `FD_MAP` and calls `ensure_fd_mapped()`, adding overhead
/// and potential contention that can deadlock the pipe reader thread
/// on free-threaded Python.
static PIPE_READ_FD: AtomicI32 = AtomicI32::new(-1);

fn get_pipe_fd() -> Option<c_int> {
    if !PIPE_FD_CHECKED.load(Ordering::Acquire) {
        // Parse the env value, then only commit it if we are the thread that
        // flips PIPE_FD_CHECKED from false → true. This makes an explicit
        // frontrun_io_set_pipe_fd() call (which also sets PIPE_FD_CHECKED) win
        // the race: if set_pipe_fd ran concurrently and already flipped the
        // flag, our compare_exchange fails and we do NOT clobber its value
        // with the stale env-derived fd.
        let env_fd = std::env::var("FRONTRUN_IO_FD")
            .ok()
            .and_then(|s| s.parse::<c_int>().ok());
        if PIPE_FD_CHECKED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            if let Some(fd) = env_fd {
                PIPE_FD.store(fd, Ordering::Release);
            }
        }
    }
    let fd = PIPE_FD.load(Ordering::Acquire);
    if fd >= 0 {
        Some(fd)
    } else {
        None
    }
}

/// Directly set (or reset) the pipe fd used for event transport.
///
/// Called from Python via ctypes when [`IOEventDispatcher`] creates a pipe
/// after the library has already been loaded (and `get_pipe_fd` has cached
/// the initial "not set" state from process startup).
///
/// Pass `-1` to disable pipe transport (reverts to log-file fallback).
#[no_mangle]
pub extern "C" fn frontrun_io_set_pipe_fd(fd: c_int) {
    PIPE_FD.store(fd, Ordering::Release);
    PIPE_FD_CHECKED.store(true, Ordering::Release);
}

/// Set (or clear) the read end of the event pipe so that `read()`
/// interception skips it entirely.  Pass `-1` to clear.
#[no_mangle]
pub extern "C" fn frontrun_io_set_pipe_read_fd(fd: c_int) {
    PIPE_READ_FD.store(fd, Ordering::Release);
}

/// Returns true if `fd` is one of the pipe fds used for event transport
/// and should be excluded from interception.
#[inline]
fn is_pipe_fd(fd: c_int) -> bool {
    fd == PIPE_FD.load(Ordering::Relaxed) || fd == PIPE_READ_FD.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Low-level write helper (uses real write, not our interceptor)
// ---------------------------------------------------------------------------

// A single event line stays within one write() call as long as it is at most
// PIPE_BUF bytes, which POSIX guarantees to be atomic on a pipe even with
// concurrent writers. With the resource-escaping fix, an event is exactly one
// line, so lines up to PIPE_BUF interleave cleanly. Lines longer than PIPE_BUF
// (very large SQL statements) may still interleave with other threads' writes
// mid-line — full atomicity beyond PIPE_BUF is not cheaply achievable here.
//
// `write_one` performs a single raw write of `buf`, returning the byte count
// (>=0) or -1 with errno set. `write_to_fd` loops over EINTR and short writes
// so a signal-interrupted or partial write never silently truncates a line.

#[cfg(target_os = "macos")]
unsafe fn write_one(fd: c_int, buf: &[u8]) -> ssize_t {
    raw_syscall::write(fd, buf.as_ptr() as *const c_void, buf.len())
}

#[cfg(not(target_os = "macos"))]
unsafe fn write_one(fd: c_int, buf: &[u8]) -> ssize_t {
    type WriteFn = unsafe extern "C" fn(c_int, *const c_void, size_t) -> ssize_t;
    match resolve!(write, WriteFn) {
        Some(real_write) => real_write(fd, buf.as_ptr() as *const c_void, buf.len()),
        None => -1,
    }
}

fn write_to_fd(fd: c_int, buf: &[u8]) {
    let mut written: usize = 0;
    while written < buf.len() {
        let n = unsafe { write_one(fd, &buf[written..]) };
        if n > 0 {
            written += n as usize;
            continue;
        }
        if n == 0 {
            // No progress and no error reported — avoid spinning forever.
            break;
        }
        // n < 0: inspect errno for retryable conditions.
        let err = unsafe { get_errno() };
        if err == libc::EINTR {
            continue; // interrupted by a signal before any byte was written
        }
        // EAGAIN/EWOULDBLOCK on a non-blocking fd, or any hard error: give up.
        // (The event pipe is created blocking on the write end, so EAGAIN is
        // not expected there; bailing avoids a busy-spin if it ever occurs.)
        break;
    }
}

// ---------------------------------------------------------------------------
// Event logging
// ---------------------------------------------------------------------------

/// Report an I/O event for a tracked fd. No-op if reentrant or fd is unknown.
fn report_io(fd: c_int, kind: &str) {
    let _guard = match ReentrancyGuard::enter() {
        Some(g) => g,
        None => return,
    };

    let map = match FD_MAP.lock() {
        Ok(m) => m,
        Err(_) => return, // poisoned lock — skip
    };
    let resource = match map.get(fd) {
        Some(r) => r.clone(),
        None => return, // unknown fd
    };
    drop(map);

    log_event(kind, &resource, fd);
}

/// Try to extract SQL from a PostgreSQL wire-protocol `send()` buffer.
/// Reports a `sql_write` event if SQL is found, otherwise falls back
/// to the standard `report_io(fd, "write")` path.
///
/// # Safety
/// `buf` must point to at least `len` valid bytes.
unsafe fn report_send_buf(fd: c_int, buf: *const c_void, len: size_t) {
    // `slice::from_raw_parts(NULL, 0)` is UB regardless of len, and a zero
    // length can never contain a query anyway — fall straight through to the
    // generic write report in either case.
    if buf.is_null() || len == 0 {
        report_io(fd, "write");
        return;
    }
    let buf_slice = std::slice::from_raw_parts(buf as *const u8, len);
    if let Some(sql) = sql_extract::extract_pg_query(buf_slice) {
        log_event("sql_write", sql, fd);
    } else {
        report_io(fd, "write");
    }
}

/// Escape a resource string so it can be embedded in a single tab-separated,
/// newline-terminated event line without breaking the framing.
///
/// Resources can contain arbitrary bytes — raw SQL (for `sql_write` events)
/// routinely contains tabs and newlines, and file paths can contain odd
/// bytes too. We escape backslash, tab, and newline byte-wise so the Python
/// parser can split on raw `\t` / `\n` and then unescape exactly once.
///
/// Pure function (no I/O; allocates only when escaping is needed) so it can
/// be unit-tested directly under `cargo test`.
fn escape_resource(resource: &str) -> std::borrow::Cow<'_, str> {
    // Fast path: the common resources (socket endpoints, file paths, fd
    // labels) contain none of the framing-relevant bytes — borrow as-is
    // instead of rebuilding the string on every logged event.
    if !resource
        .bytes()
        .any(|b| matches!(b, b'\\' | b'\t' | b'\n'))
    {
        return std::borrow::Cow::Borrowed(resource);
    }
    let mut out = String::with_capacity(resource.len() + 8);
    // `resource` is valid UTF-8 (it's a &str). None of the escaped bytes
    // (\\, \t, \n — all < 0x80) ever appear inside a multi-byte UTF-8
    // sequence, so iterating by char preserves any non-ASCII content while
    // still escaping the framing-relevant characters.
    for ch in resource.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            _ => out.push(ch),
        }
    }
    std::borrow::Cow::Owned(out)
}

fn log_event(kind: &str, resource: &str, fd: c_int) {
    #[cfg(target_os = "macos")]
    let pid = unsafe { raw_syscall::getpid() };
    #[cfg(not(target_os = "macos"))]
    let pid = unsafe { libc::getpid() };

    let tid = get_tid();
    // Escape the resource field so tabs/newlines/backslashes (common in raw
    // SQL, possible in file paths) cannot corrupt the TSV line framing. The
    // Python parser unescapes exactly once. `kind` is a fixed internal string
    // with no special characters, so it needs no escaping.
    let escaped = escape_resource(resource);
    let line = format!("{}\t{}\t{}\t{}\t{}\n", kind, escaped, fd, pid, tid);
    let buf = line.as_bytes();

    // Prefer pipe fd (FRONTRUN_IO_FD) — no open/close overhead.
    if let Some(pipe_fd) = get_pipe_fd() {
        write_to_fd(pipe_fd, buf);
        return;
    }

    // Fall back to log file (FRONTRUN_IO_LOG) — opens and closes per event.
    let log_path = match std::env::var("FRONTRUN_IO_LOG") {
        Ok(p) => p,
        Err(_) => return,
    };

    let path_cstr = match std::ffi::CString::new(log_path) {
        Ok(c) => c,
        Err(_) => return,
    };

    #[cfg(target_os = "macos")]
    let log_fd = unsafe {
        raw_syscall::open(
            path_cstr.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_APPEND,
            0o644,
        )
    };
    #[cfg(not(target_os = "macos"))]
    let log_fd = unsafe {
        libc::open(
            path_cstr.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_APPEND,
            0o644,
        )
    };

    if log_fd < 0 {
        return;
    }

    write_to_fd(log_fd, buf);

    // Close the log file fd using platform-appropriate method.
    #[cfg(target_os = "macos")]
    unsafe {
        raw_syscall::close(log_fd);
    }
    #[cfg(not(target_os = "macos"))]
    {
        type CloseFn = unsafe extern "C" fn(c_int) -> c_int;
        if let Some(real_close) = resolve!(close, CloseFn) {
            unsafe {
                real_close(log_fd);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// sockaddr → resource string
// ---------------------------------------------------------------------------

/// Render the path portion of an `AF_UNIX` address into a resource string.
///
/// `path_bytes` is the `sun_path` content bounded by `addrlen` (so it never
/// over-reads). Three cases:
///   * empty → unnamed socket (`socket:unix:`)
///   * leading NUL → Linux abstract socket; the name is the *remaining* bytes
///     (NOT null-terminated, may contain arbitrary bytes). We include the
///     name so distinct abstract sockets don't all collapse to one resource
///     and create false conflicts.
///   * otherwise → filesystem path, null-terminated within `sun_path`.
///
/// Non-printable / non-UTF-8 bytes in either name form are hex-escaped
/// (`\xNN`) so the result is a stable, framing-safe ASCII string. Pure
/// function — unit-tested below.
fn unix_socket_resource(path_bytes: &[u8]) -> String {
    if path_bytes.is_empty() {
        return "socket:unix:".to_string();
    }
    if path_bytes[0] == 0 {
        // Abstract namespace: name is everything after the leading NUL,
        // bounded by addrlen (the caller already truncated to addrlen).
        let name = &path_bytes[1..];
        format!("socket:unix:abstract:{}", hex_escape_bytes(name))
    } else {
        let len = path_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(path_bytes.len());
        format!("socket:unix:{}", hex_escape_bytes(&path_bytes[..len]))
    }
}

/// Render bytes as an ASCII string, hex-escaping any byte that is not a
/// printable, non-control ASCII character (and escaping `\` itself).
fn hex_escape_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        if b == b'\\' {
            out.push_str("\\\\");
        } else if b.is_ascii_graphic() || b == b' ' {
            out.push(b as char);
        } else {
            out.push_str(&format!("\\x{:02x}", b));
        }
    }
    out
}

fn sockaddr_to_resource(addr: *const sockaddr, addrlen: socklen_t) -> Option<String> {
    if addr.is_null() || addrlen == 0 {
        return None;
    }

    unsafe {
        let family = (*addr).sa_family as c_int;

        if family == AF_INET && addrlen as usize >= std::mem::size_of::<libc::sockaddr_in>() {
            let sin = &*(addr as *const libc::sockaddr_in);
            let ip = u32::from_be(sin.sin_addr.s_addr);
            let port = u16::from_be(sin.sin_port);
            let a = (ip >> 24) & 0xff;
            let b = (ip >> 16) & 0xff;
            let c = (ip >> 8) & 0xff;
            let d = ip & 0xff;
            Some(format!("socket:{}.{}.{}.{}:{}", a, b, c, d, port))
        } else if family == AF_INET6
            && addrlen as usize >= std::mem::size_of::<libc::sockaddr_in6>()
        {
            let sin6 = &*(addr as *const libc::sockaddr_in6);
            let port = u16::from_be(sin6.sin6_port);
            let octets = sin6.sin6_addr.s6_addr;
            Some(format!(
                "socket:[{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}]:{}",
                octets[0], octets[1], octets[2], octets[3],
                octets[4], octets[5], octets[6], octets[7],
                octets[8], octets[9], octets[10], octets[11],
                octets[12], octets[13], octets[14], octets[15],
                port
            ))
        } else if family == AF_UNIX {
            let sun_path_offset = std::mem::offset_of!(libc::sockaddr_un, sun_path);
            if (addrlen as usize) < sun_path_offset {
                return None;
            }
            let sun = &*(addr as *const libc::sockaddr_un);
            // Only read as many path bytes as addrlen allows.
            let max_path_len = (addrlen as usize) - sun_path_offset;
            let path_bytes = std::slice::from_raw_parts(
                sun.sun_path.as_ptr() as *const u8,
                max_path_len.min(sun.sun_path.len()),
            );
            Some(unix_socket_resource(path_bytes))
        } else {
            None
        }
    }
}

/// Try to get the peer address of a connected socket and convert to resource string.
#[cfg(not(target_os = "macos"))]
fn fd_to_resource_via_getpeername(fd: c_int) -> Option<String> {
    unsafe {
        let mut addr: libc::sockaddr_storage = std::mem::zeroed();
        let mut addrlen: socklen_t =
            std::mem::size_of::<libc::sockaddr_storage>() as socklen_t;
        let ret = libc::getpeername(fd, &mut addr as *mut _ as *mut sockaddr, &mut addrlen);
        if ret == 0 {
            sockaddr_to_resource(&addr as *const _ as *const sockaddr, addrlen)
        } else {
            None
        }
    }
}

#[cfg(target_os = "macos")]
fn fd_to_resource_via_getpeername(fd: c_int) -> Option<String> {
    unsafe {
        let mut addr: libc::sockaddr_storage = std::mem::zeroed();
        let mut addrlen: socklen_t =
            std::mem::size_of::<libc::sockaddr_storage>() as socklen_t;
        let ret = raw_syscall::getpeername(
            fd,
            &mut addr as *mut _ as *mut sockaddr,
            &mut addrlen,
        );
        if ret == 0 {
            sockaddr_to_resource(&addr as *const _ as *const sockaddr, addrlen)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: check if fd is a regular file and get its path
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn fd_to_file_resource(fd: c_int) -> Option<String> {
    let link_path = format!("/proc/self/fd/{}", fd);
    match std::fs::read_link(&link_path) {
        Ok(target) => {
            let path_str = target.to_string_lossy();
            // Skip special files (pipes, sockets shown as socket:[...], anon_inode, etc.)
            if path_str.starts_with('/') && !path_str.contains("(deleted)") {
                Some(format!("file:{}", path_str))
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

#[cfg(target_os = "macos")]
fn fd_to_file_resource(fd: c_int) -> Option<String> {
    let mut buf = [0i8; libc::PATH_MAX as usize];
    let ret =
        unsafe { raw_syscall::fcntl(fd, libc::F_GETPATH, buf.as_mut_ptr() as *mut c_void) };
    if ret == -1 {
        return None;
    }
    let cstr = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
    let path_str = cstr.to_string_lossy();
    if path_str.starts_with('/') {
        Some(format!("file:{}", path_str))
    } else {
        None
    }
}

/// Ensure we have a resource mapping for this fd (try socket first, then file).
fn ensure_fd_mapped(fd: c_int) {
    if fd < 0 {
        return;
    }
    let map = match FD_MAP.lock() {
        Ok(m) => m,
        Err(_) => return,
    };
    if map.get(fd).is_some() {
        return;
    }
    drop(map);

    // Try socket first
    if let Some(resource) = fd_to_resource_via_getpeername(fd) {
        if let Ok(mut map) = FD_MAP.lock() {
            map.insert(fd, resource);
        }
        return;
    }

    // Try file
    if let Some(resource) = fd_to_file_resource(fd) {
        if let Ok(mut map) = FD_MAP.lock() {
            map.insert(fd, resource);
        }
    }
}

// ===========================================================================
// Intercepted libc functions — Linux (LD_PRELOAD + dlsym)
// ===========================================================================

#[cfg(not(target_os = "macos"))]
mod linux_intercept {
    use super::*;

    /// Intercept `connect()` — record fd → endpoint mapping.
    #[no_mangle]
    pub unsafe extern "C" fn connect(
        fd: c_int,
        addr: *const sockaddr,
        addrlen: socklen_t,
    ) -> c_int {
        type ConnectFn = unsafe extern "C" fn(c_int, *const sockaddr, socklen_t) -> c_int;
        let real = match resolve!(connect, ConnectFn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        let result = real(fd, addr, addrlen);
        // Save errno immediately: non-blocking connect returns -1/EINPROGRESS
        // and callers ALWAYS read errno. The logging below (map insertion,
        // dlsym resolution, pipe/file writes) can clobber errno, so we restore
        // the real call's errno just before returning.
        let saved_errno = get_errno();

        if result == 0 || saved_errno == libc::EINPROGRESS {
            if let Some(_guard) = ReentrancyGuard::enter() {
                if let Some(resource) = sockaddr_to_resource(addr, addrlen) {
                    if let Ok(mut map) = FD_MAP.lock() {
                        map.insert(fd, resource.clone());
                    }
                    log_event("connect", &resource, fd);
                }
            }
        }

        set_errno(saved_errno);
        result
    }

    /// Intercept `send()`.
    #[no_mangle]
    pub unsafe extern "C" fn send(
        fd: c_int,
        buf: *const c_void,
        len: size_t,
        flags: c_int,
    ) -> ssize_t {
        type SendFn = unsafe extern "C" fn(c_int, *const c_void, size_t, c_int) -> ssize_t;
        let real = match resolve!(send, SendFn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        ensure_fd_mapped(fd);
        report_send_buf(fd, buf, len);
        real(fd, buf, len, flags)
    }

    /// Intercept `sendto()`.
    #[no_mangle]
    pub unsafe extern "C" fn sendto(
        fd: c_int,
        buf: *const c_void,
        len: size_t,
        flags: c_int,
        dest_addr: *const sockaddr,
        addrlen: socklen_t,
    ) -> ssize_t {
        type SendtoFn = unsafe extern "C" fn(
            c_int,
            *const c_void,
            size_t,
            c_int,
            *const sockaddr,
            socklen_t,
        ) -> ssize_t;
        let real = match resolve!(sendto, SendtoFn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        if !dest_addr.is_null() {
            if let Some(_guard) = ReentrancyGuard::enter() {
                if let Some(resource) = sockaddr_to_resource(dest_addr, addrlen) {
                    if let Ok(mut map) = FD_MAP.lock() {
                        map.insert(fd, resource);
                    }
                }
            }
        }

        ensure_fd_mapped(fd);
        report_io(fd, "write");
        real(fd, buf, len, flags, dest_addr, addrlen)
    }

    /// Intercept `sendmsg()`.
    #[no_mangle]
    pub unsafe extern "C" fn sendmsg(fd: c_int, msg: *const msghdr, flags: c_int) -> ssize_t {
        type SendmsgFn = unsafe extern "C" fn(c_int, *const msghdr, c_int) -> ssize_t;
        let real = match resolve!(sendmsg, SendmsgFn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        ensure_fd_mapped(fd);
        report_io(fd, "write");
        real(fd, msg, flags)
    }

    /// Intercept `write()`.
    #[no_mangle]
    pub unsafe extern "C" fn write(fd: c_int, buf: *const c_void, count: size_t) -> ssize_t {
        type WriteFn = unsafe extern "C" fn(c_int, *const c_void, size_t) -> ssize_t;
        let real = match resolve!(write, WriteFn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        if fd <= 2 || is_pipe_fd(fd) {
            return real(fd, buf, count);
        }

        ensure_fd_mapped(fd);
        report_io(fd, "write");
        real(fd, buf, count)
    }

    /// Intercept `writev()`.
    #[no_mangle]
    pub unsafe extern "C" fn writev(fd: c_int, iov: *const iovec, iovcnt: c_int) -> ssize_t {
        type WritevFn = unsafe extern "C" fn(c_int, *const iovec, c_int) -> ssize_t;
        let real = match resolve!(writev, WritevFn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        if fd <= 2 || is_pipe_fd(fd) {
            return real(fd, iov, iovcnt);
        }

        ensure_fd_mapped(fd);
        report_io(fd, "write");
        real(fd, iov, iovcnt)
    }

    /// Intercept `recv()`.
    #[no_mangle]
    pub unsafe extern "C" fn recv(
        fd: c_int,
        buf: *mut c_void,
        len: size_t,
        flags: c_int,
    ) -> ssize_t {
        type RecvFn = unsafe extern "C" fn(c_int, *mut c_void, size_t, c_int) -> ssize_t;
        let real = match resolve!(recv, RecvFn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        ensure_fd_mapped(fd);
        report_io(fd, "read");
        real(fd, buf, len, flags)
    }

    /// Intercept `recvfrom()`.
    #[no_mangle]
    pub unsafe extern "C" fn recvfrom(
        fd: c_int,
        buf: *mut c_void,
        len: size_t,
        flags: c_int,
        src_addr: *mut sockaddr,
        addrlen: *mut socklen_t,
    ) -> ssize_t {
        type RecvfromFn = unsafe extern "C" fn(
            c_int,
            *mut c_void,
            size_t,
            c_int,
            *mut sockaddr,
            *mut socklen_t,
        ) -> ssize_t;
        let real = match resolve!(recvfrom, RecvfromFn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        ensure_fd_mapped(fd);
        report_io(fd, "read");
        let result = real(fd, buf, len, flags, src_addr, addrlen);

        if result >= 0 && !src_addr.is_null() && !addrlen.is_null() {
            if let Some(_guard) = ReentrancyGuard::enter() {
                if let Some(resource) =
                    sockaddr_to_resource(src_addr as *const sockaddr, *addrlen)
                {
                    if let Ok(mut map) = FD_MAP.lock() {
                        map.insert(fd, resource);
                    }
                }
            }
        }

        result
    }

    /// Intercept `recvmsg()`.
    #[no_mangle]
    pub unsafe extern "C" fn recvmsg(fd: c_int, msg: *mut msghdr, flags: c_int) -> ssize_t {
        type RecvmsgFn = unsafe extern "C" fn(c_int, *mut msghdr, c_int) -> ssize_t;
        let real = match resolve!(recvmsg, RecvmsgFn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        ensure_fd_mapped(fd);
        report_io(fd, "read");
        real(fd, msg, flags)
    }

    /// Intercept `read()`.
    #[no_mangle]
    pub unsafe extern "C" fn read(fd: c_int, buf: *mut c_void, count: size_t) -> ssize_t {
        type ReadFn = unsafe extern "C" fn(c_int, *mut c_void, size_t) -> ssize_t;
        let real = match resolve!(read, ReadFn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        if fd <= 2 || is_pipe_fd(fd) {
            return real(fd, buf, count);
        }

        ensure_fd_mapped(fd);
        report_io(fd, "read");
        real(fd, buf, count)
    }

    /// Intercept `readv()`.
    #[no_mangle]
    pub unsafe extern "C" fn readv(fd: c_int, iov: *const iovec, iovcnt: c_int) -> ssize_t {
        type ReadvFn = unsafe extern "C" fn(c_int, *const iovec, c_int) -> ssize_t;
        let real = match resolve!(readv, ReadvFn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        if fd <= 2 || is_pipe_fd(fd) {
            return real(fd, iov, iovcnt);
        }

        ensure_fd_mapped(fd);
        report_io(fd, "read");
        real(fd, iov, iovcnt)
    }

    /// Intercept `close()` — remove fd from map.
    #[no_mangle]
    pub unsafe extern "C" fn close(fd: c_int) -> c_int {
        type CloseFn = unsafe extern "C" fn(c_int) -> c_int;
        let real = match resolve!(close, CloseFn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        if fd > 2 && !is_pipe_fd(fd) {
            if let Some(_guard) = ReentrancyGuard::enter() {
                if let Ok(mut map) = FD_MAP.lock() {
                    if let Some(resource) = map.remove(fd) {
                        drop(map);
                        log_event("close", &resource, fd);
                    }
                }
            }
        }

        real(fd)
    }

    /// Drop the stale FD_MAP entry for a successful `dup2`/`dup3` target fd:
    /// the target is implicitly closed and may be reused for a different
    /// resource. (No event is logged; this is purely map maintenance to stop
    /// `ensure_fd_mapped` trusting a reused fd's old mapping.)
    fn forget_duped_fd(result: c_int, newfd: c_int) {
        if result >= 0 && newfd > 2 && !is_pipe_fd(newfd) {
            if let Some(_guard) = ReentrancyGuard::enter() {
                if let Ok(mut map) = FD_MAP.lock() {
                    map.remove(newfd);
                }
            }
        }
    }

    /// Intercept `dup2()` — see `forget_duped_fd`.
    #[no_mangle]
    pub unsafe extern "C" fn dup2(oldfd: c_int, newfd: c_int) -> c_int {
        type Dup2Fn = unsafe extern "C" fn(c_int, c_int) -> c_int;
        let real = match resolve!(dup2, Dup2Fn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        let result = real(oldfd, newfd);
        forget_duped_fd(result, newfd);
        result
    }

    /// Intercept `dup3()` — like `dup2` but takes flags; see `forget_duped_fd`.
    #[no_mangle]
    pub unsafe extern "C" fn dup3(oldfd: c_int, newfd: c_int, flags: c_int) -> c_int {
        type Dup3Fn = unsafe extern "C" fn(c_int, c_int, c_int) -> c_int;
        let real = match resolve!(dup3, Dup3Fn) {
            Some(f) => f,
            None => {
                set_errno(libc::ENOSYS);
                return -1;
            }
        };

        let result = real(oldfd, newfd, flags);
        forget_duped_fd(result, newfd);
        result
    }
}

// ===========================================================================
// Intercepted libc functions — macOS (DYLD_INSERT_LIBRARIES + __interpose)
// ===========================================================================
//
// Functions are named `frontrun_<name>` to avoid colliding with the libSystem
// symbols we are interposing.  Forwarding uses raw arm64 syscalls.

#[cfg(target_os = "macos")]
mod macos_intercept {
    use super::*;

    /// Intercept `connect()` — record fd → endpoint mapping.
    #[no_mangle]
    pub unsafe extern "C" fn frontrun_connect(
        fd: c_int,
        addr: *const sockaddr,
        addrlen: socklen_t,
    ) -> c_int {
        let result = raw_syscall::connect(fd, addr, addrlen);
        // Save errno immediately (see Linux connect): callers of non-blocking
        // connect always read errno, and the logging below can clobber it.
        let saved_errno = get_errno();

        if !READY.load(Ordering::Acquire) {
            return result;
        }

        if result == 0 || saved_errno == libc::EINPROGRESS {
            if let Some(_guard) = ReentrancyGuard::enter() {
                if let Some(resource) = sockaddr_to_resource(addr, addrlen) {
                    if let Ok(mut map) = FD_MAP.lock() {
                        map.insert(fd, resource.clone());
                    }
                    log_event("connect", &resource, fd);
                }
            }
        }

        set_errno(saved_errno);
        result
    }

    /// Intercept `send()`.
    #[no_mangle]
    pub unsafe extern "C" fn frontrun_send(
        fd: c_int,
        buf: *const c_void,
        len: size_t,
        flags: c_int,
    ) -> ssize_t {
        if READY.load(Ordering::Acquire) {
            ensure_fd_mapped(fd);
            report_send_buf(fd, buf, len);
        }
        raw_syscall::send(fd, buf, len, flags)
    }

    /// Intercept `sendto()`.
    #[no_mangle]
    pub unsafe extern "C" fn frontrun_sendto(
        fd: c_int,
        buf: *const c_void,
        len: size_t,
        flags: c_int,
        dest_addr: *const sockaddr,
        addrlen: socklen_t,
    ) -> ssize_t {
        if READY.load(Ordering::Acquire) {
            if !dest_addr.is_null() {
                if let Some(_guard) = ReentrancyGuard::enter() {
                    if let Some(resource) = sockaddr_to_resource(dest_addr, addrlen) {
                        if let Ok(mut map) = FD_MAP.lock() {
                            map.insert(fd, resource);
                        }
                    }
                }
            }

            ensure_fd_mapped(fd);
            report_io(fd, "write");
        }
        raw_syscall::sendto(fd, buf, len, flags, dest_addr, addrlen)
    }

    /// Intercept `sendmsg()`.
    #[no_mangle]
    pub unsafe extern "C" fn frontrun_sendmsg(
        fd: c_int,
        msg: *const msghdr,
        flags: c_int,
    ) -> ssize_t {
        if READY.load(Ordering::Acquire) {
            ensure_fd_mapped(fd);
            report_io(fd, "write");
        }
        raw_syscall::sendmsg(fd, msg, flags)
    }

    /// Intercept `write()`.
    #[no_mangle]
    pub unsafe extern "C" fn frontrun_write(
        fd: c_int,
        buf: *const c_void,
        count: size_t,
    ) -> ssize_t {
        if fd > 2 && !is_pipe_fd(fd) && READY.load(Ordering::Acquire) {
            ensure_fd_mapped(fd);
            report_io(fd, "write");
        }
        raw_syscall::write(fd, buf, count)
    }

    /// Intercept `writev()`.
    #[no_mangle]
    pub unsafe extern "C" fn frontrun_writev(
        fd: c_int,
        iov: *const iovec,
        iovcnt: c_int,
    ) -> ssize_t {
        if fd > 2 && !is_pipe_fd(fd) && READY.load(Ordering::Acquire) {
            ensure_fd_mapped(fd);
            report_io(fd, "write");
        }
        raw_syscall::writev(fd, iov, iovcnt)
    }

    /// Intercept `recv()`.
    #[no_mangle]
    pub unsafe extern "C" fn frontrun_recv(
        fd: c_int,
        buf: *mut c_void,
        len: size_t,
        flags: c_int,
    ) -> ssize_t {
        if READY.load(Ordering::Acquire) {
            ensure_fd_mapped(fd);
            report_io(fd, "read");
        }
        raw_syscall::recv(fd, buf, len, flags)
    }

    /// Intercept `recvfrom()`.
    #[no_mangle]
    pub unsafe extern "C" fn frontrun_recvfrom(
        fd: c_int,
        buf: *mut c_void,
        len: size_t,
        flags: c_int,
        src_addr: *mut sockaddr,
        addrlen: *mut socklen_t,
    ) -> ssize_t {
        if !READY.load(Ordering::Acquire) {
            return raw_syscall::recvfrom(fd, buf, len, flags, src_addr, addrlen);
        }

        ensure_fd_mapped(fd);
        report_io(fd, "read");
        let result = raw_syscall::recvfrom(fd, buf, len, flags, src_addr, addrlen);

        if result >= 0 && !src_addr.is_null() && !addrlen.is_null() {
            if let Some(_guard) = ReentrancyGuard::enter() {
                if let Some(resource) =
                    sockaddr_to_resource(src_addr as *const sockaddr, *addrlen)
                {
                    if let Ok(mut map) = FD_MAP.lock() {
                        map.insert(fd, resource);
                    }
                }
            }
        }

        result
    }

    /// Intercept `recvmsg()`.
    #[no_mangle]
    pub unsafe extern "C" fn frontrun_recvmsg(
        fd: c_int,
        msg: *mut msghdr,
        flags: c_int,
    ) -> ssize_t {
        if READY.load(Ordering::Acquire) {
            ensure_fd_mapped(fd);
            report_io(fd, "read");
        }
        raw_syscall::recvmsg(fd, msg, flags)
    }

    /// Intercept `read()`.
    #[no_mangle]
    pub unsafe extern "C" fn frontrun_read(
        fd: c_int,
        buf: *mut c_void,
        count: size_t,
    ) -> ssize_t {
        if fd > 2 && !is_pipe_fd(fd) && READY.load(Ordering::Acquire) {
            ensure_fd_mapped(fd);
            report_io(fd, "read");
        }
        raw_syscall::read(fd, buf, count)
    }

    /// Intercept `readv()`.
    #[no_mangle]
    pub unsafe extern "C" fn frontrun_readv(
        fd: c_int,
        iov: *const iovec,
        iovcnt: c_int,
    ) -> ssize_t {
        if fd > 2 && !is_pipe_fd(fd) && READY.load(Ordering::Acquire) {
            ensure_fd_mapped(fd);
            report_io(fd, "read");
        }
        raw_syscall::readv(fd, iov, iovcnt)
    }

    /// Intercept `close()` — remove fd from map.
    #[no_mangle]
    pub unsafe extern "C" fn frontrun_close(fd: c_int) -> c_int {
        if fd > 2 && !is_pipe_fd(fd) && READY.load(Ordering::Acquire) {
            if let Some(_guard) = ReentrancyGuard::enter() {
                if let Ok(mut map) = FD_MAP.lock() {
                    if let Some(resource) = map.remove(fd) {
                        drop(map);
                        log_event("close", &resource, fd);
                    }
                }
            }
        }

        raw_syscall::close(fd)
    }
}

// ===========================================================================
// macOS dyld interpose table
// ===========================================================================
//
// The `__DATA,__interpose` section tells dyld to replace calls to the
// "original" function (from libSystem) with the "replacement" function
// (our `frontrun_*` interceptor).

#[cfg(target_os = "macos")]
mod interpose {
    use super::macos_intercept::*;

    #[repr(C)]
    struct InterposeEntry {
        replacement: *const (),
        original: *const (),
    }

    // SAFETY: These are immutable function pointers resolved at load time.
    unsafe impl Sync for InterposeEntry {}

    #[link_section = "__DATA,__interpose"]
    #[used]
    static INTERPOSE_TABLE: [InterposeEntry; 12] = [
        InterposeEntry {
            replacement: frontrun_connect as *const (),
            original: libc::connect as *const (),
        },
        InterposeEntry {
            replacement: frontrun_send as *const (),
            original: libc::send as *const (),
        },
        InterposeEntry {
            replacement: frontrun_sendto as *const (),
            original: libc::sendto as *const (),
        },
        InterposeEntry {
            replacement: frontrun_sendmsg as *const (),
            original: libc::sendmsg as *const (),
        },
        InterposeEntry {
            replacement: frontrun_write as *const (),
            original: libc::write as *const (),
        },
        InterposeEntry {
            replacement: frontrun_writev as *const (),
            original: libc::writev as *const (),
        },
        InterposeEntry {
            replacement: frontrun_recv as *const (),
            original: libc::recv as *const (),
        },
        InterposeEntry {
            replacement: frontrun_recvfrom as *const (),
            original: libc::recvfrom as *const (),
        },
        InterposeEntry {
            replacement: frontrun_recvmsg as *const (),
            original: libc::recvmsg as *const (),
        },
        InterposeEntry {
            replacement: frontrun_read as *const (),
            original: libc::read as *const (),
        },
        InterposeEntry {
            replacement: frontrun_readv as *const (),
            original: libc::readv as *const (),
        },
        InterposeEntry {
            replacement: frontrun_close as *const (),
            original: libc::close as *const (),
        },
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_resource_plain() {
        // Strings with no special bytes pass through unchanged.
        assert_eq!(escape_resource("file:/tmp/data.db"), "file:/tmp/data.db");
        assert_eq!(escape_resource("socket:127.0.0.1:5432"), "socket:127.0.0.1:5432");
        assert_eq!(escape_resource(""), "");
    }

    #[test]
    fn test_escape_resource_tab() {
        // Tabs in SQL would split into extra TSV fields without escaping.
        assert_eq!(escape_resource("SELECT\t1"), "SELECT\\t1");
    }

    #[test]
    fn test_escape_resource_newline() {
        // Newlines in multi-line SQL would split into extra event lines.
        assert_eq!(
            escape_resource("UPDATE t\nSET x = 1"),
            "UPDATE t\\nSET x = 1"
        );
    }

    #[test]
    fn test_escape_resource_backslash() {
        // Backslashes must be doubled so unescaping is unambiguous.
        assert_eq!(escape_resource("a\\b"), "a\\\\b");
        // A literal backslash followed by 't' must NOT be confused with a tab.
        assert_eq!(escape_resource("a\\tb"), "a\\\\tb");
    }

    #[test]
    fn test_escape_resource_combined() {
        let sql = "INSERT INTO t\nVALUES (\t'a\\b')";
        assert_eq!(escape_resource(sql), "INSERT INTO t\\nVALUES (\\t'a\\\\b')");
    }

    #[test]
    fn test_escape_resource_preserves_utf8() {
        // Multi-byte UTF-8 must survive intact (continuation bytes >= 0x80
        // never collide with the escaped ASCII characters).
        assert_eq!(escape_resource("café\tμ"), "café\\tμ");
    }

    #[test]
    fn test_unix_socket_resource_path() {
        let path = b"/var/run/postgres/.s.PGSQL.5432\0extra";
        assert_eq!(
            unix_socket_resource(path),
            "socket:unix:/var/run/postgres/.s.PGSQL.5432"
        );
    }

    #[test]
    fn test_unix_socket_resource_empty() {
        assert_eq!(unix_socket_resource(b""), "socket:unix:");
    }

    #[test]
    fn test_unix_socket_resource_abstract_distinct_names() {
        // Two different abstract sockets must produce different resources,
        // not both collapse to a single "socket:unix:abstract".
        let a = unix_socket_resource(b"\0my-service-a");
        let b = unix_socket_resource(b"\0my-service-b");
        assert_eq!(a, "socket:unix:abstract:my-service-a");
        assert_eq!(b, "socket:unix:abstract:my-service-b");
        assert_ne!(a, b);
    }

    #[test]
    fn test_unix_socket_resource_abstract_empty_name() {
        // Leading NUL with no following bytes — abstract with empty name.
        assert_eq!(unix_socket_resource(b"\0"), "socket:unix:abstract:");
    }

    #[test]
    fn test_unix_socket_resource_abstract_binary_name() {
        // Abstract names may contain arbitrary bytes (incl. embedded NUL);
        // non-printable bytes are hex-escaped, keeping distinct names distinct.
        let r = unix_socket_resource(b"\0a\x00\xffb");
        assert_eq!(r, "socket:unix:abstract:a\\x00\\xffb");
    }
}
