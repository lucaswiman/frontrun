//! LD_PRELOAD / DYLD_INSERT_LIBRARIES library for frontrun I/O interception.
//!
//! Intercepts libc I/O syscall wrappers (`connect`, `send`, `recv`, `read`,
//! `write`, `close`, etc.) to maintain a file-descriptor → resource mapping.
//! Events are written to a log file specified by `FRONTRUN_IO_LOG`.
//!
//! The dynamic linker resolves symbols in LD_PRELOAD libraries first, so when
//! any library (libpq, libcurl, etc.) calls `send()`, it gets our `send()`.
//! We do bookkeeping, then forward to the real libc function via `dlsym(RTLD_NEXT, ...)`.

use libc::{
    c_char, c_int, c_void, iovec, msghdr, size_t, sockaddr, socklen_t, ssize_t, AF_INET,
    AF_INET6, AF_UNIX, RTLD_NEXT,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Real libc function pointers (resolved lazily via dlsym)
// ---------------------------------------------------------------------------
//
// We use AtomicUsize instead of OnceLock because these function pointers are
// resolved extremely early in process startup (before Rust's standard library
// may be fully initialized). AtomicUsize with relaxed ordering is safe here
// because the stored value (a function pointer from dlsym) is immutable once
// resolved, and the worst case of a data race is resolving it twice.

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
// Per-thread reentrancy guard
// ---------------------------------------------------------------------------
//
// Prevents infinite recursion when our logging code calls write()/close()
// which would trigger our interceptors again.

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

static FD_MAP: std::sync::LazyLock<Mutex<FdMap>> =
    std::sync::LazyLock::new(|| Mutex::new(FdMap::new()));

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

fn log_event(kind: &str, resource: &str, fd: c_int) {
    // Write to log file if configured
    let log_path = match std::env::var("FRONTRUN_IO_LOG") {
        Ok(p) => p,
        Err(_) => return,
    };

    // Use raw libc write to avoid recursion through our interceptors.
    // Open with O_APPEND|O_CREAT|O_WRONLY.
    let path_cstr = match std::ffi::CString::new(log_path) {
        Ok(c) => c,
        Err(_) => return,
    };
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

    let pid = unsafe { libc::getpid() };
    let tid = unsafe { libc::syscall(libc::SYS_gettid) };
    let line = format!("{}\t{}\t{}\t{}\t{}\n", kind, resource, fd, pid, tid);
    let buf = line.as_bytes();

    // Use the real write (resolved), not our interceptor
    type WriteFn = unsafe extern "C" fn(c_int, *const c_void, size_t) -> ssize_t;
    if let Some(real_write) = resolve!(write, WriteFn) {
        unsafe {
            real_write(log_fd, buf.as_ptr() as *const c_void, buf.len());
        }
    }

    // Use the real close
    type CloseFn = unsafe extern "C" fn(c_int) -> c_int;
    if let Some(real_close) = resolve!(close, CloseFn) {
        unsafe {
            real_close(log_fd);
        }
    }
}

// ---------------------------------------------------------------------------
// sockaddr → resource string
// ---------------------------------------------------------------------------

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
            let sun = &*(addr as *const libc::sockaddr_un);
            let path_bytes = &sun.sun_path;
            let len = path_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(path_bytes.len());
            if len == 0 {
                Some("socket:unix:abstract".to_string())
            } else {
                let path = std::str::from_utf8(std::slice::from_raw_parts(
                    path_bytes.as_ptr() as *const u8,
                    len,
                ))
                .unwrap_or("unix:???");
                Some(format!("socket:unix:{}", path))
            }
        } else {
            None
        }
    }
}

/// Try to get the peer address of a connected socket and convert to resource string.
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

// ---------------------------------------------------------------------------
// Helper: check if fd is a regular file and get its path
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Intercepted libc functions
// ---------------------------------------------------------------------------

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
            *libc::__errno_location() = libc::ENOSYS;
            return -1;
        }
    };

    let result = real(fd, addr, addrlen);

    if result == 0 || *libc::__errno_location() == libc::EINPROGRESS {
        if let Some(_guard) = ReentrancyGuard::enter() {
            if let Some(resource) = sockaddr_to_resource(addr, addrlen) {
                if let Ok(mut map) = FD_MAP.lock() {
                    map.insert(fd, resource.clone());
                }
                log_event("connect", &resource, fd);
            }
        }
    }

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
            *libc::__errno_location() = libc::ENOSYS;
            return -1;
        }
    };

    ensure_fd_mapped(fd);
    report_io(fd, "write");
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
    type SendtoFn =
        unsafe extern "C" fn(c_int, *const c_void, size_t, c_int, *const sockaddr, socklen_t) -> ssize_t;
    let real = match resolve!(sendto, SendtoFn) {
        Some(f) => f,
        None => {
            *libc::__errno_location() = libc::ENOSYS;
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
            *libc::__errno_location() = libc::ENOSYS;
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
            *libc::__errno_location() = libc::ENOSYS;
            return -1;
        }
    };

    // Skip stdin/stdout/stderr to avoid infinite recursion in logging
    if fd <= 2 {
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
            *libc::__errno_location() = libc::ENOSYS;
            return -1;
        }
    };

    if fd <= 2 {
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
            *libc::__errno_location() = libc::ENOSYS;
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
            *libc::__errno_location() = libc::ENOSYS;
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
            *libc::__errno_location() = libc::ENOSYS;
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
            *libc::__errno_location() = libc::ENOSYS;
            return -1;
        }
    };

    // Skip stdin/stdout/stderr
    if fd <= 2 {
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
            *libc::__errno_location() = libc::ENOSYS;
            return -1;
        }
    };

    if fd <= 2 {
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
            *libc::__errno_location() = libc::ENOSYS;
            return -1;
        }
    };

    // Remove from map before closing
    if fd > 2 {
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
