/*
 * frontrun_xproc_sched — isolated LD_PRELOAD scheduling shim (Phase 4 PoC).
 *
 * Demonstrates the core Phase 4 thesis of ideas/cross_process_exploration.md:
 * an *unmodified* worker (no frontrun import, any language) can be scheduled at
 * external-access granularity by blocking inside the C-level socket hooks until
 * a coordinator grants the turn. The worker needs no Python patches at all.
 *
 * This is deliberately separate from the production crates/io library (which
 * the whole test suite preloads) so it can never destabilise it: it is loaded
 * only by the Phase 4 e2e's own worker subprocesses, and is fully inert unless
 * FRONTRUN_XPROC_SCHED_PATH is set.
 *
 * Scope: schedules at the granularity of one send()/write() on a connected TCP
 * socket. It does NOT parse the SQL wire protocol to classify statements or
 * transaction boundaries — that is the separately-deferred wire-parsing roadmap
 * item. Grant/abort is a single byte; the coordinator owns all policy.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#define MAX_FDS 65536
#define REQ_HELLO 0x48 /* 'H' */
#define REQ_SEND 0x53  /* 'S' */
#define REPLY_GRANT 0x01
#define REPLY_ABORT 0x00

static ssize_t (*real_send)(int, const void *, size_t, int) = NULL;
static ssize_t (*real_write)(int, const void *, size_t) = NULL;
static int (*real_connect)(int, const struct sockaddr *, socklen_t) = NULL;
static int (*real_close)(int) = NULL;

static int g_enabled = 0;   /* set once the coordinator socket is connected */
static int g_coord_fd = -1; /* our control socket to the coordinator */
static unsigned char g_worker_id = 0;
static char g_tracked[MAX_FDS]; /* g_tracked[fd] == 1 => scheduled TCP socket */
static __thread int g_in_hook = 0;

static void resolve_reals(void) {
    if (!real_send) real_send = dlsym(RTLD_NEXT, "send");
    if (!real_write) real_write = dlsym(RTLD_NEXT, "write");
    if (!real_connect) real_connect = dlsym(RTLD_NEXT, "connect");
    if (!real_close) real_close = dlsym(RTLD_NEXT, "close");
}

__attribute__((constructor)) static void init_shim(void) {
    resolve_reals();
    const char *path = getenv("FRONTRUN_XPROC_SCHED_PATH");
    if (!path || !path[0]) return;
    const char *wid = getenv("FRONTRUN_XPROC_WORKER_ID");
    g_worker_id = (unsigned char)(wid ? atoi(wid) : 0);

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (real_connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return;
    }
    /* Announce ourselves: [HELLO][worker_id]. */
    unsigned char hello[2] = {REQ_HELLO, g_worker_id};
    if (real_send(fd, hello, 2, 0) != 2) {
        close(fd);
        return;
    }
    g_coord_fd = fd;
    g_enabled = 1;
}

/* Send one scheduling request for this send() and block for the grant byte.
 * Returns 1 to proceed, 0 to proceed-without-scheduling (coordinator gone). */
static int await_grant(void) {
    unsigned char req[2] = {REQ_SEND, g_worker_id};
    if (real_send(g_coord_fd, req, 2, 0) != 2) {
        g_enabled = 0; /* coordinator gone: stop scheduling, run free */
        return 1;
    }
    unsigned char reply = REPLY_GRANT;
    ssize_t n = recv(g_coord_fd, &reply, 1, 0);
    if (n != 1) {
        g_enabled = 0;
        return 1;
    }
    return reply == REPLY_GRANT ? 1 : 1; /* abort still lets the worker finish */
}

static void schedule_point(int fd) {
    if (!g_enabled || g_in_hook) return;
    if (fd < 0 || fd >= MAX_FDS || !g_tracked[fd]) return;
    g_in_hook = 1;
    await_grant();
    g_in_hook = 0;
}

int connect(int fd, const struct sockaddr *addr, socklen_t len) {
    resolve_reals();
    int rc = real_connect(fd, addr, len);
    if (g_enabled && rc == 0 && addr && fd >= 0 && fd < MAX_FDS) {
        if (addr->sa_family == AF_INET || addr->sa_family == AF_INET6) g_tracked[fd] = 1;
    }
    return rc;
}

ssize_t send(int fd, const void *buf, size_t len, int flags) {
    resolve_reals();
    schedule_point(fd);
    return real_send(fd, buf, len, flags);
}

ssize_t write(int fd, const void *buf, size_t len) {
    resolve_reals();
    schedule_point(fd);
    return real_write(fd, buf, len);
}

int close(int fd) {
    resolve_reals();
    if (fd >= 0 && fd < MAX_FDS) g_tracked[fd] = 0;
    return real_close(fd);
}
