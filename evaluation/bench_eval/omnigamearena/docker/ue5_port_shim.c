/* ===========================================================================
 * LD_PRELOAD shim that moves UE5's RemoteInput listener off its hardcoded port.
 *
 * OmniGameArena's RemoteInput server binds 12345 with no way to configure it, so
 * running several instances on one node used to require a network namespace per
 * instance -- and a k8s pod has no CAP_NET_ADMIN, so `ip netns add` fails. The
 * shipping binary imports bind@GLIBC_2.2.5, so interposing bind() achieves the
 * same isolation with no capabilities at all: instance i binds 12345+i and the
 * driver connects there.
 *
 * PvP/Coop maps serve their second player from a second listener on 12346, so a
 * span wider than one port can be moved as a block: 12345+k -> override+k. Only
 * ports inside that block are rewritten; every other bind (engine telemetry,
 * ephemeral sockets) passes through untouched.
 *
 *   UE5_PORT_OVERRIDE  port to bind instead        (required; no-op if unset)
 *   UE5_PORT_FROM      first port to rewrite       (default 12345)
 *   UE5_PORT_SPAN      how many consecutive ports  (default 1; 2 for PvP/Coop)
 *   UE5_PORT_SHIM_QUIET set to 1 to silence the log line
 *
 * build: gcc -shared -fPIC -O2 -o ue5_port_shim.so ue5_port_shim.c -ldl
 * ===========================================================================*/
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

typedef int (*bind_fn)(int, const struct sockaddr *, socklen_t);

static bind_fn real_bind;
static unsigned short from_port, to_port, span;
static int initialised;

static void shim_init(void)
{
    const char *to = getenv("UE5_PORT_OVERRIDE");
    const char *from = getenv("UE5_PORT_FROM");
    const char *n = getenv("UE5_PORT_SPAN");
    real_bind = (bind_fn)dlsym(RTLD_NEXT, "bind");
    to_port = to ? (unsigned short)atoi(to) : 0;
    from_port = (from && *from) ? (unsigned short)atoi(from) : 12345;
    span = (n && *n) ? (unsigned short)atoi(n) : 1;
    if (span < 1)
        span = 1;
    initialised = 1;
}

int bind(int fd, const struct sockaddr *addr, socklen_t len)
{
    if (!initialised)
        shim_init();
    if (!real_bind) {
        errno = ENOSYS;
        return -1;
    }
    if (!to_port || !addr)
        return real_bind(fd, addr, len);

    /* The caller owns addr and it is const, so rewrite a copy. */
    struct sockaddr_storage copy;
    unsigned short port = 0;

    if (addr->sa_family == AF_INET && len >= (socklen_t)sizeof(struct sockaddr_in))
        port = ntohs(((const struct sockaddr_in *)addr)->sin_port);
    else if (addr->sa_family == AF_INET6 && len >= (socklen_t)sizeof(struct sockaddr_in6))
        port = ntohs(((const struct sockaddr_in6 *)addr)->sin6_port);
    else
        return real_bind(fd, addr, len);

    if (port < from_port || port >= from_port + span)
        return real_bind(fd, addr, len);

    unsigned short mapped = (unsigned short)(to_port + (port - from_port));

    if (len > (socklen_t)sizeof(copy))
        return real_bind(fd, addr, len);
    memcpy(&copy, addr, (size_t)len);
    if (addr->sa_family == AF_INET)
        ((struct sockaddr_in *)&copy)->sin_port = htons(mapped);
    else
        ((struct sockaddr_in6 *)&copy)->sin6_port = htons(mapped);

    if (!getenv("UE5_PORT_SHIM_QUIET"))
        fprintf(stderr, "[port-shim] bind %u -> %u\n", port, mapped);
    return real_bind(fd, (const struct sockaddr *)&copy, len);
}
