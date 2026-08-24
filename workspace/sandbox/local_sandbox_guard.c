#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static char **allowed = NULL;
static size_t allowed_count = 0;
static char *guard_preload = NULL;
static char *guard_prefixes = NULL;
static char *guard_quiet = NULL;
extern char **environ;

static void guard_log(const char *op, const char *path) {
    const char *quiet = getenv("LOCAL_SANDBOX_GUARD_QUIET");
    if (quiet && strcmp(quiet, "1") == 0) {
        return;
    }
    fprintf(stderr, "[local-sandbox-guard] blocked %s outside workspace: %s\n",
            op, path ? path : "(null)");
}

static char *join_path(const char *a, const char *b) {
    char tmp[PATH_MAX * 2];
    if (!a || !*a) {
        snprintf(tmp, sizeof(tmp), "%s", b ? b : "");
    } else if (!b || !*b) {
        snprintf(tmp, sizeof(tmp), "%s", a);
    } else if (a[strlen(a) - 1] == '/') {
        snprintf(tmp, sizeof(tmp), "%s%s", a, b);
    } else {
        snprintf(tmp, sizeof(tmp), "%s/%s", a, b);
    }
    return strdup(tmp);
}

static char *absolute_from_dirfd(int dirfd, const char *path) {
    if (!path) {
        return NULL;
    }
    if (path[0] == '/') {
        return strdup(path);
    }

    char base[PATH_MAX];
    if (dirfd == AT_FDCWD) {
        if (!getcwd(base, sizeof(base))) {
            return strdup(path);
        }
    } else {
        char link_path[64];
        snprintf(link_path, sizeof(link_path), "/proc/self/fd/%d", dirfd);
        ssize_t n = readlink(link_path, base, sizeof(base) - 1);
        if (n < 0) {
            return strdup(path);
        }
        base[n] = '\0';
    }
    return join_path(base, path);
}

static char *canonical_for_check(const char *path) {
    if (!path) {
        return NULL;
    }

    char resolved[PATH_MAX];
    if (realpath(path, resolved)) {
        return strdup(resolved);
    }

    char work[PATH_MAX * 2];
    snprintf(work, sizeof(work), "%s", path);

    char suffix[PATH_MAX * 2] = "";
    while (1) {
        char *slash = strrchr(work, '/');
        if (!slash) {
            break;
        }
        if (slash == work) {
            if (realpath("/", resolved)) {
                return join_path(resolved, suffix[0] == '/' ? suffix + 1 : suffix);
            }
            break;
        }

        char component[PATH_MAX];
        snprintf(component, sizeof(component), "%s", slash + 1);
        *slash = '\0';

        char new_suffix[PATH_MAX * 2];
        if (suffix[0]) {
            snprintf(new_suffix, sizeof(new_suffix), "%s/%s", component, suffix);
        } else {
            snprintf(new_suffix, sizeof(new_suffix), "%s", component);
        }
        snprintf(suffix, sizeof(suffix), "%s", new_suffix);

        if (realpath(work, resolved)) {
            return join_path(resolved, suffix);
        }
    }
    return strdup(path);
}

static bool has_prefix_boundary(const char *path, const char *prefix) {
    size_t n = strlen(prefix);
    return strcmp(path, prefix) == 0 || (strncmp(path, prefix, n) == 0 && path[n] == '/');
}

static bool path_allowed(int dirfd, const char *path) {
    if (!path || allowed_count == 0) {
        return true;
    }
    if (strcmp(path, "/dev/null") == 0 || strcmp(path, "/dev/tty") == 0 ||
        strncmp(path, "/dev/pts/", 9) == 0) {
        return true;
    }
    char *abs_path = absolute_from_dirfd(dirfd, path);
    char *canon = canonical_for_check(abs_path);
    free(abs_path);
    if (!canon) {
        return false;
    }
    for (size_t i = 0; i < allowed_count; i++) {
        if (has_prefix_boundary(canon, allowed[i])) {
            free(canon);
            return true;
        }
    }
    free(canon);
    return false;
}

static bool write_flags(int flags) {
    return (flags & O_WRONLY) || (flags & O_RDWR) || (flags & O_CREAT) ||
           (flags & O_TRUNC) || (flags & O_APPEND)
#ifdef O_TMPFILE
           || ((flags & O_TMPFILE) == O_TMPFILE)
#endif
           ;
}

static bool write_mode(const char *mode) {
    if (!mode) {
        return false;
    }
    return strchr(mode, 'w') || strchr(mode, 'a') || strchr(mode, '+');
}

__attribute__((constructor)) static void init_guard(void) {
    const char *prefixes = getenv("LOCAL_SANDBOX_RW_PREFIXES");
    const char *preload = getenv("LD_PRELOAD");
    const char *quiet = getenv("LOCAL_SANDBOX_GUARD_QUIET");
    guard_preload = preload ? strdup(preload) : NULL;
    guard_prefixes = prefixes ? strdup(prefixes) : NULL;
    guard_quiet = quiet ? strdup(quiet) : NULL;
    if (!prefixes || !*prefixes) {
        return;
    }
    char *copy = strdup(prefixes);
    char *saveptr = NULL;
    for (char *tok = strtok_r(copy, ":", &saveptr); tok; tok = strtok_r(NULL, ":", &saveptr)) {
        char *canon = canonical_for_check(tok);
        if (!canon) {
            continue;
        }
        allowed = realloc(allowed, sizeof(char *) * (allowed_count + 1));
        allowed[allowed_count++] = canon;
    }
    free(copy);
}

static bool env_key_matches(const char *entry, const char *key) {
    size_t n = strlen(key);
    return strncmp(entry, key, n) == 0 && entry[n] == '=';
}

static char *make_env_entry(const char *key, const char *val) {
    size_t n = strlen(key) + 1 + strlen(val) + 1;
    char *entry = malloc(n);
    snprintf(entry, n, "%s=%s", key, val);
    return entry;
}

static char **with_guard_env(char *const envp[]) {
    char *const *src = envp ? envp : environ;
    size_t count = 0;
    while (src && src[count]) {
        count++;
    }

    const char *keys[] = {
        "LD_PRELOAD",
        "LOCAL_SANDBOX_RW_PREFIXES",
        "LOCAL_SANDBOX_GUARD_QUIET",
    };
    const char *vals[] = {
        guard_preload ? guard_preload : "",
        guard_prefixes ? guard_prefixes : "",
        guard_quiet ? guard_quiet : "0",
    };

    char **out = calloc(count + 4, sizeof(char *));
    size_t j = 0;
    bool seen[3] = {false, false, false};
    for (size_t i = 0; i < count; i++) {
        bool replaced = false;
        for (size_t k = 0; k < 3; k++) {
            if (env_key_matches(src[i], keys[k])) {
                out[j++] = make_env_entry(keys[k], vals[k]);
                seen[k] = true;
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            out[j++] = src[i];
        }
    }
    for (size_t k = 0; k < 3; k++) {
        if (!seen[k] && vals[k] && vals[k][0]) {
            out[j++] = make_env_entry(keys[k], vals[k]);
        }
    }
    out[j] = NULL;
    return out;
}

int execve(const char *pathname, char *const argv[], char *const envp[]) {
    int (*real_execve)(const char *, char *const[], char *const[]) = dlsym(RTLD_NEXT, "execve");
    return real_execve(pathname, argv, with_guard_env(envp));
}

int execv(const char *pathname, char *const argv[]) {
    int (*real_execve)(const char *, char *const[], char *const[]) = dlsym(RTLD_NEXT, "execve");
    return real_execve(pathname, argv, with_guard_env(environ));
}

int execvpe(const char *file, char *const argv[], char *const envp[]) {
    int (*real_execvpe)(const char *, char *const[], char *const[]) = dlsym(RTLD_NEXT, "execvpe");
    return real_execvpe(file, argv, with_guard_env(envp));
}

int execvp(const char *file, char *const argv[]) {
    int (*real_execvpe)(const char *, char *const[], char *const[]) = dlsym(RTLD_NEXT, "execvpe");
    return real_execvpe(file, argv, with_guard_env(environ));
}

int open(const char *pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    if (write_flags(flags) && !path_allowed(AT_FDCWD, pathname)) {
        guard_log("open", pathname);
        errno = EACCES;
        return -1;
    }
    int (*real_open)(const char *, int, ...) = dlsym(RTLD_NEXT, "open");
    return (flags & O_CREAT) ? real_open(pathname, flags, mode) : real_open(pathname, flags);
}

int open64(const char *pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    if (write_flags(flags) && !path_allowed(AT_FDCWD, pathname)) {
        guard_log("open64", pathname);
        errno = EACCES;
        return -1;
    }
    int (*real_open64)(const char *, int, ...) = dlsym(RTLD_NEXT, "open64");
    return (flags & O_CREAT) ? real_open64(pathname, flags, mode) : real_open64(pathname, flags);
}

int openat(int dirfd, const char *pathname, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    if (write_flags(flags) && !path_allowed(dirfd, pathname)) {
        guard_log("openat", pathname);
        errno = EACCES;
        return -1;
    }
    int (*real_openat)(int, const char *, int, ...) = dlsym(RTLD_NEXT, "openat");
    return (flags & O_CREAT) ? real_openat(dirfd, pathname, flags, mode) : real_openat(dirfd, pathname, flags);
}

int creat(const char *pathname, mode_t mode) {
    if (!path_allowed(AT_FDCWD, pathname)) {
        guard_log("creat", pathname);
        errno = EACCES;
        return -1;
    }
    int (*real_creat)(const char *, mode_t) = dlsym(RTLD_NEXT, "creat");
    return real_creat(pathname, mode);
}

FILE *fopen(const char *pathname, const char *mode) {
    if (write_mode(mode) && !path_allowed(AT_FDCWD, pathname)) {
        guard_log("fopen", pathname);
        errno = EACCES;
        return NULL;
    }
    FILE *(*real_fopen)(const char *, const char *) = dlsym(RTLD_NEXT, "fopen");
    return real_fopen(pathname, mode);
}

int mkdir(const char *pathname, mode_t mode) {
    if (!path_allowed(AT_FDCWD, pathname)) {
        guard_log("mkdir", pathname);
        errno = EACCES;
        return -1;
    }
    int (*real_mkdir)(const char *, mode_t) = dlsym(RTLD_NEXT, "mkdir");
    return real_mkdir(pathname, mode);
}

int mkdirat(int dirfd, const char *pathname, mode_t mode) {
    if (!path_allowed(dirfd, pathname)) {
        guard_log("mkdirat", pathname);
        errno = EACCES;
        return -1;
    }
    int (*real_mkdirat)(int, const char *, mode_t) = dlsym(RTLD_NEXT, "mkdirat");
    return real_mkdirat(dirfd, pathname, mode);
}

int unlink(const char *pathname) {
    if (!path_allowed(AT_FDCWD, pathname)) {
        guard_log("unlink", pathname);
        errno = EACCES;
        return -1;
    }
    int (*real_unlink)(const char *) = dlsym(RTLD_NEXT, "unlink");
    return real_unlink(pathname);
}

int unlinkat(int dirfd, const char *pathname, int flags) {
    if (!path_allowed(dirfd, pathname)) {
        guard_log("unlinkat", pathname);
        errno = EACCES;
        return -1;
    }
    int (*real_unlinkat)(int, const char *, int) = dlsym(RTLD_NEXT, "unlinkat");
    return real_unlinkat(dirfd, pathname, flags);
}

int rmdir(const char *pathname) {
    if (!path_allowed(AT_FDCWD, pathname)) {
        guard_log("rmdir", pathname);
        errno = EACCES;
        return -1;
    }
    int (*real_rmdir)(const char *) = dlsym(RTLD_NEXT, "rmdir");
    return real_rmdir(pathname);
}

int rename(const char *oldpath, const char *newpath) {
    if (!path_allowed(AT_FDCWD, oldpath) || !path_allowed(AT_FDCWD, newpath)) {
        guard_log("rename", newpath);
        errno = EACCES;
        return -1;
    }
    int (*real_rename)(const char *, const char *) = dlsym(RTLD_NEXT, "rename");
    return real_rename(oldpath, newpath);
}

int renameat(int olddirfd, const char *oldpath, int newdirfd, const char *newpath) {
    if (!path_allowed(olddirfd, oldpath) || !path_allowed(newdirfd, newpath)) {
        guard_log("renameat", newpath);
        errno = EACCES;
        return -1;
    }
    int (*real_renameat)(int, const char *, int, const char *) = dlsym(RTLD_NEXT, "renameat");
    return real_renameat(olddirfd, oldpath, newdirfd, newpath);
}

int renameat2(int olddirfd, const char *oldpath, int newdirfd, const char *newpath, unsigned int flags) {
    if (!path_allowed(olddirfd, oldpath) || !path_allowed(newdirfd, newpath)) {
        guard_log("renameat2", newpath);
        errno = EACCES;
        return -1;
    }
    int (*real_renameat2)(int, const char *, int, const char *, unsigned int) = dlsym(RTLD_NEXT, "renameat2");
    if (!real_renameat2) {
        errno = ENOSYS;
        return -1;
    }
    return real_renameat2(olddirfd, oldpath, newdirfd, newpath, flags);
}

int remove(const char *pathname) {
    if (!path_allowed(AT_FDCWD, pathname)) {
        guard_log("remove", pathname);
        errno = EACCES;
        return -1;
    }
    int (*real_remove)(const char *) = dlsym(RTLD_NEXT, "remove");
    return real_remove(pathname);
}

int truncate(const char *path, off_t length) {
    if (!path_allowed(AT_FDCWD, path)) {
        guard_log("truncate", path);
        errno = EACCES;
        return -1;
    }
    int (*real_truncate)(const char *, off_t) = dlsym(RTLD_NEXT, "truncate");
    return real_truncate(path, length);
}
