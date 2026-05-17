/*
 * naime_guardian.c  -  Native Windows shutdown coordinator for NAIME training.
 *
 * Architecture:
 *   launch_train_detached.py
 *     -> naime_guardian.exe (this, native, ~20KB, no Python dependency)
 *          -> pythonw.exe -m naime_hybrid.training.train ... (child process)
 *
 * The coordinator registers native Win32 shutdown/control handlers. When a
 * stop event arrives it writes the trainer's STOP file and waits for the
 * Python trainer to checkpoint and exit on its own. It never force-kills the
 * trainer; if Windows or an administrator hard-kills processes, no user-space
 * code can guarantee another write opportunity.
 *
 * Compile (MinGW-w64):
 *   gcc -O2 -s -o naime_guardian.exe naime_guardian.c -lole32
 *
 * Usage:
 *   naime_guardian.exe --repo <path> --trainer-python <path> --run-dir <path>
 *       [--max-restarts N] [--restart-delay N] [--] <trainer args...>
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <process.h>
#include <tlhelp32.h>

/* ---- configuration ---- */
typedef struct {
    wchar_t repo[MAX_PATH];
    wchar_t trainer_python[MAX_PATH];
    wchar_t run_dir[MAX_PATH];
    int     max_restarts;
    int     restart_delay;
    wchar_t trainer_args[4096];
} Config;

typedef struct {
    volatile LONG stop_requested;
    wchar_t stop_reason[128];
    HANDLE  trainer_process;
    DWORD   trainer_pid;
    int     restart_count;
    HANDLE  stop_file_event;
    Config  config;
    CRITICAL_SECTION cs;
} State;

static State g_state = {0};

/* ---- helpers ---- */
static void wcscat_safe(wchar_t *dst, size_t dst_size, const wchar_t *src) {
    size_t len = wcslen(dst);
    size_t src_len = wcslen(src);
    if (len + src_len < dst_size) {
        wcscpy_s(dst + len, dst_size - len, src);
    }
}

static void write_pid_file(const wchar_t *path, DWORD pid) {
    FILE *f = _wfopen(path, L"w, ccs=UTF-8");
    if (f) { fwprintf(f, L"%lu", pid); fclose(f); }
}

static void write_stop_file(const wchar_t *run_dir) {
    wchar_t stop_path[MAX_PATH];
    wcscpy_s(stop_path, MAX_PATH, run_dir);
    wcscat_safe(stop_path, MAX_PATH, L"\\STOP");
    FILE *f = _wfopen(stop_path, L"w, ccs=UTF-8");
    if (f) { fwprintf(f, L"guardian"); fclose(f); }
}

static void request_stop(const wchar_t *reason) {
    if (InterlockedCompareExchange(&g_state.stop_requested, 1, 0) == 0) {
        EnterCriticalSection(&g_state.cs);
        wcscpy_s(g_state.stop_reason, 128, reason);
        LeaveCriticalSection(&g_state.cs);
        write_stop_file(g_state.config.run_dir);
    }
}

/* ---- signal/interrupt handling ---- */

static BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    const wchar_t *reason = L"";
    switch (ctrl_type) {
        case CTRL_C_EVENT:        reason = L"CTRL_C_EVENT"; break;
        case CTRL_BREAK_EVENT:    reason = L"CTRL_BREAK_EVENT"; break;
        case CTRL_CLOSE_EVENT:    reason = L"CTRL_CLOSE_EVENT"; break;
        case CTRL_LOGOFF_EVENT:   reason = L"CTRL_LOGOFF_EVENT"; break;
        case CTRL_SHUTDOWN_EVENT: reason = L"CTRL_SHUTDOWN_EVENT"; break;
        default: return FALSE;
    }
    request_stop(reason);
    return TRUE;
}

/* ---- trainer process management ---- */

static int start_trainer(Config *cfg) {
    if (cfg->trainer_args[0] == 0) {
        fwprintf(stderr, L"guardian: no trainer arguments\n");
        return -1;
    }

    wchar_t cmdline[8192];
    wcscpy_s(cmdline, 8192, L"\"");
    wcscat_safe(cmdline, 8192, cfg->trainer_python);
    wcscat_safe(cmdline, 8192, L"\" -m naime_hybrid.training.train ");
    wcscat_safe(cmdline, 8192, cfg->trainer_args);

    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi = {0};

    /* redirect trainer's stdout/stderr to files for diagnostics */
    wchar_t trainer_out_path[MAX_PATH], trainer_err_path[MAX_PATH];
    wcscpy_s(trainer_out_path, MAX_PATH, cfg->run_dir);
    wcscpy_s(trainer_err_path, MAX_PATH, cfg->run_dir);
    wcscat_safe(trainer_out_path, MAX_PATH, L"\\trainer.stdout.log");
    wcscat_safe(trainer_err_path, MAX_PATH, L"\\trainer.stderr.log");
    HANDLE h_out = CreateFileW(trainer_out_path, GENERIC_WRITE, FILE_SHARE_READ,
                               &sa, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    HANDLE h_err = CreateFileW(trainer_err_path, GENERIC_WRITE, FILE_SHARE_READ,
                               &sa, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h_out != INVALID_HANDLE_VALUE) SetFilePointer(h_out, 0, NULL, FILE_END);
    if (h_err != INVALID_HANDLE_VALUE) SetFilePointer(h_err, 0, NULL, FILE_END);

    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = INVALID_HANDLE_VALUE;
    si.hStdOutput = h_out != INVALID_HANDLE_VALUE ? h_out : INVALID_HANDLE_VALUE;
    si.hStdError  = h_err != INVALID_HANDLE_VALUE ? h_err : INVALID_HANDLE_VALUE;

    DWORD flags = CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP;

    BOOL ok = CreateProcessW(
        cfg->trainer_python, cmdline, NULL, NULL, TRUE,
        flags, NULL, cfg->repo, &si, &pi
    );
    if (!ok) {
        fwprintf(stderr, L"guardian: CreateProcess failed (error %lu)\n", GetLastError());
        return -1;
    }
    CloseHandle(pi.hThread);
    g_state.trainer_process = pi.hProcess;
    g_state.trainer_pid     = pi.dwProcessId;

    /* write pid files */
    wchar_t daemon_pid_path[MAX_PATH], trainer_pid_path[MAX_PATH];
    wcscpy_s(daemon_pid_path,  MAX_PATH, cfg->run_dir);
    wcscpy_s(trainer_pid_path, MAX_PATH, cfg->run_dir);
    wcscat_safe(daemon_pid_path,  MAX_PATH, L"\\daemon.pid");
    wcscat_safe(trainer_pid_path, MAX_PATH, L"\\trainer.pid");
    write_pid_file(daemon_pid_path,  GetCurrentProcessId());
    write_pid_file(trainer_pid_path, pi.dwProcessId);

    fwprintf(stdout, L"guardian: trainer started pid=%lu\n", pi.dwProcessId);
    return 0;
}

static void poll_stop_file(void) {
    wchar_t stop_path[MAX_PATH];
    wcscpy_s(stop_path, MAX_PATH, g_state.config.run_dir);
    wcscat_safe(stop_path, MAX_PATH, L"\\STOP");

    while (!g_state.stop_requested) {
        if (GetFileAttributesW(stop_path) != INVALID_FILE_ATTRIBUTES) {
            request_stop(L"STOP_FILE");
            return;
        }
        Sleep(2000);
    }
}

static DWORD WINAPI stop_file_poll_thread(LPVOID param) {
    (void)param;
    poll_stop_file();
    return 0;
}

/* ---- main ---- */

static void parse_args(int argc, wchar_t *argv[], Config *cfg) {
    memset(cfg, 0, sizeof(Config));
    int i, after_sep = 0, trainer_idx = 0;
    for (i = 1; i < argc; i++) {
        if (wcscmp(argv[i], L"--") == 0) {
            after_sep = 1;
            continue;
        }
        if (after_sep) {
            /* trainer args: concatenate with space */
            if (trainer_idx > 0) wcscat_safe(cfg->trainer_args, 4096, L" ");
            wcscat_safe(cfg->trainer_args, 4096, argv[i]);
            trainer_idx++;
            continue;
        }
        if (i + 1 < argc) {
            if (wcscmp(argv[i], L"--repo") == 0) {
                wcscpy_s(cfg->repo, MAX_PATH, argv[++i]);
            } else if (wcscmp(argv[i], L"--trainer-python") == 0) {
                wcscpy_s(cfg->trainer_python, MAX_PATH, argv[++i]);
            } else if (wcscmp(argv[i], L"--run-dir") == 0) {
                wcscpy_s(cfg->run_dir, MAX_PATH, argv[++i]);
            } else if (wcscmp(argv[i], L"--max-restarts") == 0) {
                cfg->max_restarts = _wtoi(argv[++i]);
            } else if (wcscmp(argv[i], L"--restart-delay") == 0) {
                cfg->restart_delay = _wtoi(argv[++i]);
            } else {
                /* skip unknown */
                i++;
            }
        }
    }
    if (cfg->restart_delay <= 0) cfg->restart_delay = 10;
}

int wmain(int argc, wchar_t *argv[]) {
    Config cfg;
    parse_args(argc, argv, &cfg);

    if (cfg.repo[0] == 0 || cfg.trainer_python[0] == 0 || cfg.run_dir[0] == 0) {
        fwprintf(stderr, L"usage: naime_guardian --repo <path> --trainer-python <path> --run-dir <path> [--max-restarts N] [--restart-delay N] [--] <trainer args>\n");
        return 1;
    }

    /* ensure run_dir exists */
    CreateDirectoryW(cfg.run_dir, NULL);

    /* init state */
    memset(&g_state, 0, sizeof(g_state));
    g_state.config = cfg;
    g_state.trainer_process = INVALID_HANDLE_VALUE;
    InitializeCriticalSection(&g_state.cs);

    /* unbuffer stdout/stderr so log lines appear immediately in the launcher file */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    /* install native interrupt handlers */
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    CreateThread(NULL, 0, stop_file_poll_thread, NULL, 0, NULL);

    fwprintf(stdout, L"guardian: started pid=%lu run_dir=%ls\n",
             GetCurrentProcessId(), cfg.run_dir);

    /* main loop: start and monitor trainer */
    int exit_code = 0;
    while (!g_state.stop_requested) {
        if (g_state.trainer_process == INVALID_HANDLE_VALUE) {
            /* check existing trainer.pid (adopt orphan) */
            wchar_t pid_path[MAX_PATH];
            wcscpy_s(pid_path, MAX_PATH, cfg.run_dir);
            wcscat_safe(pid_path, MAX_PATH, L"\\trainer.pid");
            FILE *f = _wfopen(pid_path, L"r, ccs=UTF-8");
            if (f) {
                DWORD existing_pid;
                if (fwscanf_s(f, L"%lu", &existing_pid) == 1) {
                    HANDLE h = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                                           FALSE, existing_pid);
                    if (h) {
                        DWORD wait = WaitForSingleObject(h, 0);
                        if (wait == WAIT_TIMEOUT) {
                            g_state.trainer_process = h;
                            g_state.trainer_pid = existing_pid;
                            fwprintf(stdout, L"guardian: adopted existing trainer pid=%lu\n", existing_pid);
                        } else {
                            CloseHandle(h);
                        }
                    }
                }
                fclose(f);
            }
            if (g_state.trainer_process == INVALID_HANDLE_VALUE) {
                if (start_trainer(&cfg) != 0) {
                    fwprintf(stderr, L"guardian: failed to start trainer\n");
                    exit_code = 1;
                    break;
                }
            }
        }

        /* wait for trainer to exit or stop request */
        HANDLE wait_handles[2];
        wait_handles[0] = g_state.trainer_process;
        wait_handles[1] = NULL; /* no second handle, just the process */
        DWORD wait = WaitForSingleObject(g_state.trainer_process, 2000);

        if (wait == WAIT_OBJECT_0) {
            /* trainer exited */
            DWORD code = 0;
            GetExitCodeProcess(g_state.trainer_process, &code);
            CloseHandle(g_state.trainer_process);
            g_state.trainer_process = INVALID_HANDLE_VALUE;
            g_state.trainer_pid = 0;

            /* clear trainer.pid */
            wchar_t pid_path[MAX_PATH];
            wcscpy_s(pid_path, MAX_PATH, cfg.run_dir);
            wcscat_safe(pid_path, MAX_PATH, L"\\trainer.pid");
            DeleteFileW(pid_path);

            if (g_state.stop_requested) {
                fwprintf(stdout, L"guardian: trainer stopped as requested (code=%lu reason=%s)\n",
                         code, g_state.stop_reason);
                break;
            }
            if (code == 0) {
                fwprintf(stdout, L"guardian: trainer exited cleanly\n");
                break;
            }
            if (cfg.max_restarts >= 0 && g_state.restart_count >= cfg.max_restarts) {
                fwprintf(stdout, L"guardian: max restarts (%d) reached\n", cfg.max_restarts);
                exit_code = (int)code;
                break;
            }
            if (cfg.max_restarts > 0) {
                g_state.restart_count++;
                fwprintf(stdout, L"guardian: trainer crashed (code=%lu) restarting in %ds (%d/%d)\n",
                         code, cfg.restart_delay, g_state.restart_count, cfg.max_restarts);
                Sleep(cfg.restart_delay * 1000);
            } else {
                fwprintf(stdout, L"guardian: trainer exited with code=%lu; no restart configured\n", code);
                exit_code = (int)code;
                break;
            }
        }
        /* else timeout: loop back, check stop_requested */
    }

    /* cleanup */
    if (g_state.trainer_process != INVALID_HANDLE_VALUE &&
        g_state.trainer_process != NULL) {
        if (WaitForSingleObject(g_state.trainer_process, 0) != WAIT_OBJECT_0) {
            /* still alive: write STOP and wait; never force-kill the trainer. */
            write_stop_file(cfg.run_dir);
            DWORD wait = WaitForSingleObject(g_state.trainer_process, 300000);
            if (wait == WAIT_TIMEOUT) {
                fwprintf(stderr, L"guardian: trainer still alive after graceful-stop wait; leaving it to finish checkpointing\n");
            }
        }
        CloseHandle(g_state.trainer_process);
    }

    /* clean up pid files */
    wchar_t dpid[MAX_PATH];
    wcscpy_s(dpid, MAX_PATH, cfg.run_dir);
    wcscat_safe(dpid, MAX_PATH, L"\\daemon.pid");
    DeleteFileW(dpid);
    wchar_t tpid[MAX_PATH];
    wcscpy_s(tpid, MAX_PATH, cfg.run_dir);
    wcscat_safe(tpid, MAX_PATH, L"\\trainer.pid");
    DeleteFileW(tpid);

    DeleteCriticalSection(&g_state.cs);
    fwprintf(stdout, L"guardian: exited\n");
    return exit_code;
}
