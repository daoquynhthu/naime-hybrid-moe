import math
import os
import shutil
import sys
import time

_PROGRESS_CHARS = " ▏▎▍▌▋▊▉█"

R = "\033[0m"
B = "\033[1m"
D = "\033[2m"
K = "\033[90m"
R_ = "\033[91m"
G = "\033[92m"
Y = "\033[93m"
L = "\033[94m"
M = "\033[95m"
C = "\033[96m"


def _is_tty() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _v(val, w=6, d=3):
    if val is None or not math.isfinite(val):
        return f"{'-':>{w}}"
    if abs(val) >= 1000:
        return f"{val:>{w}.1f}"
    if abs(val) >= 100:
        return f"{val:>{w}.2f}"
    if abs(val) >= 10:
        return f"{val:>{w}.3f}"
    return f"{val:>{w}.{d}f}"


def _p(val, w=4):
    if val is None or not math.isfinite(val):
        return f"{'-':>{w}}"
    return f"{val * 100:>{w}.0f}%"


def _bar(fraction: float, width: int) -> str:
    if width < 5:
        return ""
    filled = fraction * width
    full = int(filled)
    partial = int((filled - full) * 8)
    s = "\u2588" * full
    if full < width:
        s += _PROGRESS_CHARS[partial]
        s += " " * (width - full - 1)
    return s


def _dur(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m"


# ---- colored field builders ----


def _lm(v):
    return f"{B}{G}{_v(v)}{R}"


def _ppl(v):
    return f"{L}{_v(v, 7, 1)}{R}"


def _alpha(v):
    return f"{C}{_v(v, 5)}{R}"


def _ent(v):
    return f"{M}{_v(v, 5)}{R}"


def _grad(v):
    return f"{K}{_v(v, 5)}{R}"


def _lr(v):
    return f"{D}{v:.1e}{R}"


def _sp(v):
    return f"{Y}{_v(v, 5)}{R}"


def _best(v):
    return f"{G}{_v(v, 7)}{R}"


def _gap(v):
    return f"{R_}{_v(v, 5)}{R}"


def _conf(v):
    return f"{G}{_v(v, 5)}{R}"


def _cos(v):
    return f"{L}{_v(v, 5)}{R}"


def _rent(v):
    return f"{M}{_v(v, 5)}{R}"


def _pred(v):
    return f"{Y}{_v(v, 7, 4)}{R}"


def _div(v):
    return f"{R_}{_v(v, 5, 4)}{R}"


def _stab(v):
    return f"{D}{_v(v, 5, 4)}{R}"


def _gate(v):
    return f"{C}{_v(v, 5)}{R}"


def _dlt(v):
    return f"{Y}{_v(v, 5, 4)}{R}"


def _went(v):
    return f"{M}{_v(v, 5)}{R}"


def _wmax(v):
    return f"{G}{_v(v, 5)}{R}"


def _mix_a(v):
    return f"{G}{_p(v)}{R}"


def _mix_c(v):
    return f"{Y}{_p(v)}{R}"


def _mix_s(v):
    return f"{L}{_p(v)}{R}"


def _mem(v):
    return f"{C}{_v(v, 5)}{R}"


def _read(v):
    return f"{G}{_v(v, 5)}{R}"


def _nov(v):
    return f"{M}{_v(v, 5, 4)}{R}"


class TrainingProgress:
    def __init__(self, total_steps: int, architecture: str = "naime_v5_world_state_moe"):
        self.total_steps = total_steps
        self.architecture = architecture
        self._tty = _is_tty()
        self._tok_ema = 0.0
        self._ema_d = 0.9
        self._t0 = time.time()
        self._drawn = 0

        if "v6" in architecture:
            self._lines = 5
        elif "v5" in architecture:
            self._lines = 5
        elif "v4" in architecture:
            self._lines = 4
        else:
            self._lines = 2

    def _up(self) -> None:
        if self._drawn > 0:
            sys.stdout.write(f"\033[{self._drawn}A")
        self._drawn = 0

    def _put(self, text: str) -> None:
        sys.stdout.write("\033[G\033[2K" + text + "\n")
        self._drawn += 1

    def _pad(self) -> None:
        while self._drawn < self._lines:
            self._put("")

    def _flush(self) -> None:
        sys.stdout.flush()

    def _header(self, step: int, eta_s: float, extra: str = "") -> str:
        f = min(1.0, step / max(1, self.total_steps))
        w = shutil.get_terminal_size((120, 30)).columns
        extra_chars = len(extra) - extra.count("\033") * 5 if extra else 0
        bw = max(8, w - 55 - extra_chars)
        pct = f * 100
        pct_str = f"{pct:5.1f}%"
        suffix = f"  {extra}" if extra else ""
        return (
            f"{C}{_bar(f, bw)}{R} "
            f"{B}{pct_str:>5s}{R}  "
            f"{step}/{self.total_steps}  "
            f"{Y}ETA {_dur(eta_s)}{R}  "
            f"{D}tok {int(self._tok_ema)}/s{R}{suffix}"
        )

    # ── public API ──────────────────────────────────────────────

    def render_step(self, p: dict) -> None:
        step = p["step"]
        tok = p.get("tok_s", 0)
        self._tok_ema = tok if self._tok_ema == 0 else self._ema_d * self._tok_ema + (1.0 - self._ema_d) * tok
        eta = 0.0
        if self._tok_ema > 0:
            rem = self.total_steps - step
            tps = p.get("tokens_per_step", 3072)
            eta = (rem * tps) / self._tok_ema

        if not self._tty:
            line = (
                f"[{step / max(1, self.total_steps) * 100:3.0f}% {step}/{self.total_steps} "
                f"ETA {_dur(eta)}]  "
                f"lm {p.get('lm', 0):.4f}  ppl {min(9999, p.get('ppl', 0)):.1f}  "
                f"alpha {p.get('alpha_downstream_mean', 0):.3f}  ent {p.get('router_entropy', 0):.3f}"
            )
            if "v6" in self.architecture:
                line += (
                    f"  v6_cos {p.get('v6_slot_cosine', 0):.3f}"
                    f"  v6_ctx {p.get('v6_slot_context_cosine', 0):.3f}"
                    f"  self {p.get('v6_boundary_self', 0):.2f}"
                    f"  world {p.get('v6_boundary_world', 0):.2f}"
                )
            elif "v5" in self.architecture:
                line += f"  v5_cos {p.get('v5_slot_cosine', 0):.3f}  conf {p.get('v5_slot_confidence', 0):.3f}"
            print(line)
            return

        self._up()
        self._put(self._header(step, eta))
        self._put(
            f"{C}tr{R}  "
            f"{B}lm {_lm(p.get('lm', 0))}{R}  "
            f"ppl {_ppl(min(9999.0, p.get('ppl', 0)))}  "
            f"\u03b1 {_alpha(p.get('alpha_downstream_mean', 0))}  "
            f"ent {_ent(p.get('router_entropy', 0))}  "
            f"grad {_grad(p.get('grad_norm', 0))}  "
            f"lr {_lr(p.get('lr', 0))}  "
            f"sp_\u03bb {_sp(p.get('lambda_sparse_effective', 0))}"
        )

        if "v6" in self.architecture:
            self._render_v6(p)
        elif "v5" in self.architecture:
            self._render_v5(p)
        elif "v4" in self.architecture:
            self._render_v4(p)

        self._pad()
        self._flush()

    def _render_v5(self, p: dict) -> None:
        self._put(
            f"{M}v5  slots{R} \u2502 "
            f"conf {_conf(p.get('v5_slot_confidence', 0))}  "
            f"cos {_cos(p.get('v5_slot_cosine', 0))}  "
            f"r_ent {_rent(p.get('v5_slot_read_entropy', 0))}  "
            f"pred {_pred(p.get('v5_state_pred', 0))}  "
            f"div {_div(p.get('v5_slot_diversity', 0))}  "
            f"stab {_stab(p.get('v5_slot_stability', 0))}"
        )
        self._put(
            f"{M}v5  write{R} \u2502 "
            f"gate {_gate(p.get('v5_slot_update_gate', 0))}  "
            f"dlt {_dlt(p.get('v5_slot_delta', 0))}  "
            f"w_ent {_went(p.get('v5_slot_write_entropy', 0))}  "
            f"w_m {_wmax(p.get('v5_slot_write_max', 0))}  "
            f"w_n {_wmax(p.get('v5_slot_write_min', 0))}  "
            f"w_a {_wmax(p.get('v5_slot_write_active', 0))}"
        )
        self._put(
            f"{M}v5  mixer{R} \u2502 "
            f"\u03b1 {_mix_a(p.get('gate_mix_alpha_weight', 0))}  "
            f"clean {_mix_c(p.get('gate_mix_clean_weight', 0))}  "
            f"state {_mix_s(p.get('gate_mix_state_weight', 0))}"
        )

    def _render_v6(self, p: dict) -> None:
        self._put(
            f"{M}v5 world{R} \u2502 "
            f"conf {_conf(p.get('v5_slot_confidence', 0))}  "
            f"cos {_cos(p.get('v5_slot_cosine', 0))}  "
            f"pred {_pred(p.get('v5_state_pred', 0))}  "
            f"w_ent {_went(p.get('v5_slot_write_entropy', 0))}"
        )
        self._put(
            f"{C}v6 self {R} \u2502 "
            f"pred {_pred(p.get('v6_self_pred', 0))}  "
            f"cos {_cos(p.get('v6_slot_cosine', 0))}  "
            f"ctx {_cos(p.get('v6_slot_context_cosine', 0))}  "
            f"dlt {_dlt(p.get('v6_state_delta', 0))}  "
            f"refl {_mem(p.get('v6_reflection_norm', 0))}"
        )
        self._put(
            f"{C}v6 bnd  {R} \u2502 "
            f"ent {_ent(p.get('v6_boundary_entropy', 0))}  "
            f"self {_mix_a(p.get('v6_boundary_self', 0))}  "
            f"world {_mix_c(p.get('v6_boundary_world', 0))}  "
            f"other {_mix_s(p.get('v6_boundary_other', 0))}"
        )

    def _render_v4(self, p: dict) -> None:
        self._put(
            f"{M}v4  state{R} \u2502 "
            f"conf {_conf(p.get('v4_state_confidence', 0))}  "
            f"dlt {_dlt(p.get('v4_state_delta', 0))}  "
            f"mem {_mem(p.get('v4_memory_norm', 0))}  "
            f"read {_read(p.get('v4_memory_read_strength', 0))}  "
            f"nov {_nov(p.get('v4_memory_novelty', 0))}"
        )
        self._put(
            f"{M}v4  mixer{R} \u2502 "
            f"\u03b1 {_mix_a(p.get('gate_mix_alpha_weight', 0))}  "
            f"clean {_mix_c(p.get('gate_mix_clean_weight', 0))}  "
            f"state {_mix_s(p.get('gate_mix_state_weight', 0))}"
        )

    def render_eval(self, p: dict) -> None:
        if not self._tty:
            return
        step = p["step"]
        eta = 0.0
        if self._tok_ema > 0:
            rem = self.total_steps - step
            tps = p.get("tokens_per_step", 3072)
            eta = (rem * tps) / self._tok_ema

        val_lm = p.get("val_lm_loss", 0)
        val_ppl = p.get("val_ppl_val") or (math.exp(min(20.0, val_lm)) if val_lm else 0)
        self._up()
        self._put(self._header(step, eta, extra=f"{B}{M}eval...{R}"))
        self._put(
            f"{B}{M}eval{R}  "
            f"{B}lm {_lm(val_lm)}{R}  "
            f"ppl {_ppl(val_ppl)}  "
            f"\u03b1 {_alpha(p.get('val_alpha_downstream_mean', 0))}  "
            f"ent {_ent(p.get('val_router_entropy', 0))}  "
            f"best {_best(p.get('best_val_lm_loss', 0))}"
        )
        self._pad()
        self._flush()

    def render_save(self, step: int, is_best: bool = False) -> None:
        self._up()
        if self._tty:
            tag = f"{B}{G}save best{R}" if is_best else f"{D}save{R}"
            msg = f"{tag} step {step}"
            self._put(msg)
            self._pad()
        else:
            msg = f"{'save best' if is_best else 'save'} step {step}"
            print(msg)
        self._flush()

    def finalize(self) -> None:
        self._up()
        elapsed = _dur(time.time() - self._t0)
        if self._tty:
            msg = f"{G}done{R}  total time {elapsed}"
            self._put(msg)
            self._pad()
        else:
            msg = f"done  total time {elapsed}"
            print(msg)
        self._flush()
