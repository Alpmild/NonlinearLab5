import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from Euler import euler
from EulerKromer import euler_kromer
from EMP import emp
from KD import kd
from RK4 import rk4

# ----------------------------- параметры ------------------------------------
h = 0.0001
ct = 2000.0
CUT = 0.8  # доля начала траектории, отбрасываемая как переходный процесс
IC_RANGE = (0.0, 1.0)
SEED = 12345
start_params = (1.0, 1.0, 1.0)
LAM_TOL = 1e-3  # |lambda| <= LAM_TOL считаем нулевым показателем
PLOT_POINTS = 3000
ERR_FLOOR = 1e-18
save_folder = "t2_sync_plots"
os.makedirs(save_folder, exist_ok=True)

# ----------------------------- подбор K -------------------------------------
K_MAX = 20.0        # верхняя граница поиска K
N_COARSE = 41       # число точек грубой сетки (от 0 до K_MAX)
N_REFINE = 3        # число раундов уточнения
N_FINE = 21         # число точек сетки в каждом раунде уточнения
K_MARGIN = 1.15     # итоговое K = K_крит * K_MARGIN (запас от границы)
CT_SCAN = 500.0     # время расчёта при подборе (короче, чем ct)
CUT_SCAN = 0.5      # доля переходного процесса при подборе

VAR_NAMES = ["x", "y", "z"]
VAR_MAP = {"x": 0, "y": 1, "z": 2}
ACTIVE_INDICES = [VAR_MAP[name] for name in VAR_NAMES]


@njit(cache=True)
def f(u, a1, a2, a3):
    x = u[0]
    y = u[1]
    z = u[2]
    return np.array([
        a1 * y * z,
        x - a2 * y,
        a3 - x * y
    ])


methods = [
    # ("Euler", njit(euler)),
    # ("Euler-Cromer", njit(euler_kromer)),
    # ("EMP", njit(emp)),
    ("VSKD", njit(kd)),
    # ("RK4", njit(rk4)),
]


@njit(cache=True)
def jac3(u, a1, a2, a3):
    J = np.zeros((3, 3))
    J[0, 1] = a1 * u[2]
    J[0, 2] = a1 * u[1]
    J[1, 0] = 1.0
    J[1, 1] = -a2
    J[2, 0] = -u[1]
    J[2, 1] = -u[0]
    return J


@njit(cache=True)
def mm3(A, B):
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            C[i, j] = A[i, 0] * B[0, j] + A[i, 1] * B[1, j] + A[i, 2] * B[2, j]
    return C


@njit(cache=True)
def gram_schmidt(Q, sums):
    for c in range(3):
        for r in range(c):
            dot = 0.0
            for i in range(3):
                dot += Q[i, r] * Q[i, c]
            for i in range(3):
                Q[i, c] -= dot * Q[i, r]
        nrm = 0.0
        for i in range(3):
            nrm += Q[i, c] ** 2
        nrm = np.sqrt(nrm)
        for i in range(3):
            Q[i, c] /= nrm
        sums[c] += np.log(nrm)


# ============================================================================
# ПОДБОР K: поперечные показатели Ляпунова сразу для массива значений eps.
# Синхронная траектория от eps не зависит, поэтому считается один раз, а
# уравнения в вариациях ведутся параллельно для каждого eps.
# Возвращает массив (len(eps_arr), 3).
# ============================================================================
@njit
def lyap_multi(method, f, k, eps_arr, u0, h_val, n, start, a1, a2, a3):
    m = len(eps_arr)
    n_keep = n - start

    u = u0.copy()
    for i in range(start):
        _, s = method(f, u, 0.0, h_val, h_val, a1, a2, a3)
        u = s[1].copy()

    Qs = np.zeros((m, 3, 3))
    for e in range(m):
        Qs[e] = np.eye(3)
    sums = np.zeros((m, 3))

    for j in range(n_keep):
        u_old = u
        _, s = method(f, u_old, 0.0, h_val, h_val, a1, a2, a3)
        u = s[1].copy()
        f0 = f(u_old, a1, a2, a3)
        f1 = f(u, a1, a2, a3)
        u_mid = 0.5 * (u_old + u) + h_val * (f0 - f1) / 8.0

        J0b = jac3(u_old, a1, a2, a3)
        Jmb = jac3(u_mid, a1, a2, a3)
        J1b = jac3(u, a1, a2, a3)

        for e in range(m):
            J0 = J0b.copy()
            Jm = Jmb.copy()
            J1 = J1b.copy()
            J0[k, k] -= 2.0 * eps_arr[e]
            Jm[k, k] -= 2.0 * eps_arr[e]
            J1[k, k] -= 2.0 * eps_arr[e]

            Q = Qs[e].copy()
            k1 = mm3(J0, Q)
            k2 = mm3(Jm, Q + 0.5 * h_val * k1)
            k3 = mm3(Jm, Q + 0.5 * h_val * k2)
            k4 = mm3(J1, Q + h_val * k3)
            Q = Q + h_val / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

            sm = sums[e].copy()
            gram_schmidt(Q, sm)
            sums[e] = sm
            Qs[e] = Q

    return sums / (n_keep * h_val)


def scan_lmax(method, k, eps_arr, u0):
    a1, a2, a3 = start_params
    n = int(round(CT_SCAN / h))
    start = int(CUT_SCAN * n)
    lam = lyap_multi(method, f, k, np.asarray(eps_arr, dtype=np.float64),
                     u0.copy(), h, n, start, a1, a2, a3)
    return lam.max(axis=1)  # старший поперечный показатель для каждого eps


def find_K(method, k, u0, var_name=""):
    """Минимальное K, при котором все поперечные показатели < -LAM_TOL.
    Возвращает (K_крит или None, (eps_грубой_сетки, lmax_грубой_сетки))."""
    eps = np.linspace(0.0, K_MAX, N_COARSE)
    lmax = scan_lmax(method, k, eps, u0)
    stable = lmax < -LAM_TOL
    coarse = (eps, lmax)

    # наименьший индекс, начиная с которого устойчиво всё до конца диапазона
    idx = None
    for i in range(len(eps) - 1, -1, -1):
        if stable[i]:
            idx = i
        else:
            break
    if idx is None:
        print(f"  связь по {var_name}: синхронизации нет при K <= {K_MAX}")
        return None, coarse

    lo = eps[idx - 1] if idx > 0 else 0.0
    hi = eps[idx]
    if idx == 0:
        return hi, coarse

    # уточнение внутри интервала [lo, hi]
    for _ in range(N_REFINE):
        grid = np.linspace(lo, hi, N_FINE)
        lm = scan_lmax(method, k, grid, u0)
        ok = np.where(lm < -LAM_TOL)[0]
        if len(ok) == 0:
            break
        j = ok[0]
        hi = grid[j]
        lo = grid[j - 1] if j > 0 else lo
    print(f"  связь по {var_name}: K_крит ~ {hi:.4f}")
    return hi, coarse


# ============================================================================
# ДВУНАПРАВЛЕННАЯ СВЯЗЬ (расщепление по Странгу, см. исходный код)
# ============================================================================
@njit
def bidirectional_run(method, f, k, eps, u0_drive, u0_resp, h_val, n, start,
                      a1, a2, a3):
    n_keep = n - start
    err = np.empty(n_keep + 1)
    x1k = np.empty(n_keep + 1)
    x2k = np.empty(n_keep + 1)

    u = u0_drive.copy()
    for i in range(start):
        _, s = method(f, u, 0.0, h_val, h_val, a1, a2, a3)
        u = s[1].copy()

    u1 = u.copy()
    u2 = u0_resp.copy()
    decay = np.exp(-eps * h_val)
    Q = np.eye(3)
    sums = np.zeros(3)
    err[0] = np.sqrt(np.sum((u1 - u2) ** 2))
    x1k[0] = u1[k]
    x2k[0] = u2[k]

    for j in range(n_keep):
        m = 0.5 * (u1[k] + u2[k])
        d = 0.5 * (u1[k] - u2[k]) * decay
        u1[k] = m + d
        u2[k] = m - d

        _, s1 = method(f, u1, 0.0, h_val, h_val, a1, a2, a3)
        u1 = s1[1].copy()
        _, s2 = method(f, u2, 0.0, h_val, h_val, a1, a2, a3)
        u2 = s2[1].copy()

        m = 0.5 * (u1[k] + u2[k])
        d = 0.5 * (u1[k] - u2[k]) * decay
        u1[k] = m + d
        u2[k] = m - d

        err[j + 1] = np.sqrt(np.sum((u1 - u2) ** 2))
        x1k[j + 1] = u1[k]
        x2k[j + 1] = u2[k]

        u_old = u
        _, s = method(f, u_old, 0.0, h_val, h_val, a1, a2, a3)
        u = s[1].copy()
        f0 = f(u_old, a1, a2, a3)
        f1 = f(u, a1, a2, a3)
        u_mid = 0.5 * (u_old + u) + h_val * (f0 - f1) / 8.0

        J0 = jac3(u_old, a1, a2, a3)
        Jm = jac3(u_mid, a1, a2, a3)
        J1 = jac3(u, a1, a2, a3)
        J0[k, k] -= 2.0 * eps
        Jm[k, k] -= 2.0 * eps
        J1[k, k] -= 2.0 * eps

        k1 = mm3(J0, Q)
        k2 = mm3(Jm, Q + 0.5 * h_val * k1)
        k3 = mm3(Jm, Q + 0.5 * h_val * k2)
        k4 = mm3(J1, Q + h_val * k3)
        Q = Q + h_val / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        gram_schmidt(Q, sums)

    return err, sums / (n_keep * h_val), x1k, x2k


def verdict(lams):
    lmax = max(lams)
    if lmax < -LAM_TOL: return True, "синхронизация есть (все показатели < 0)"
    if lmax > LAM_TOL: return False, "синхронизации нет (есть показатель > 0)"
    return False, "синхронизации нет (старший показатель ~ 0)"


def analyze_bidirectional(method, k, u0_drive, u0_resp, eps):
    a1, a2, a3 = start_params
    n = int(round(ct / h))
    start = int(CUT * n)

    err, lam, x1k, x2k = bidirectional_run(method, f, k, eps, u0_drive.copy(),
                                           u0_resp.copy(), h, n, start,
                                           a1, a2, a3)
    if not (np.all(np.isfinite(err)) and np.all(np.isfinite(lam))):
        return None

    synced, text = verdict(lam)
    t = CUT * ct + h * np.arange(len(err))
    return {"t": t, "err": err, "lam": tuple(lam), "synced": synced,
            "text": text, "eps": eps, "drv": x1k, "resp": x2k}


# ============================================================================
# Графики
# ============================================================================
def plot_axis(method_name, var_name, r):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    t = r["t"]
    idx_pts = np.linspace(0, len(t) - 1, PLOT_POINTS).astype(int)

    ax1.plot(t[idx_pts], r["drv"][idx_pts], color="blue", linewidth=1.2,
             label=f"ведущая {var_name}1(t)")
    ax1.plot(t[idx_pts], r["resp"][idx_pts], color="red", linewidth=1.0,
             linestyle="--", label=f"ведомая {var_name}2(t)")
    ax1.set_ylabel(var_name)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ax1.set_title(f"Координата {var_name}")

    e = np.nan_to_num(r["err"][idx_pts], nan=np.inf)
    ax2.semilogy(t[idx_pts], np.maximum(e, ERR_FLOOR), color="green",
                 linewidth=1.0)
    ax2.set_xlabel("t")
    ax2.set_ylabel("||u1 - u2||")
    ax2.grid(True, alpha=0.3)
    l1, l2, l3 = r["lam"]
    ax2.set_title(f"Ошибка, lambda = ({l1:+.4f}; {l2:+.4f}; {l3:+.4f})")

    fig.suptitle(f"Sprott Case B: двунаправленная связь по {var_name}, "
                 f"K = {r['eps']:.4f}, {method_name}")
    fig.tight_layout()

    filepath = os.path.join(save_folder,
                            f"SyncBidir_Case_B_{method_name}_{var_name}.png")
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"График сохранен: {filepath}")


def plot_scan(method_name, var_name, coarse, k_crit, k_used):
    eps, lmax = coarse
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, lmax, "o-", color="purple", markersize=4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    if k_crit is not None:
        ax.axvline(k_crit, color="orange", linestyle="--",
                   label=f"K_крит = {k_crit:.4f}")
        ax.axvline(k_used, color="green", linestyle="--",
                   label=f"K (с запасом) = {k_used:.4f}")
        ax.legend()
    ax.set_xlabel("K")
    ax.set_ylabel("старший поперечный показатель Ляпунова")
    ax.set_title(f"Подбор K, связь по {var_name}, {method_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    filepath = os.path.join(save_folder,
                            f"ScanK_Case_B_{method_name}_{var_name}.png")
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"График подбора K сохранен: {filepath}")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    u0_drive = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=3)
    u0_resp = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=3)

    summary = []  # (метод, переменная, K, lam, текст)

    for method_name, method in methods:
        for k in ACTIVE_INDICES:
            var_name = VAR_NAMES[ACTIVE_INDICES.index(k)]
            print(f"{method_name}, связь по {var_name}: подбор K...")

            k_crit, coarse = find_K(method, k, u0_drive, var_name)
            if k_crit is None:
                plot_scan(method_name, var_name, coarse, None, None)
                summary.append((method_name, var_name, None, None,
                                f"синхронизации нет при K <= {K_MAX}"))
                continue

            K = k_crit * K_MARGIN if k_crit > 0 else 0.0
            plot_scan(method_name, var_name, coarse, k_crit, K)

            print(f"{method_name}, связь по {var_name}: расчёт при K = {K:.4f}")
            r = analyze_bidirectional(method, k, u0_drive, u0_resp, K)
            if r is None:
                print(f"{method_name} по {var_name}: решение расходится")
                summary.append((method_name, var_name, K, None, "расходится"))
                continue

            plot_axis(method_name, var_name, r)
            summary.append((method_name, var_name, K, r["lam"], r["text"]))
            del r

    print("\nПодобранное K, поперечные показатели Ляпунова и вывод:")
    for method_name, var_name, K, lam, text in summary:
        if K is None or lam is None:
            print(f"{method_name:13s} по {var_name}: {text}")
            continue
        l1, l2, l3 = lam
        print(f"{method_name:13s} по {var_name}: K = {K:.4f}, "
              f"lambda = ({l1:+.4f}; {l2:+.4f}; {l3:+.4f}) -> {text}")

    print("\nПостроение всех графиков синхронизации завершено.")
