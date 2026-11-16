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
save_folder = "t1_sync_plots"
os.makedirs(save_folder, exist_ok=True)

# Список переменных для анализа.
# Индексы: x=0, y=1, z=2
VAR_NAMES = [
    "x",
    "y",
    "z"
]

# Создаем словарь для быстрого перевода имени в индекс
VAR_MAP = {"x": 0, "y": 1, "z": 2}
# Получаем список индексов для цикла
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
    ("RK4", njit(rk4)),
]


# ============================================================================
# Метод Пекоры-Кэррола и вспомогательные функции (без изменений)
# ============================================================================
@njit(cache=True)
def hermite(X, D, h, t):
    n = X.shape[0] - 1
    i = int(t / h)
    if i < 0: i = 0
    if i > n - 1: i = n - 1
    s = t / h - i
    s2 = s * s
    s3 = s2 * s
    return ((2 * s3 - 3 * s2 + 1) * X[i]
            + (s3 - 2 * s2 + s) * h * D[i]
            + (-2 * s3 + 3 * s2) * X[i + 1]
            + (s3 - s2) * h * D[i + 1])


def hermite_vec(X, D, h_val, t):
    """То же, что hermite, но для массива моментов времени t (для графиков)."""
    n = X.shape[0] - 1
    i = np.clip((t / h_val).astype(np.int64), 0, n - 1)
    s = t / h_val - i
    s2 = s * s
    s3 = s2 * s
    return ((2 * s3 - 3 * s2 + 1) * X[i]
            + (s3 - 2 * s2 + s) * h_val * D[i]
            + (-2 * s3 + 3 * s2) * X[i + 1]
            + (s3 - s2) * h_val * D[i + 1])


def make_response_f(k, X, D, h_val):
    # Передаем h внутрь, чтобы избежать замыкания на глобальную переменную,
    # хотя в numba это часто работает и так, но так надежнее.
    @njit
    def f_resp(u, a1, a2, a3):
        drv = hermite(X, D, h_val, u[k])
        x = u[0];
        y = u[1];
        z = u[2]
        if k == 0:
            x = drv
        elif k == 1:
            y = drv
        else:
            z = drv

        v = np.empty(3)
        v[0] = a1 * y * z
        v[1] = x - a2 * y
        v[2] = a3 - x * y
        v[k] = 1.0
        return v

    return f_resp


def drive_derivative(s, k, a1, a2, a3):
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    return [a1 * y * z, x - a2 * y, a3 - x * y][k]


@njit(cache=True)
def jac(k, v, a1, a2, a3):
    J = np.zeros((2, 2))
    if k == 0:
        J[0, 0] = -a2;
        J[1, 0] = -v
    elif k == 1:
        J[0, 1] = a1 * v;
        J[1, 0] = -v
    else:
        J[0, 1] = a1 * v;
        J[1, 0] = 1.0;
        J[1, 1] = -a2
    return J


@njit(cache=True)
def mm(A, B):
    C = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            C[i, j] = A[i, 0] * B[0, j] + A[i, 1] * B[1, j]
    return C


@njit(cache=True)
def conditional_lyapunov(k, X, D, h_val, a1, a2, a3):
    n = X.shape[0] - 1
    Q = np.eye(2)
    s1 = 0.0;
    s2 = 0.0
    for i in range(n):
        v0 = X[i];
        v1 = X[i + 1]
        vm = 0.5 * (v0 + v1) + h_val * (D[i] - D[i + 1]) / 8.0
        J0 = jac(k, v0, a1, a2, a3)
        Jm = jac(k, vm, a1, a2, a3)
        J1 = jac(k, v1, a1, a2, a3)

        k1 = mm(J0, Q)
        k2 = mm(Jm, Q + 0.5 * h_val * k1)
        k3 = mm(Jm, Q + 0.5 * h_val * k2)
        k4 = mm(J1, Q + h_val * k3)
        Q = Q + h_val / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        na = np.sqrt(Q[0, 0] ** 2 + Q[1, 0] ** 2)
        Q[0, 0] /= na;
        Q[1, 0] /= na
        dot = Q[0, 0] * Q[0, 1] + Q[1, 0] * Q[1, 1]
        Q[0, 1] -= dot * Q[0, 0];
        Q[1, 1] -= dot * Q[1, 0]
        nb = np.sqrt(Q[0, 1] ** 2 + Q[1, 1] ** 2)
        Q[0, 1] /= nb;
        Q[1, 1] /= nb

        s1 += np.log(na);
        s2 += np.log(nb)
    T = n * h_val
    return s1 / T, s2 / T


def verdict(l1, l2):
    lmax = max(l1, l2)
    if lmax < -LAM_TOL: return True, "синхронизация есть (все показатели < 0)"
    if lmax > LAM_TOL: return False, "синхронизации нет (есть показатель > 0)"
    return False, "синхронизации нет (старший показатель ~ 0)"


# ============================================================================
# Один расчёт
# ============================================================================
def analyze(method, k, u0_drive, u0_resp):
    a1, a2, a3 = start_params
    t, s = method(f, u0_drive.copy(), 0.0, ct, h, a1, a2, a3)
    if not np.all(np.isfinite(s)): return None

    start = int(CUT * (len(s) - 1))
    t = t[start:] - t[start]
    s = s[start:]
    ct_eff = (len(s) - 1) * h

    X = np.ascontiguousarray(s[:, k])
    D = np.ascontiguousarray(drive_derivative(s, k, a1, a2, a3))

    l1, l2 = conditional_lyapunov(k, X, D, h, a1, a2, a3)
    synced, text = verdict(l1, l2)

    f_resp = make_response_f(k, X, D, h)
    ur = u0_resp.copy()
    ur[k] = 0.0
    _, r = method(f_resp, ur, 0.0, ct_eff, h, a1, a2, a3)

    others = [j for j in range(3) if j != k]
    with np.errstate(all="ignore"):
        err = np.sqrt(np.sum((s[:, others] - r[:, others]) ** 2, axis=1))

    return {"t": t + CUT * ct, "err": err, "lam": (l1, l2), "synced": synced,
            "text": text,
            "drv": s[:, k],
            # значение переменной k, которое реально использует ведомая система
            # (в состоянии ведомой системы на месте k хранится время)
            "resp": hermite_vec(X, D, h, r[:, k])}


# ============================================================================
# Построение графиков: одна картинка на (метод, ось синхронизации), на ней два
# графика с одной и той же осью времени (после отсечения CUT * 100 % значений):
#   1) значение передаваемой координаты k: ведущая (сплошная) и ведомая (пунктир);
#   2) ошибка ||e|| по двум остальным координатам.
# ============================================================================
def plot_axis(method_name, var_name, r):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    t = r["t"]
    # одни и те же точки для обоих графиков
    idx_pts = np.linspace(0, len(t) - 1, PLOT_POINTS).astype(int)

    # 1) значение координаты ведущей и ведомой систем
    ax1.plot(t[idx_pts], r["drv"][idx_pts], color="blue", linewidth=1.2,
             label=f"ведущая {var_name}(t)")
    ax1.plot(t[idx_pts], r["resp"][idx_pts], color="red", linewidth=1.0,
             linestyle="--", label=f"ведомая {var_name}(t)")
    ax1.set_ylabel(var_name)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ax1.set_title(f"Координата {var_name}")

    # 2) ошибка
    e = np.nan_to_num(r["err"][idx_pts], nan=np.inf)
    ax2.semilogy(t[idx_pts], np.maximum(e, ERR_FLOOR), color="green", linewidth=1.0)
    ax2.set_xlabel("t")
    ax2.set_ylabel("||e||")
    ax2.grid(True, alpha=0.3)
    l1, l2 = r["lam"]
    ax2.set_title(f"Ошибка, lambda = ({l1:+.4f}; {l2:+.4f})")

    fig.suptitle(f"Sprott Case B: метод Пекоры-Кэррола по {var_name}, {method_name}")
    fig.tight_layout()

    filepath = os.path.join(save_folder,
                            f"Sync_Case_B_{method_name}_{var_name}.png")
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    u0_drive = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=3)
    u0_resp = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=3)

    summary = []  # (метод, переменная, lam, текст) или None, если драйвер расходится

    for method_name, method in methods:
        for k in ACTIVE_INDICES:
            var_name = VAR_NAMES[ACTIVE_INDICES.index(k)]
            print(f"{method_name}, синхронизация по {var_name}...")

            # каждая ось - отдельная синхронизация
            r = analyze(method, k, u0_drive, u0_resp)
            if r is None:
                print(f"{method_name} по {var_name}: драйвер расходится")
                summary.append((method_name, var_name, None, None))
                continue

            plot_axis(method_name, var_name, r)
            summary.append((method_name, var_name, r["lam"], r["text"]))
            del r  # освобождаем память (массивы длинные)

    print("\nУсловные показатели Ляпунова и вывод о синхронизации:")
    for method_name, var_name, lam, text in summary:
        if lam is None:
            print(f"{method_name:13s} по {var_name}: драйвер расходится")
            continue
        l1, l2 = lam
        print(f"{method_name:13s} по {var_name}: "
              f"lambda = ({l1:+.4f}; {l2:+.4f}) -> {text}")

    print("\nПостроение всех графиков синхронизации завершено.")