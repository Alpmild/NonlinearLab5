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
COUPLING_K = 1.0  # сила связи K
TAU = 1.0  # задержка tau (округляется до целого числа шагов h)
save_folder = "t3_feedback_sync_plots"
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
# ОДНОСТОРОННЯЯ СВЯЗЬ С ЗАДЕРЖКОЙ (обратная связь, Drive -> Response)
#
#     dx1/dt = F(x1)
#     dx2/dt = F(x2) + K * [x1_k(t - tau) - x2_k(t)] * e_k
#
# Связь действует по координате k (по каждой из VAR_NAMES).
# Система x1 (драйвер) свободна, x2 (ведомая) получает сигнал x1_k(t - tau).
#
# Все методы трёхмерные, поэтому ведомая система считается расщеплением по
# Странгу: половина шага связи -> шаг метода method(f, u, 0, h, h, a1, a2, a3)
# -> половина шага связи. Связь линейна по x2_k, поэтому при постоянном сигнале
# c на интервале s она решается точно:
#     x2_k -> c + (x2_k - c) * exp(-K * s).
# Сигнал c берётся из истории драйвера в середине каждого полушага
# (линейная интерполяция между узлами, задержка = m шагов, m = round(tau / h)).
#
# При tau != 0 полной синхронизации x2 = x1 нет, но существует решение
# запаздывающей синхронизации x2(t) = x1(t - tau): тогда связующий член равен
# нулю. Поэтому ошибка считается как ||x2(t) - x1(t - tau)||.
#
# Вывод о синхронизации - по трём поперечным показателям Ляпунова: уравнение
# в вариациях для e = x2(t) - x1(t - tau) не содержит запаздывания:
#     de/dt = (J(x) - K * E_kk) * e,
# где J - полный 3x3 якобиан F, E_kk - матрица с единицей на месте (k, k),
# x - траектория на аттракторе. Три вектора возмущения (столбцы Q)
# интегрируются RK4 и после каждого шага ортонормируются по Граму-Шмидту
# (алгоритм Бенеттина); показатель j = сумма log(норм j-го столбца) / время.
# Синхронизация есть, если все три показателя отрицательны.
# ============================================================================
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


@njit
def feedback_run(method, f, k, K, m, u0_drive, u0_resp, h_val, n, start,
                 a1, a2, a3):
    n_keep = n - start
    err = np.empty(n_keep + 1)
    resp = np.empty(n_keep + 1)  # x2_k(t) ведомой системы

    # история драйвера: hist[j] = x1 в момент (start - m + j) * h
    hist = np.empty((n_keep + m + 1, 3))

    # переходный процесс драйвера (запоминаем последние m значений)
    u = u0_drive.copy()
    for i in range(start):
        if i >= start - m:
            hist[i - (start - m)] = u
        _, s = method(f, u, 0.0, h_val, h_val, a1, a2, a3)
        u = s[1].copy()
    hist[m] = u

    # ведомая система стартует с другими начальными условиями
    x2 = u0_resp.copy()
    decay = np.exp(-K * 0.5 * h_val)  # затухание за половину шага связи
    Q = np.eye(3)
    sums = np.zeros(3)
    err[0] = np.sqrt(np.sum((x2 - hist[0]) ** 2))
    resp[0] = x2[k]

    for j in range(n_keep):
        # --- драйвер: шаг метода + уравнение в вариациях (RK4) ---
        u_old = u
        _, s = method(f, u_old, 0.0, h_val, h_val, a1, a2, a3)
        u = s[1].copy()
        hist[m + j + 1] = u

        f0 = f(u_old, a1, a2, a3)
        f1 = f(u, a1, a2, a3)
        u_mid = 0.5 * (u_old + u) + h_val * (f0 - f1) / 8.0
        J0 = jac3(u_old, a1, a2, a3)
        Jm = jac3(u_mid, a1, a2, a3)
        J1 = jac3(u, a1, a2, a3)
        J0[k, k] -= K
        Jm[k, k] -= K
        J1[k, k] -= K

        k1 = mm3(J0, Q)
        k2 = mm3(Jm, Q + 0.5 * h_val * k1)
        k3 = mm3(Jm, Q + 0.5 * h_val * k2)
        k4 = mm3(J1, Q + h_val * k3)
        Q = Q + h_val / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

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

        # --- ведомая система: связь по x1_k(t - tau) ---
        # задержанный сигнал в серединах двух полушагов
        c1 = 0.75 * hist[j, k] + 0.25 * hist[j + 1, k]
        c2 = 0.25 * hist[j, k] + 0.75 * hist[j + 1, k]

        x2[k] = c1 + (x2[k] - c1) * decay
        _, s2 = method(f, x2, 0.0, h_val, h_val, a1, a2, a3)
        x2 = s2[1].copy()
        x2[k] = c2 + (x2[k] - c2) * decay

        # ошибка относительно запаздывающего драйвера x1(t - tau)
        err[j + 1] = np.sqrt(np.sum((x2 - hist[j + 1]) ** 2))
        resp[j + 1] = x2[k]

    # ведущая координата с задержкой: x1_k(t - tau)
    drv = hist[:n_keep + 1, k].copy()
    return err, sums / (n_keep * h_val), drv, resp


def verdict(lams):
    lmax = max(lams)
    if lmax < -LAM_TOL: return True, "синхронизация есть (все показатели < 0)"
    if lmax > LAM_TOL: return False, "синхронизации нет (есть показатель > 0)"
    return False, "синхронизации нет (старший показатель ~ 0)"


def analyze_feedback(method, k, u0_drive, u0_resp, K=COUPLING_K, tau=TAU):
    a1, a2, a3 = start_params
    n = int(round(ct / h))
    start = int(CUT * n)
    m = int(round(tau / h))
    if m > start:
        raise ValueError("TAU должно быть не больше CUT * ct")

    err, lam, drv, resp = feedback_run(method, f, k, K, m, u0_drive.copy(),
                            u0_resp.copy(), h, n, start, a1, a2, a3)
    if not (np.all(np.isfinite(err)) and np.all(np.isfinite(lam))):
        return None

    synced, text = verdict(lam)
    t = CUT * ct + h * np.arange(len(err))
    it = start + np.arange(len(err))  # номер итерации (шага) от начала расчёта
    return {"t": t, "it": it, "err": err, "lam": tuple(lam), "synced": synced,
            "text": text, "drv": drv, "resp": resp}


# ============================================================================
# Построение графиков: одна картинка на (метод, ось), на ней три графика
# (после отсечения первых CUT * 100 % значений):
#   1) значения координаты: ведущая x1_k(t - tau) и ведомая x2_k(t);
#   2) ошибка ||x2(t) - x1(t - tau)|| по времени (ось времени общая с 1);
#   3) та же ошибка по номеру итерации (шага метода).
# ============================================================================
def plot_axis(method_name, var_name, r):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    ax2.sharex(ax1)  # графики 1 и 2 имеют общую ось времени

    t = r["t"]
    # одни и те же точки для обоих графиков
    idx_pts = np.linspace(0, len(t) - 1, PLOT_POINTS).astype(int)

    # 1) значения координаты
    ax1.plot(t[idx_pts], r["drv"][idx_pts], color="blue", linewidth=1.2,
             label=f"ведущая {var_name}1(t - tau)")
    ax1.plot(t[idx_pts], r["resp"][idx_pts], color="red", linewidth=1.0,
             linestyle="--", label=f"ведомая {var_name}2(t)")
    ax1.set_ylabel(var_name)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ax1.set_title(f"Значения координаты {var_name}")

    # 2) ошибка
    e = np.nan_to_num(r["err"][idx_pts], nan=np.inf)
    ax2.semilogy(t[idx_pts], np.maximum(e, ERR_FLOOR), color="green", linewidth=1.0)
    ax2.set_xlabel("t")
    ax2.set_ylabel("||x2(t) - x1(t - tau)||")
    ax2.grid(True, alpha=0.3)
    l1, l2, l3 = r["lam"]
    ax2.set_title(f"Ошибка, lambda = ({l1:+.4f}; {l2:+.4f}; {l3:+.4f})")

    # 3) ошибка по итерациям
    ax3.semilogy(r["it"][idx_pts], np.maximum(e, ERR_FLOOR), color="purple", linewidth=1.0)
    ax3.set_xlabel("номер итерации")
    ax3.set_ylabel("||x2(t) - x1(t - tau)||")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Ошибка по итерациям")

    fig.suptitle(f"Sprott Case B: связь по {var_name} с запаздыванием, "
                 f"K = {COUPLING_K}, tau = {TAU}, {method_name}")
    fig.tight_layout()

    filepath = os.path.join(save_folder,
                            f"SyncDelay_Case_B_{method_name}_{var_name}.png")
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    u0_drive = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=3)
    u0_resp = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=3)

    summary = []  # (метод, переменная, lam, текст) или None, если расходится

    for method_name, method in methods:
        for k in ACTIVE_INDICES:
            var_name = VAR_NAMES[ACTIVE_INDICES.index(k)]
            print(f"{method_name}, связь по {var_name}...")

            # каждая ось - отдельный расчёт со своей связью
            r = analyze_feedback(method, k, u0_drive, u0_resp)
            if r is None:
                print(f"{method_name} по {var_name}: решение расходится")
                summary.append((method_name, var_name, None, None))
                continue

            plot_axis(method_name, var_name, r)
            summary.append((method_name, var_name, r["lam"], r["text"]))
            del r  # освобождаем память (массивы длинные)

    print(f"\nПоперечные показатели Ляпунова и вывод о синхронизации "
          f"(связь с запаздыванием, K = {COUPLING_K}, tau = {TAU}):")
    for method_name, var_name, lam, text in summary:
        if lam is None:
            print(f"{method_name:13s} по {var_name}: решение расходится")
            continue
        l1, l2, l3 = lam
        print(f"{method_name:13s} по {var_name}: "
              f"lambda = ({l1:+.4f}; {l2:+.4f}; {l3:+.4f}) -> {text}")

    print("\nПостроение всех графиков синхронизации завершено.")