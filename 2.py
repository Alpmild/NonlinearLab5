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
COUPLING_EPS = 1.0  # сила двунаправленной связи eps
save_folder = "t2_sync_plots"
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
# ДВУНАПРАВЛЕННАЯ СВЯЗЬ
#
# Две одинаковые системы связаны по координате k взаимной диффузионной связью:
#     du1/dt = f(u1) + eps * (u2_k - u1_k) * e_k
#     du2/dt = f(u2) + eps * (u1_k - u2_k) * e_k
# Каждая система одновременно и драйвер, и ведомая.
#
# Все методы трёхмерные, поэтому пара (6 переменных) считается расщеплением
# по Странгу: половина шага связи -> шаг метода для каждой системы отдельно
# (method(f, u, 0, h, h, a1, a2, a3), т.е. те же аргументы и тот же f) ->
# половина шага связи. Линейная связь решается точно: полусумма u1_k, u2_k
# сохраняется, а полуразность умножается на exp(-eps * h) за половину шага.
#
# Вывод о синхронизации - по трём поперечным показателям Ляпунова: это
# уравнение в вариациях для разности delta = u1 - u2 вдоль синхронной
# траектории u (отдельная система без связи):
#     d(delta)/dt = (J(u) - 2 * eps * E_kk) * delta,
# где J - полный 3x3 якобиан f, E_kk - матрица с единицей на месте (k, k).
# Три вектора возмущения (столбцы Q) интегрируются RK4 и после каждого шага
# ортонормируются по Граму-Шмидту (алгоритм Бенеттина). Показатель с номером j
# = сумма log(норма j-го столбца до нормировки) / время. Синхронизация есть,
# если все три показателя отрицательны.
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
def bidirectional_run(method, f, k, eps, u0_drive, u0_resp, h_val, n, start,
                      a1, a2, a3):
    n_keep = n - start
    err = np.empty(n_keep + 1)
    x1k = np.empty(n_keep + 1)  # координата k первой системы
    x2k = np.empty(n_keep + 1)  # координата k второй системы

    # переходный процесс: считаем одну систему (она же синхронная траектория)
    u = u0_drive.copy()
    for i in range(start):
        _, s = method(f, u, 0.0, h_val, h_val, a1, a2, a3)
        u = s[1].copy()

    # дальше: связанная пара + синхронная траектория для показателей Ляпунова
    u1 = u.copy()
    u2 = u0_resp.copy()
    decay = np.exp(-eps * h_val)  # полуразность за половину шага связи
    Q = np.eye(3)
    sums = np.zeros(3)
    err[0] = np.sqrt(np.sum((u1 - u2) ** 2))
    x1k[0] = u1[k]
    x2k[0] = u2[k]

    for j in range(n_keep):
        # 1) половина шага связи
        m = 0.5 * (u1[k] + u2[k])
        d = 0.5 * (u1[k] - u2[k]) * decay
        u1[k] = m + d
        u2[k] = m - d

        # 2) шаг метода для каждой системы
        _, s1 = method(f, u1, 0.0, h_val, h_val, a1, a2, a3)
        u1 = s1[1].copy()
        _, s2 = method(f, u2, 0.0, h_val, h_val, a1, a2, a3)
        u2 = s2[1].copy()

        # 3) вторая половина шага связи
        m = 0.5 * (u1[k] + u2[k])
        d = 0.5 * (u1[k] - u2[k]) * decay
        u1[k] = m + d
        u2[k] = m - d

        err[j + 1] = np.sqrt(np.sum((u1 - u2) ** 2))
        x1k[j + 1] = u1[k]
        x2k[j + 1] = u2[k]

        # синхронная траектория и уравнение в вариациях (RK4)
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

        # ортонормировка Грама-Шмидта
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

    return err, sums / (n_keep * h_val), x1k, x2k


def verdict(lams):
    lmax = max(lams)
    if lmax < -LAM_TOL: return True, "синхронизация есть (все показатели < 0)"
    if lmax > LAM_TOL: return False, "синхронизации нет (есть показатель > 0)"
    return False, "синхронизации нет (старший показатель ~ 0)"


def analyze_bidirectional(method, k, u0_drive, u0_resp, eps=COUPLING_EPS):
    a1, a2, a3 = start_params
    n = int(round(ct / h))
    start = int(CUT * n)

    err, lam, x1k, x2k = bidirectional_run(method, f, k, eps, u0_drive.copy(),
                                 u0_resp.copy(), h, n, start, a1, a2, a3)
    if not (np.all(np.isfinite(err)) and np.all(np.isfinite(lam))):
        return None

    synced, text = verdict(lam)
    t = CUT * ct + h * np.arange(len(err))
    return {"t": t, "err": err, "lam": tuple(lam), "synced": synced,
            "text": text, "eps": eps, "drv": x1k, "resp": x2k}


# ============================================================================
# Построение графиков: одна картинка на (метод, ось связи), на ней два графика
# с одной и той же осью времени (после отсечения CUT * 100 % значений):
#   1) значение координаты k первой ("ведущей") и второй ("ведомой") систем;
#   2) ошибка ||u1 - u2|| по всем трём компонентам.
# ============================================================================
def plot_axis(method_name, var_name, r):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    t = r["t"]
    # одни и те же точки для обоих графиков
    idx_pts = np.linspace(0, len(t) - 1, PLOT_POINTS).astype(int)

    # 1) значение координаты двух систем
    ax1.plot(t[idx_pts], r["drv"][idx_pts], color="blue", linewidth=1.2,
             label=f"ведущая {var_name}1(t)")
    ax1.plot(t[idx_pts], r["resp"][idx_pts], color="red", linewidth=1.0,
             linestyle="--", label=f"ведомая {var_name}2(t)")
    ax1.set_ylabel(var_name)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ax1.set_title(f"Координата {var_name}")

    # 2) ошибка
    e = np.nan_to_num(r["err"][idx_pts], nan=np.inf)
    ax2.semilogy(t[idx_pts], np.maximum(e, ERR_FLOOR), color="green", linewidth=1.0)
    ax2.set_xlabel("t")
    ax2.set_ylabel("||u1 - u2||")
    ax2.grid(True, alpha=0.3)
    l1, l2, l3 = r["lam"]
    ax2.set_title(f"Ошибка, lambda = ({l1:+.4f}; {l2:+.4f}; {l3:+.4f})")

    fig.suptitle(f"Sprott Case B: двунаправленная связь по {var_name}, "
                 f"eps = {COUPLING_EPS}, {method_name}")
    fig.tight_layout()

    filepath = os.path.join(save_folder,
                            f"SyncBidir_Case_B_{method_name}_{var_name}.png")
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
            r = analyze_bidirectional(method, k, u0_drive, u0_resp)
            if r is None:
                print(f"{method_name} по {var_name}: решение расходится")
                summary.append((method_name, var_name, None, None))
                continue

            plot_axis(method_name, var_name, r)
            summary.append((method_name, var_name, r["lam"], r["text"]))
            del r  # освобождаем память (массивы длинные)

    print(f"\nПоперечные показатели Ляпунова и вывод о синхронизации "
          f"(двунаправленная связь, eps = {COUPLING_EPS}):")
    for method_name, var_name, lam, text in summary:
        if lam is None:
            print(f"{method_name:13s} по {var_name}: решение расходится")
            continue
        l1, l2, l3 = lam
        print(f"{method_name:13s} по {var_name}: "
              f"lambda = ({l1:+.4f}; {l2:+.4f}; {l3:+.4f}) -> {text}")

    print("\nПостроение всех графиков синхронизации завершено.")