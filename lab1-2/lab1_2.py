from math import sqrt



X0 = 0.05
DELTA0 = 0.01
SIGMA = 0.001
EPS = 0.0001  


def f(x: float) -> float:
    return (x - 1) ** 2 * (x + 1) ** 4 * (x - 2) ** 3


class FunctionCounter:

    def __init__(self):
        self.nf = 0

    def __call__(self, x: float) -> float:
        self.nf += 1
        return f(x)


def svenn(F, x0: float, delta: float):
    if delta == 0:
        raise ValueError("Початковий крок Δ₀ не може дорівнювати нулю.")

    f0 = F(x0)
    x1 = x0 + delta
    f1 = F(x1)

    if f1 >= f0:
        delta = -delta
        x1 = x0 + delta
        f1 = F(x1)
        if f1 >= f0:
            return min(x0 - abs(delta), x0 + abs(delta)), max(x0 - abs(delta), x0 + abs(delta))

    x_prev, x_cur, f_cur = x0, x1, f1
    k = 1
    while k <= 1000:
        x_next = x_cur + (2**k) * delta
        f_next = F(x_next)
        if f_next >= f_cur:
            return min(x_prev, x_next), max(x_prev, x_next)
        x_prev, x_cur, f_cur = x_cur, x_next, f_next
        k += 1

    raise RuntimeError("Алгоритм Свенна не знайшов межу інтервалу за 1000 кроків.")


def dichotomy(F, a: float, b: float, sigma: float, eps: float):
    """Метод дихотомії"""
    if not (a < b and 0 < eps < sigma):
        raise ValueError("Потрібно a < b та 0 < ε < σ.")

    while b - a > sigma:
        x1 = (a + b) / 2 - eps / 2
        x2 = x1 + eps
        f1, f2 = F(x1), F(x2)
        if f1 < f2:
            b = x2
        elif f1 > f2:
            a = x1
        else:
           
            a, b = x1, x2

    return (a + b) / 2, b - a


def halving(F, a: float, b: float, sigma: float):
    """Трьохточковий половинний"""
    if not (a < b and sigma > 0):
        raise ValueError("Потрібно a < b та σ > 0.")

    xm = (a + b) / 2
    fm = F(xm)
    while b - a > sigma:
        length = b - a
        x1 = a + length / 4
        x2 = b - length / 4
        f1, f2 = F(x1), F(x2)

        if f1 < fm:
            b, xm, fm = xm, x1, f1
        elif f2 < fm:
            a, xm, fm = xm, x2, f2
        else:
            a, b = x1, x2

    return (a + b) / 2, b - a


def main():
    x_analytic = (7 - 2 * sqrt(10)) / 9
    f_analytic = f(x_analytic)

    print("Лабораторні 1–2 · варіант 11")
    print("f(x) = (x−1)²(x+1)⁴(x−2)³")
    print(f"x₀ = {X0}; Δ₀ = {DELTA0}; σ = {SIGMA}; ε = {EPS}")
    print(f"Аналітичний локальний мінімум: x* = {x_analytic:.8f}, f(x*) = {f_analytic:.8f}\n")

    F_svenn = FunctionCounter()
    a0, b0 = svenn(F_svenn, X0, DELTA0)
    print(
        f"Свенн: інтервал [{a0:.6f}, {b0:.6f}], L₀ = {b0 - a0:.6f}, "
        f"Nf пошуку = {F_svenn.nf}; мінімум в інтервалі: {a0 <= x_analytic <= b0}"
    )

    F_dichotomy = FunctionCounter()
    x_d, length_d = dichotomy(F_dichotomy, a0, b0, SIGMA, EPS)
    print(
        f"Дихотомія: x* ≈ {x_d:.8f}, f(x*) = {f(x_d):.8f}, "
        f"L = {length_d:.6f}, Nf пошуку = {F_dichotomy.nf}, "
        f"|x*−x_ан| = {abs(x_d - x_analytic):.2e}"
    )

    F_halving = FunctionCounter()
    x_h, length_h = halving(F_halving, a0, b0, SIGMA)
    print(
        f"Половинний поділ: x* ≈ {x_h:.8f}, f(x*) = {f(x_h):.8f}, "
        f"L = {length_h:.6f}, Nf пошуку = {F_halving.nf}, "
        f"|x*−x_ан| = {abs(x_h - x_analytic):.2e}"
    )


if __name__ == "__main__":
    main()
