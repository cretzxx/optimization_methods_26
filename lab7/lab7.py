import math

x0_start = 0.05
delta = 0.01
eps = 0.001


def f(x):
    return (x - 1) ** 2 * (x + 1) ** 4 * (x - 2) ** 3


def df(x):
    return (x - 1) * (x + 1) ** 3 * (x - 2) ** 2 * (
        2 * (x + 1) * (x - 2) + 4 * (x - 1) * (x - 2) + 3 * (x - 1) * (x + 1)
    )


def svenn(F, x0, h):
    f0, fr = F(x0), F(x0 + h)
    if fr >= f0:
        fl = F(x0 - h)
        if fl >= f0:
            return x0 - h, x0 + h
        h = -h
        fr = fl
    xp, x, fx = x0, x0 + h, fr
    k = 1
    while True:
        xn = x + 2**k * h
        fn = F(xn)
        if fn < fx:
            xp, x, fx = x, xn, fn
            k += 1
        else:
            return min(xp, xn), max(xp, xn)


a0, b0 = svenn(f, x0_start, delta)
xm, xk = a0, b0
pm, pk = df(xm), df(xk)

assert pm * pk < 0, "Похідна має змінювати знак на кінцях інтервалу!"

print("=== ЛАБОРАТОРНА РОБОТА № 7: Метод січних ===")
print(
    f"{'k':>2} {'x_{k-1}':>10} {'f\'(x_{k-1})':>13} {'x_k':>10}"
    f" {'f\'(x_k)':>12} {'x_{k+1}':>10} {'f\'(x_{k+1})':>13}"
    f" {'f(x_{k+1})':>13}  Дія"
)

max_iter = 100
for k in range(1, max_iter + 1):
    denom = pk - pm
    if abs(denom) < 1e-12:
        raise RuntimeError("Різниця похідних близька до нуля; метод січних зупинено.")

    xn = (pk * xm - pm * xk) / denom
    pn = df(xn)
    fn = f(xn)

    if pn > 0:
        action = "xk <- xn"
    else:
        action = "xm <- xn"

    print(
        f"{k:2d} {xm:10.6f} {pm:13.6f} {xk:10.6f} {pk:12.6f} {xn:10.6f}"
        f" {pn:13.6f} {fn:13.6f}  {action}"
    )

    if abs(pn) <= eps:
        x_star = xn
        break

    if pn > 0:
        xk, pk = xn, pn
    else:
        xm, pm = xn, pn
else:
    raise RuntimeError(f"Метод січних не досяг точності за {max_iter} ітерацій.")

print(
    f"\nРезультат методу січних: x* = {x_star:.6f}, f(x*) = {f(x_star):.8f}"
)
print(f"Загальна кількість ітерацій: {k}")
print(f"Кількість обчислень похідної f': {2 + k}")
