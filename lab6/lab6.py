import math

x0_start = 0.05
delta = 0.01
eps = 0.001
alpha = 0.5


def f(x):
    return (x - 1) ** 2 * (x + 1) ** 4 * (x - 2) ** 3


def df(x):
    return (x - 1) * (x + 1) ** 3 * (x - 2) ** 2 * (
        2 * (x + 1) * (x - 2) + 4 * (x - 1) * (x - 2) + 3 * (x - 1) * (x + 1)
    )


def d2f(x):

    p = (x - 1) * (x + 1) ** 3 * (x - 2) ** 2
    q = 2 * (x + 1) * (x - 2) + 4 * (x - 1) * (x - 2) + 3 * (x - 1) * (x + 1)
    dp = (x + 1) ** 2 * (x - 2) * (
        (x + 1) * (x - 2) + 3 * (x - 1) * (x - 2) + 2 * (x - 1) * (x + 1)
    )
    dq = 20 * x - 14
    return dp * q + p * dq


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
x0 = a0

print("=== ЧАСТИНА А: Класичний метод Ньютона ===")
print(
    f"{'k':>2} {'x_k':>10} {'f(x_k)':>12} {'f\'(x_k)':>12} {'|f\'|':>10}"
    f" {'f\"(x_k)':>10}"
)

xk = x0
for k in range(10):
    fk = f(xk)
    dfk = df(xk)
    d2fk = d2f(xk)
    abs_dfk = abs(dfk)

    print(
        f"{k:2d} {xk:10.6f} {fk:12.6f} {dfk:12.6f} {abs_dfk:10.6f}"
        f" {d2fk:10.6f}"
    )

    if abs_dfk <= eps:
        break
    if d2fk <= 1e-12:
        raise RuntimeError("Для кроку Ньютона потрібна додатна друга похідна.")
    xk = xk - dfk / d2fk

print(f"\nРезультат класичного Ньютона: x* = {xk:.6f}, f(x*) = {f(xk):.8f}\n")

print("=== ЧАСТИНА Б: Метод Ньютона-Рафсона з регулюванням кроку ===")
print(
    f"{'k':>2} {'x_k':>10} {'f(x_k)':>10} {'f\'':>10} {'f\"':>10} {'t':>5}"
    f" {'x_bar':>10} {'f(x_bar)':>10} {'f\'(x_bar)':>10} {'Умова (3.10)':>12}"
)

xk = x0
k_step = 0
while True:
    fk = f(xk)
    dfk = df(xk)
    d2fk = d2f(xk)

    if abs(dfk) <= eps:
        break
    if d2fk <= 1e-12:
        raise RuntimeError("Для кроку Ньютона-Рафсона потрібна додатна друга похідна.")

    t = 1.0
    while True:
        x_bar = xk - t * dfk / d2fk
        f_bar = f(x_bar)
        df_bar = df(x_bar)

        cond = f_bar - fk <= -alpha * t * (dfk**2) / d2fk

        print(
            f"{k_step:2d} {xk:10.6f} {fk:10.6f} {dfk:10.6f} {d2fk:10.6f} {t:5.2f}"
            f" {x_bar:10.6f} {f_bar:10.6f} {df_bar:10.6f}"
            f" {'ok' if cond else 'no':>12}"
        )

        if cond:
            xk = x_bar
            break
        t /= 2.0
        if t < 1e-8:
            raise RuntimeError("Крок став замалим: умова зменшення функції не виконана.")

    k_step += 1
    if abs(df_bar) <= eps:
        break

print(
    f"\nРезультат Ньютона-Рафсона з кроком: x* = {xk:.6f}, f(x*) = {f(xk):.8f}"
)
