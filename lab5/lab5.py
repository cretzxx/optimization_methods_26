import math

x0 = 0.05
delta = 0.01
sigma_f = 0.01
sigma_x = 0.001


def f(x):
    return (x - 1) ** 2 * (x + 1) ** 4 * (x - 2) ** 3


class Counter:

    def __init__(self, func):
        self.func = func
        self.nf = 0
        self.points = []

    def __call__(self, x):
        self.nf += 1
        self.points.append(x)
        return self.func(x)


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


def powell(F, x1_start, dx):
    x1 = x1_start
    x2 = x1 + dx
    f1 = F(x1)
    f2 = F(x2)

    if f1 > f2:
        x3 = x1 + 2 * dx
    else:
        x3 = x1 - dx
    f3 = F(x3)

    rows = []
    k = 1

    for k in range(1, 51):
        xs = [x1, x2, x3]
        fs = [f1, f2, f3]

        f_min_val = min(fs)
        f_max_val = max(fs)
        i_min = fs.index(f_min_val)
        i_max = fs.index(f_max_val)

        x_min = xs[i_min]
        x_max = xs[i_max]

        if abs(x2 - x1) < 1e-12 or abs(x3 - x1) < 1e-12 or abs(x3 - x2) < 1e-12:
            raise RuntimeError("Три точки Пауелла збіглися; змініть початковий крок Δx.")
        a1 = (f2 - f1) / (x2 - x1)
        a2 = ((f3 - f1) / (x3 - x1) - a1) / (x3 - x2)

        if a2 > 0:
            xt = -a1 / (2 * a2) + (x1 + x2) / 2
        else:
            xt = 2 * x_min - x_max

        ft = F(xt)

        diff_f = abs(f_min_val - ft)
        diff_x = abs(x_min - xt)

        action = f"x{i_max+1} <- x_tilde"
        row = [
            k,
            x1,
            x2,
            x3,
            f1,
            f2,
            f3,
            a1,
            a2,
            xt,
            ft,
            diff_f,
            diff_x,
            action,
        ]
        rows.append(row)

        if diff_f <= sigma_f and diff_x <= sigma_x:
            row[13] = "Умови виконано -> СТОП"
            return rows, xt

        xs[i_max] = xt
        fs[i_max] = ft
        x1, x2, x3 = xs[0], xs[1], xs[2]
        f1, f2, f3 = fs[0], fs[1], fs[2]

    raise RuntimeError("Метод Пауелла не збігся за 50 ітерацій.")


S = Counter(f)
a0, b0 = svenn(S, x0, delta)

P = Counter(f)
rows, x_star = powell(P, a0, delta)

print("Варіант 11: f(x) = (x - 1)^2 (x + 1)^4 (x - 2)^3")
print(f"Початкова точка зі Свенна x1 = a0 = {a0:.6f}, dx = {delta}\n")

print(
    f"{'k':>2} {'x1':>9} {'x2':>9} {'x3':>9} {'f(x1)':>10} {'f(x2)':>10}"
    f" {'f(x3)':>10} {'a1':>10} {'a2':>10} {'x_til':>9} {'f_til':>10}"
    f" {'|f_min-ft|':>11} {'|x_min-xt|':>11}  Дія"
)
for r in rows:
    print(
        f"{r[0]:>2} {r[1]:9.6f} {r[2]:9.6f} {r[3]:9.6f} {r[4]:10.6f}"
        f" {r[5]:10.6f} {r[6]:10.6f} {r[7]:10.6f} {r[8]:10.6f} {r[9]:9.6f}"
        f" {r[10]:10.6f} {r[11]:11.6f} {r[12]:11.6f}  {r[13]}"
    )

print(f"\nx* = {x_star:.6f}, f(x*) = {f(x_star):.8f}")
print(f"Загальна кількість обчислень f(x): {P.nf}")
