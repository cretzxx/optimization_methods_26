import math

x0 = 0.05
delta = 0.01
eps = 0.0001


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


def fibonacci(F, a, b, eps):
    L0 = b - a
    R = L0 / eps

    fib = [1, 1]
    while fib[-1] < R:
        fib.append(fib[-1] + fib[-2])

    N = len(fib) - 1
    d = eps / 10

    x1 = a + (fib[N - 2] / fib[N]) * (b - a)
    x2 = a + (fib[N - 1] / fib[N]) * (b - a)
    f1, f2 = F(x1), F(x2)

    rows = []

    for k in range(1, N - 1):
        m = N - k + 1
        L = b - a

        if k == N - 2:
            x1 = (a + b) / 2
            f1 = F(x1)
            x2 = x1 + d
            f2 = F(x2)
            dec = "m = 3 → зсув δ, СТОП"
        elif f1 <= f2:
            dec = "f1 <= f2 → b = x2 (успадковано)"
        else:
            dec = "f1 > f2 → a = x1 (успадковано)"

        row = [k, m, a, b, x1, x2, f1, f2, L, dec]
        rows.append(row)

        if k == N - 2:
            if f1 <= f2:
                b = x2
            else:
                a = x1
            break

        if f1 <= f2:
            b = x2
            x2, f2 = x1, f1
            x1 = a + (fib[m - 3] / fib[m - 1]) * (b - a)
            f1 = F(x1)
        else:
            a = x1
            x1, f1 = x2, f2
            x2 = a + (fib[m - 2] / fib[m - 1]) * (b - a)
            f2 = F(x2)

    return rows, (a + b) / 2, b - a, N, fib[N]


x_ex = (7 - 2 * math.sqrt(10)) / 9

print("Варіант 11: f(x) = (x - 1)^2 (x + 1)^4 (x - 2)^3")
print(f"x0 = {x0}, delta = {delta}, eps = {eps}\n")

S = Counter(f)
a0, b0 = svenn(S, x0, delta)
L0 = b0 - a0
print(
    f"Свенн: [a0, b0] = [{a0:.6f}, {b0:.6f}], L0 = {L0:.6f}, обчислень:"
    f" {S.nf}\n"
)

Fi = Counter(f)
rows, x_fib, L_fib, N, FN = fibonacci(Fi, a0, b0, eps)

print(f"Пошук N: R = L0/eps = {L0/eps:.2f} <= F_{N} = {FN}")
print(
    f"{'k':>2} {'m':>2} {'a_k':>9} {'b_k':>9} {'x1':>9} {'x2':>9}"
    f" {'f(x1)':>12} {'f(x2)':>12} {'L_k':>9}  Рішення"
)
for r in rows:
    print(
        f"{r[0]:>2} {r[1]:>2} {r[2]:9.6f} {r[3]:9.6f} {r[4]:9.6f}"
        f" {r[5]:9.6f} {r[6]:12.8f} {r[7]:12.8f} {r[8]:9.6f}  {r[9]}"
    )

print(f"\nx* = {x_fib:.6f}, f(x*) = {f(x_fib):.8f}")
print(f"Обчислень f(x): {Fi.nf}, похибка: {abs(x_fib - x_ex):.8f}")