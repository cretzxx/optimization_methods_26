import math

x0 = 0.05
delta = 0.01
sigma = 0.001
eps = 0.0001
tau = (math.sqrt(5) - 1) / 2

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
        xn = x + 2 ** k * h
        fn = F(xn)
        if fn < fx:
            xp, x, fx = x, xn, fn
            k += 1
        else:
            return min(xp, xn), max(xp, xn)

def golden_section(F, a, b, eps):
    x1 = a + (1 - tau) * (b - a)
    x2 = a + tau * (b - a)
    f1, f2 = F(x1), F(x2)
    rows = []
    k = 1
    while True:
        L = b - a
        row = [k, a, b, x1, x2, f1, f2, L, ""]
        rows.append(row)
        if L <= eps:
            row[8] = f"L = {L:.6f} ≤ ε = {eps} → СТОП"
            break
        if f1 <= f2:
            row[8] = "f₁ ≤ f₂ → b = x₂, успадковано x₂"
            b = x2
            x2, f2 = x1, f1
            x1 = a + (1 - tau) * (b - a)
            f1 = F(x1)
        else:
            row[8] = "f₁ > f₂ → a = x₁, успадковано x₁"
            a = x1
            x1, f1 = x2, f2
            x2 = a + tau * (b - a)
            f2 = F(x2)
        k += 1
    return rows, (a + b) / 2, b - a

def dichotomy(F, a, b, eps):
    k = 1
    while b - a > eps:
        m = (a + b) / 2
        delta_eps = eps / 4
        x1, x2 = m - delta_eps, m + delta_eps
        if F(x1) <= F(x2):
            b = x2
        else:
            a = x1
        k += 1
    return (a + b) / 2, b - a

def half_division(F, a, b, eps):
    xc = (a + b) / 2
    fc = F(xc)
    while b - a > eps:
        L = b - a
        x1, x2 = a + L / 4, b - L / 4
        f1, f2 = F(x1), F(x2)
        if f1 < fc:
            b, xc, fc = xc, x1, f1
        elif f2 < fc:
            a, xc, fc = xc, x2, f2
        else:
            a, b = x1, x2
    return xc, b - a

x_ex = (7 - 2 * math.sqrt(10)) / 9

print("Варіант 11: f(x) = (x - 1)^2 (x + 1)^4 (x - 2)^3")
print(f"x0 = {x0}, delta = {delta}, eps = {eps}")
print(f"Точний мінімум для перевірки: x = {x_ex:.6f}, f = {f(x_ex):.8f}\n")

S = Counter(f)
a0, b0 = svenn(S, x0, delta)
L0 = b0 - a0
print(f"Свенн: [a0, b0] = [{a0:.6f}, {b0:.6f}], L0 = {L0:.6f}, обчислень f(x): {S.nf}")
print(f"Мінімум в інтервалі: {a0 <= x_ex <= b0}\n")

G = Counter(f)
rows, xg, Lg = golden_section(G, a0, b0, eps)

print(f"{'k':>2} {'a_k':>9} {'b_k':>9} {'x1':>9} {'x2':>9} {'f(x1)':>12} {'f(x2)':>12} {'L_k':>9}  Рішення")
for r in rows:
    print(f"{r[0]:>2} {r[1]:9.6f} {r[2]:9.6f} {r[3]:9.6f} {r[4]:9.6f} "
          f"{r[5]:12.8f} {r[6]:12.8f} {r[7]:9.6f}  {r[8]}")

print(f"\nx* = ({rows[-1][1]:.6f} + {rows[-1][2]:.6f}) / 2 = {xg:.6f}, f(x*) = {f(xg):.8f}")
n_iter = len(rows) - 1
print(f"Ітерацій стиснення: {n_iter}, обчислень f(x): {G.nf}")

D = Counter(f)
xd, Ld = dichotomy(D, a0, b0, eps)
H = Counter(f)
xh, Lh = half_division(H, a0, b0, eps)

print("\nПорівняння методів")
print(f"Свенн:           обчислень {S.nf}, інтервал L0 = {L0:.6f}")
for name, C, xs, Ln, th in [
    ("Золотий переріз", G, xg, Lg, tau),
    ("Дихотомія", D, xd, Ld, 1 / math.sqrt(2)),
    ("Половинний поділ", H, xh, Lh, 1 / math.sqrt(2)),
]:
    eta = (Ln / L0) ** (1 / C.nf) if C.nf > 0 else 0
    print(f"{name + ':':<17} обчислень {C.nf:2d}, x* = {xs:.6f}, похибка {abs(xs - x_ex):.6f}, "
          f"η = {eta:.4f} (теор. {th:.4f})")
