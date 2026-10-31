import numpy as np

def F(x):
    return 2*x[0]**2 + 2*x[1]**2 - x[0]*x[1] + x[0] + 11*x[1]

def grad(x):
    return np.array([4*x[0] - x[1] + 1, 4*x[1] - x[0] + 11])

H = np.array([[4.0, -1.0], [-1.0, 4.0]])

def svenn(phi):
    delta = 0.01
    f0 = phi(0.0)
    h1 = delta
    f1 = phi(h1)
    if f1 < f0:
        h_prev, h_curr = 0.0, h1
        f_curr = f1
        step = delta
        while True:
            step *= 2.0
            h_next = h_curr + step
            f_next = phi(h_next)
            if f_next >= f_curr:
                return h_prev, h_next
            h_prev = h_curr
            h_curr = h_next
            f_curr = f_next
    else:
        return 0.0, delta

def golden(phi, a, b, eps=1e-6):
    tau = (np.sqrt(5) - 1) / 2
    x1 = b - tau * (b - a)
    x2 = a + tau * (b - a)
    f1, f2 = phi(x1), phi(x2)
    while (b - a) > eps:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - tau * (b - a)
            f1 = phi(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + tau * (b - a)
            f2 = phi(x2)
    return (a + b) / 2

def line_search(F, x, d):
    calls = 0
    def phi(h):
        nonlocal calls
        calls += 1
        return F(x + h * d)
    a, b = svenn(phi)
    h = golden(phi, a, b)
    return h, calls

def fr_method(F, grad, x0, eps=0.01, n=2, restart=True):
    x = np.array(x0, float)
    g = grad(x)
    d = -g
    k = 0
    fx = F(x)
    nf = 1
    nvf = 1
    hist = []
    while np.linalg.norm(g) > eps:
        if np.dot(g, d) >= 0:
            d = -g.copy()
        hist.append([k, x.copy(), fx, g.copy(), np.linalg.norm(g), d.copy(), None, None])
        h, line_nf = line_search(F, x, d)
        nf += line_nf
        hist[-1][6] = h
        x = x + h*d
        fx = F(x)
        nf += 1
        g_new = grad(x)
        nvf += 1
        beta = (g_new @ g_new) / (g @ g)
        if restart and (k + 1) % n == 0:
            beta = 0.0
        hist[-1][7] = beta
        d = -g_new + beta*d
        g = g_new
        k += 1
    hist.append([k, x.copy(), fx, g.copy(), np.linalg.norm(g), None, None, None])
    return x, k, nf, nvf, hist

def sd_method(F, grad, x0, eps=0.01):
    x = np.array(x0, float)
    g = grad(x)
    k = 0
    fx = F(x)
    nf = 1
    nvf = 1
    hist = []
    while np.linalg.norm(g) > eps:
        hist.append([k, x.copy(), fx])
        d = -g
        h, line_nf = line_search(F, x, d)
        nf += line_nf
        x = x + h*d
        fx = F(x)
        nf += 1
        g = grad(x)
        nvf += 1
        k += 1
        if k > 1000:
            break
    hist.append([k, x.copy(), fx])
    return x, k, nf, nvf, hist

def rosen(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_rosen(x):
    return np.array([-400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]), 200*(x[1] - x[0]**2)])

print("Аналітичний мінімум")
print("Функція F(x1, x2) = 2*x1^2 + 2*x2^2 - x1*x2 + x1 + 11*x2")
print("Градієнт: (4*x1 - x2 + 1, 4*x2 - x1 + 11)")
print("Гесіан H = [[4, -1], [-1, 4]]")
print("Мінімум: x* = (-1.0, -3.0), F(x*) = -17.0\n")

print("Перевірка лінійного пошуку")
x0 = np.array([3.0, -6.0])
d0 = np.array([-19.0, 16.0])
phi = lambda h: F(x0 + h*d0)
a, b = svenn(phi)
h_exact = golden(phi, a, b)
h_analytical = -np.dot(grad(x0), d0) / (d0 @ H @ d0)
print(f"h_чисельне (Свенн + золотий переріз) = {h_exact:.8f}")
print(f"h_аналітичне = {h_analytical:.8f}\n")

print("Метод Флетчера-Рівса")
res_fr = fr_method(F, grad, [3.0, -6.0])
print(f"{'k':>2} | {'x_(k)':>24} | {'F(x_(k))':>10} | {'grad F':>24} | {'||grad F||':>10} | {'d_(k)':>24} | {'h_k':>10} | {'beta_k+1':>10}")
print("-" * 140)
for row in res_fr[4]:
    k, x, fx, g, ng, d, h, beta = row
    x_str = f"({x[0]:8.5f}; {x[1]:8.5f})" if x is not None else ""
    g_str = f"({g[0]:8.5f}; {g[1]:8.5f})" if g is not None else ""
    d_str = f"({d[0]:8.5f}; {d[1]:8.5f})" if d is not None else ""
    h_str = f"{h:10.6f}" if h is not None else ""
    beta_str = f"{beta:10.6f}" if beta is not None else ""
    print(f"{k:2d} | {x_str:>24} | {fx:10.6f} | {g_str:>24} | {ng:10.6f} | {d_str:>24} | {h_str:>10} | {beta_str:>10}")
print()

print("Перевірки:")
x1 = res_fr[4][1][1]
d0 = np.array([-19.0, 16.0])
g1 = grad(x1)
d1 = res_fr[4][1][5]
print(f"1) <grad F(x_(1)), d_(0)> = {np.dot(g1, d0):.2e} (має бути 0)")
print(f"2) d_(1)^T H d_(0) = {d1 @ H @ d0:.2e} (має бути 0)")
print(f"3) <grad F(x_(1)), d_(1)> = {np.dot(g1, d1):.6f} (має бути < 0)")
print(f"Кількість ітерацій: {res_fr[1]}\n")

print("Антиградієнт + точний пошук (beta=0)")
res_sd = sd_method(F, grad, [3.0, -6.0])
print(f"{'k':>2} | {'x_(k)':>24} | {'F(x_(k))':>10}")
print("-" * 45)
for row in res_sd[4]:
    k, x, fx = row
    print(f"{k:2d} | ({x[0]:8.5f}; {x[1]:8.5f}) | {fx:10.6f}")
print(f"Всього ітерацій: {res_sd[1]}\n")

print("Функція Розенброка")
res_ros_restart = fr_method(rosen, grad_rosen, [-1.2, 1.0], restart=True)
res_ros_no_restart = fr_method(rosen, grad_rosen, [-1.2, 1.0], restart=False)
print(f"З перезапуском (n=2): ітерацій={res_ros_restart[1]}, NF={res_ros_restart[2]}, Nvf={res_ros_restart[3]}")
print(f"Без перезапуску: ітерацій={res_ros_no_restart[1]}, NF={res_ros_no_restart[2]}, Nvf={res_ros_no_restart[3]}\n")

print("Порівняння обчислень")
print(f"Антиградієнт із точним лінійним пошуком: ітерацій={res_sd[1]}, NF={res_sd[2]}, N∇F={res_sd[3]}")
print(f"Флетчер–Рівс: ітерацій={res_fr[1]}, NF={res_fr[2]}, N∇F={res_fr[3]}")
