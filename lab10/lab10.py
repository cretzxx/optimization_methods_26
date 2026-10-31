import math

class FuncWithCounter:
    def __init__(self, func):
        self.func = func
        self.nf = 0
    def __call__(self, x):
        self.nf += 1
        return self.func(x)

F11_raw = lambda x: 2*x[0]**2 + 2*x[1]**2 - x[0]*x[1] + x[0] + 11*x[1]
Frosen_raw = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
FG_raw = lambda x: 2*abs(x[0] - 1) + 3*abs(x[1] + 2) + 0.5*(x[0] - 1)**2

def explore(F, y, Fy, d):
    z = list(y)
    Fz = Fy
    for i in range(len(z)):
        t = list(z)
        t[i] += d
        Ft = F(t)
        if Ft < Fz:
            z = t
            Fz = Ft
            continue
        t = list(z)
        t[i] -= d
        Ft = F(t)
        if Ft < Fz:
            z = t
            Fz = Ft
    return z, Fz

def hooke_jeeves(F, x0, d0=1.0, eps=0.01, verbose=False):
    base = list(x0)
    Fb = F(base)
    y, Fy, d, from_pattern = list(base), Fb, d0, False
    k = 0
    
    if verbose:
        print(f"{'k':>2} | {'Дія':<22} | {'Звідки':<15} | {'F':>8} | {'Δ':>6} | {'Куди':<15} | {'F':>8} | {'Nf':>3}")
        print("-" * 95)

    while True:
        z, Fz = explore(F, y, Fy, d)
        
        if Fz < Fb:
            y = [z[0] + (z[0] - base[0]), z[1] + (z[1] - base[1])]
            base, Fb, from_pattern = list(z), Fz, True
            Fy = F(y)
            if verbose:
                print(f"{k:2d} | {'нова база → зразок':<22} | ({z[0]:5.2f};{z[1]:5.2f}) | {Fz:8.4f} | {d:6.4f} | ({y[0]:5.2f};{y[1]:5.2f}) | {Fy:8.4f} | {F.nf}")
            k += 1
        elif from_pattern:
            y, Fy, from_pattern = list(base), Fb, False
            if verbose:
                print(f"{k:2d} | {'повернення в базу':<22} | ({z[0]:5.2f};{z[1]:5.2f}) | {Fz:8.4f} | {d:6.4f} | ({y[0]:5.2f};{y[1]:5.2f}) | {Fy:8.4f} | {F.nf}")
        else:
            d /= 2
            if verbose:
                print(f"{k:2d} | {'дроблення Δ':<22} | ({z[0]:5.2f};{z[1]:5.2f}) | {Fz:8.4f} | {d:6.4f} | ({z[0]:5.2f};{z[1]:5.2f}) | {Fz:8.4f} | {F.nf}")
            if d < eps:
                if verbose:
                    print("-" * 95)
                return base, Fb, k, F.nf

print("Аналітичний мінімум")
print("F(x) = 2x1^2 + 2x2^2 - x1x2 + x1 + 11x2")
print("Мінімум: x* = (-1.0, -3.0), F(x*) = -17.0\n")

print("Хук-Дживс")
F11_c1 = FuncWithCounter(F11_raw)
base1, Fb1, k1, nf1 = hooke_jeeves(F11_c1, [3.0, -6.0], d0=1.0, eps=0.01, verbose=True)
print(f"\nРезультат: x* = ({base1[0]:.5f}, {base1[1]:.5f}), F = {Fb1:.5f}")
print(f"Кількість базових точок (k): {k1}, Всього обчислень F (Nf): {nf1}")
err1 = math.sqrt((base1[0] - (-1.0))**2 + (base1[1] - (-3.0))**2)
print(f"Похибка: {err1:.2e}\n")

print("Прогін з Δ0 = 0.7, ε = 0.01")
F11_c2 = FuncWithCounter(F11_raw)
base2, Fb2, k2, nf2 = hooke_jeeves(F11_c2, [3.0, -6.0], d0=0.7, eps=0.01, verbose=False)
err2 = math.sqrt((base2[0] - (-1.0))**2 + (base2[1] - (-3.0))**2)
print(f"Результат: x* = ({base2[0]:.5f}, {base2[1]:.5f}), F = {Fb2:.5f}")
print(f"Кількість базових точок (k): {k2}, Всього обчислень F (Nf): {nf2}")
print(f"Похибка: {err2:.2e}")
print("Пояснення: при Δ0 = 1 точка мінімуму (-1, -3) лежить на цілочисельній сітці, тому метод влучає точно. При Δ0 = 0.7 сітка зсувається, і метод зупиняється, коли крок Δ стає меншим за ε, тому виникає похибка.\n")

print("Негладка функція G")
FG_c = FuncWithCounter(FG_raw)
baseG, FbG, kG, nfG = hooke_jeeves(FG_c, [5.0, -3.0], d0=1.0, eps=0.01, verbose=False)
print(f"Результат: x* = ({baseG[0]:.5f}, {baseG[1]:.5f}), G = {FbG:.5f}")
print(f"Кількість базових точок (k): {kG}, Всього обчислень G (Nf): {nfG}")
print("Пояснення: для G(x) = 2|x1-1| + 3|x2+2| + 0.5(x1-1)^2 градієнт у точці мінімуму не існує (злам). Тому методи ЛР 8-9 зупиняються або 'стрибають' навколо мінімуму. Хук-Дживс просто порівнює значення F і успішно знаходить (1, -2).\n")

print("Функція Розенброка")
Frosen_c1 = FuncWithCounter(Frosen_raw)
baseR1, FbR1, kR1, nfR1 = hooke_jeeves(Frosen_c1, [-1.2, 1.0], d0=1.0, eps=0.01, verbose=False)
print(f"ε = 0.01: x* = ({baseR1[0]:.5f}, {baseR1[1]:.5f}), F = {FbR1:.5f}, Nf = {nfR1}")
print("Пастка: метод зупинився далеко від (1, 1), бо крок Δ став меншим за ε через вигнуту долину яру.")

Frosen_c2 = FuncWithCounter(Frosen_raw)
baseR2, FbR2, kR2, nfR2 = hooke_jeeves(Frosen_c2, [-1.2, 1.0], d0=1.0, eps=1e-4, verbose=False)
print(f"ε = 1e-4: x* = ({baseR2[0]:.5f}, {baseR2[1]:.5f}), F = {FbR2:.2e}, Nf = {nfR2}")
print("зменшення ε дозволяє методу подолати вигнутість і дійти до справжнього мінімуму (1, 1).\n")