import numpy as np

def F(x):
    return 2*x[0]**2 + 2*x[1]**2 - x[0]*x[1] + x[0] + 11*x[1]

def grad(x):
    return np.array([4*x[0] - x[1] + 1, 4*x[1] - x[0] + 11])

def optimize(x0, h0, metric, eps=0.01):
    x = np.array(x0, float)
    h = h0
    Fx = F(x)
    nf = 1
    nvf = 0
    k = 0
    frac = 0

    h_min = 1e-10
    while k < 5000:
        g = grad(x)
        nvf += 1
        if np.linalg.norm(g) <= eps:
            break
            
        if metric == "sphere":
            d = -g / np.linalg.norm(g)
        elif metric == "cube":
            d = -np.sign(g)
        elif metric == "octa":
            d = np.zeros_like(g)
            j = np.argmax(np.abs(g))
            d[j] = -np.sign(g[j])

        while True:
            xb = x + h * d
            Fb = F(xb)
            nf += 1
            if Fb < Fx:
                break
            h /= 2
            frac += 1
            if h < h_min:

                raise RuntimeError(
                    f"{metric}: крок став меншим за {h_min:g}, але градієнт ще не досяг ε"
                )
        
        x = xb
        Fx = Fb
        k += 1
        
    if k >= 5000:
        raise RuntimeError(f"{metric}: перевищено ліміт ітерацій без збіжності")
    return x, Fx, k, nf, nvf, frac

for h0 in [1.0, 0.7]:
    print(f"h0 = {h0}")
    for m in ["sphere", "cube", "octa"]:
        x, Fx, k, nf, nvf, frac = optimize([3.0, -6.0], h0, m)
        err = np.linalg.norm(x - np.array([-1.0, -3.0]))
        print(f"{m}: n={k}, Nf={nf}, Nvf={nvf}, Разом={nf+nvf}, x*={x[0]:.4f}, {x[1]:.4f}, F={Fx:.4f}, Похибка={err:.4f}, Дроблень={frac}")
