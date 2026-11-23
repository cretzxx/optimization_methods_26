import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


STUDENT_ID = 11
SEED = (STUDENT_ID * 7919 + 104729) % (2**31 - 1)

NOISE_SIGMA = 0.0671
X0 = 0.5593
DELTA0 = 0.1978
EPS = 1e-5
LAMBDA0 = 0.4145
SIMPLEX_EDGE = 0.2707
SIZE = 128

GAUSSIAN_MIN = 0.05
GAUSSIAN_MAX = 8.0
BILATERAL_LO = (0.3, 0.01)
BILATERAL_HI = (5.0, 1.0)


MASK64 = (1 << 64) - 1

class Rng:

    def __init__(self, seed):
        self.state = int(seed) & MASK64

    def next_u64(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & MASK64
        return self.state

    def uniform(self):
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def normal(self):
        u1 = max(self.uniform(), 1e-300)
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

def clean_image(size=128):
    image = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            u = j / (size - 1)
            v = i / (size - 1)
            value = 0.25 + 0.35 * (0.6 * u + 0.4 * v)
            value += 0.08 * math.cos(
                2.0 * math.pi * math.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2)
            )
            if (u - 0.32) ** 2 + (v - 0.34) ** 2 <= 0.16**2:
                value = 0.92
            if abs(u - 0.70) <= 0.15 and abs(v - 0.30) <= 0.15:
                value = 0.10
            if 0.60 < v < 0.90 and abs(u - 0.30) < 0.9 * (v - 0.60):
                value = 0.68
            if 0.58 < u < 0.92 and 0.60 < v < 0.90:
                cell = max(3, size // 12)
                value = 0.85 if ((j // cell) + (i // cell)) % 2 == 0 else 0.20
            image[i, j] = min(max(value, 0.0), 1.0)

    for index in range(4):
        thickness = max(2, (4 - index) * max(1, size // 64))
        row = int(0.04 * size) + index * max(3, size // 22)
        if row + thickness < size:
            image[row:row + thickness, int(0.05 * size):int(0.95 * size)] = 0.98
    return image

def add_noise(clean, sigma, seed):
    rng = Rng(seed)
    raw = np.zeros_like(clean, dtype=float)
    for i in range(clean.shape[0]):
        for j in range(clean.shape[1]):
            raw[i, j] = clean[i, j] + sigma * rng.normal()
    return raw, np.clip(raw, 0.0, 1.0)

def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))

def psnr(a, b):
    value = mse(a, b)
    return float("inf") if value <= 0.0 else 10.0 * math.log10(1.0 / value)

def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)

class Oracle:

    def __init__(self, function):
        self.function = function
        self.n_f = 0
        self.history = []

    def value(self, x):
        self.n_f += 1
        value = float(self.function(x))
        if isinstance(x, np.ndarray):
            saved_x = x.copy()
        elif isinstance(x, (list, tuple)):
            saved_x = tuple(float(v) for v in x)
        else:
            saved_x = float(x)
        self.history.append((saved_x, value))
        return value

    def peek(self, x):
        return float(self.function(x))

    def reset(self):
        self.n_f = 0
        self.history.clear()

def write_csv(path, rows, fields=None):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fields is None:
        fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean_row = {}
            for key, value in row.items():
                if isinstance(value, np.ndarray):
                    value = ";".join(f"{v:.10g}" for v in value.ravel())
                clean_row[key] = value
            writer.writerow(clean_row)


def gaussian_blur(image, sigma):
    if sigma <= 1e-6:
        return image.copy()
    radius = int(4.0 * sigma + 0.5)
    if radius == 0:
        return image.copy()
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets.astype(float) ** 2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    height, width = image.shape

    temp = np.zeros_like(image)
    for offset in offsets:
        columns = np.clip(np.arange(width) + offset, 0, width - 1)
        temp += kernel[offset + radius] * image[:, columns]

    result = np.zeros_like(image)
    for offset in offsets:
        rows = np.clip(np.arange(height) + offset, 0, height - 1)
        result += kernel[offset + radius] * temp[rows, :]
    return result

def bilateral_filter(image, sigma_spatial, sigma_range):
    sigma_spatial = max(float(sigma_spatial), 1e-3)
    sigma_range = max(float(sigma_range), 1e-3)
    radius = int(min(max(round(2.0 * sigma_spatial), 1), 5))
    height, width = image.shape
    padded = np.pad(image, radius, mode="reflect")
    numerator = np.zeros_like(image)
    denominator = np.zeros_like(image)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = padded[radius + dy:radius + dy + height,
                             radius + dx:radius + dx + width]
            spatial_weight = math.exp(
                -(dx * dx + dy * dy) / (2.0 * sigma_spatial * sigma_spatial)
            )
            range_weight = np.exp(
                -((shifted - image) ** 2) / (2.0 * sigma_range * sigma_range)
            )
            weight = spatial_weight * range_weight
            numerator += weight * shifted
            denominator += weight
    return numerator / denominator


def svenn(oracle, x0, delta0, lo=0.05, hi=8.0, max_iter=40):
    rows = []

    def ask(x, k, step, direction, comment):
        value = oracle.value(x)
        rows.append({"k": k, "x": x, "f": value, "delta": step,
                     "p": direction, "comment": comment})
        return value

    f0 = ask(x0, 0, delta0, 0, "початкова точка")
    xr = min(x0 + delta0, hi)
    fr = ask(xr, 0, delta0, 0, "пробний крок праворуч")
    xl = max(x0 - delta0, lo)
    fl = ask(xl, 0, -delta0, 1, "пробний крок ліворуч")

    if fl >= f0 <= fr:
        return {"a": min(xl, xr), "b": max(xl, xr), "p": 0,
                "iterations": 0, "n_f": oracle.n_f, "rows": rows}
    if fr <= f0:
        direction, step, x_cur, f_cur = 0, delta0, xr, fr
        rows[-1]["comment"] += " -> рух праворуч"
    else:
        direction, step, x_cur, f_cur = 1, -delta0, xl, fl
        rows[-1]["comment"] += " -> рух ліворуч"

    x_prev = x0
    x_new = x_cur
    k = 1
    while k <= max_iter:
        x_new = min(max(x_cur + (2.0 ** k) * step, lo), hi)
        if x_new == x_cur:
            break
        f_new = ask(x_new, k, (2.0 ** k) * step, direction, f"розгін: крок {k}")
        if f_new >= f_cur:
            rows[-1]["comment"] += " -> функція зросла, зупинка"
            break
        x_prev, x_cur, f_cur = x_cur, x_new, f_new
        k += 1

    a, b = min(x_prev, x_new), max(x_prev, x_new)
    return {"a": max(a, lo), "b": min(b, hi), "p": direction,
            "iterations": k, "n_f": oracle.n_f, "rows": rows}

def dichotomy(oracle, a, b, eps=1e-5, delta=None, max_iter=500):
    if delta is None:
        delta = eps / 3.0
    if not 0.0 < delta < eps:
        raise ValueError("Для дихотомії має виконуватися 0 < δ < ε")
    rows = []
    k = 0
    while b - a > eps and k < max_iter:
        x1 = (a + b - delta) / 2.0
        x2 = (a + b + delta) / 2.0
        f1 = oracle.value(x1)
        f2 = oracle.value(x2)
        if f1 <= f2:
            b = x2
            decision = "b = x2"
        else:
            a = x1
            decision = "a = x1"
        k += 1
        rows.append({"k": k, "a": a, "b": b, "x1": x1, "x2": x2,
                     "f1": f1, "f2": f2, "L": b - a, "decision": decision})
    x_best = (a + b) / 2.0
    return {"x": x_best, "f": oracle.value(x_best), "a": a, "b": b,
            "iterations": k, "n_f": oracle.n_f, "rows": rows}

def golden_section(oracle, a, b, eps=1e-5, max_iter=500):
    tau = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = b - tau * (b - a)
    x2 = a + tau * (b - a)
    f1 = oracle.value(x1)
    f2 = oracle.value(x2)
    rows = []
    k = 0

    while b - a > eps and k < max_iter:
        if f1 <= f2:
            b = x2
            x2, f2 = x1, f1
            x1 = b - tau * (b - a)
            f1 = oracle.value(x1)
            decision = "b = x2; успадковано x1"
        else:
            a = x1
            x1, f1 = x2, f2
            x2 = a + tau * (b - a)
            f2 = oracle.value(x2)
            decision = "a = x1; успадковано x2"
        k += 1
        rows.append({"k": k, "a": a, "b": b, "x1": x1, "x2": x2,
                     "f1": f1, "f2": f2, "L": b - a, "decision": decision})

    x_best = (a + b) / 2.0
    return {"x": x_best, "f": oracle.value(x_best), "a": a, "b": b,
            "iterations": k, "n_f": oracle.n_f, "rows": rows}


def grad_xy(image):
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[:, :-1] = image[:, 1:] - image[:, :-1]
    gy[:-1, :] = image[1:, :] - image[:-1, :]
    return gx, gy

def divergence(px, py):
    result = np.zeros_like(px)
    result[:, 0] += px[:, 0]
    result[:, 1:-1] += px[:, 1:-1] - px[:, :-2]
    result[:, -1] -= px[:, -2]
    result[0, :] += py[0, :]
    result[1:-1, :] += py[1:-1, :] - py[:-2, :]
    result[-1, :] -= py[-2, :]
    return result

def laplacian(image):
    return divergence(*grad_xy(image))

def energy(image, observed, lam):
    gx, gy = grad_xy(image)
    fidelity = 0.5 * float(np.sum((image - observed) ** 2))
    smoothness = 0.5 * lam * float(np.sum(gx * gx + gy * gy))
    return fidelity + smoothness

def energy_grad(image, observed, lam):
    return image - observed - lam * laplacian(image)


def make_bilateral_objective(noisy, clean):
    cache = {}

    def objective(theta):
        point = np.asarray(theta, dtype=float)
        clamped = np.clip(point, BILATERAL_LO, BILATERAL_HI)
        key = (float(clamped[0]), float(clamped[1]))
        if key not in cache:
            cache[key] = mse(bilateral_filter(noisy, *clamped), clean)
        gap = point - clamped
        return cache[key] + 1000.0 * float(gap @ gap)

    return objective

def nelder_mead(oracle, x0, step, tol=1e-7, max_iter=200):
    x0 = np.asarray(x0, dtype=float)
    lo = np.asarray(BILATERAL_LO)
    hi = np.asarray(BILATERAL_HI)
    simplex = [x0,
               np.clip(x0 + np.array([step, 0.0]), lo, hi),
               np.clip(x0 + np.array([0.0, 0.4 * step]), lo, hi)]
    values = [oracle.value(point) for point in simplex]
    rows = []

    for k in range(1, max_iter + 1):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        best, middle, worst = simplex
        f_best, f_middle, f_worst = values
        centroid = (best + middle) / 2.0
        reflected = np.clip(centroid + (centroid - worst), lo, hi)
        f_reflected = oracle.value(reflected)

        if f_reflected < f_best:
            expanded = np.clip(centroid + 2.0 * (reflected - centroid), lo, hi)
            f_expanded = oracle.value(expanded)
            if f_expanded < f_reflected:
                simplex[2], values[2], operation = expanded, f_expanded, "розтягування"
            else:
                simplex[2], values[2], operation = reflected, f_reflected, "відображення"
        elif f_reflected < f_middle:
            simplex[2], values[2], operation = reflected, f_reflected, "відображення"
        else:
            if f_reflected < f_worst:
                contracted = np.clip(centroid + 0.5 * (reflected - centroid), lo, hi)
                operation = "зовнішнє стиснення"
            else:
                contracted = np.clip(centroid + 0.5 * (worst - centroid), lo, hi)
                operation = "внутрішнє стиснення"
            f_contracted = oracle.value(contracted)
            if f_contracted < min(f_reflected, f_worst):
                simplex[2], values[2] = contracted, f_contracted
            else:
                for index in (1, 2):
                    simplex[index] = np.clip(best + 0.5 * (simplex[index] - best), lo, hi)
                    values[index] = oracle.value(simplex[index])
                operation = "редукція"

        best_index = int(np.argmin(values))
        size = max(float(np.linalg.norm(point - simplex[best_index])) for point in simplex)
        rows.append({"k": k, "operation": operation, "x": simplex[best_index].copy(),
                     "f": values[best_index], "size": size, "n_f": oracle.n_f})
        if size <= tol and max(values) - min(values) <= tol:
            break

    best_index = int(np.argmin(values))
    return {"x": simplex[best_index], "f": values[best_index],
            "iterations": k, "n_f": oracle.n_f, "rows": rows}

def line_minimize(oracle, point, direction, delta=0.1, eps=1e-4, alpha_max=1000.0):
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    lo = np.asarray(BILATERAL_LO)
    hi = np.asarray(BILATERAL_HI)
    line_oracle = Oracle(lambda alpha: oracle.value(np.clip(point + alpha * direction, lo, hi)))
    bracket = svenn(line_oracle, 0.0, delta, -alpha_max, alpha_max)
    result = golden_section(line_oracle, bracket["a"], bracket["b"], eps)
    alpha = result["x"]
    new_point = np.clip(point + alpha * direction, lo, hi)
    value = oracle.value(new_point)
    calls = {"svenn": bracket["n_f"],
             "golden": result["n_f"] - bracket["n_f"],
             "final_check": 1}
    return new_point, value, alpha, calls

def powell(oracle, x0, tol=1e-4, max_cycles=50):
    point = np.asarray(x0, dtype=float)
    value = oracle.value(point)
    directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    rows = []
    line_costs = []
    line_details = []

    for cycle in range(1, max_cycles + 1):
        start = point.copy()
        start_value = value
        gains = []
        alphas = []
        cycle_costs = []
        cycle_details = []
        for index in range(2):
            before = oracle.n_f
            point_new, value_new, alpha, calls = line_minimize(
                oracle, point, directions[index]
            )
            gains.append(value - value_new)
            alphas.append(alpha)
            point, value = point_new, value_new
            cycle_costs.append(oracle.n_f - before)
            cycle_details.append(calls)

        shift = point - start
        shift_norm = float(np.linalg.norm(shift))
        rows.append({"cycle": cycle, "x": point.copy(), "f": value,
                     "shift": shift_norm, "alpha1": alphas[0],
                     "alpha2": alphas[1], "n_f": oracle.n_f})
        line_costs.append(cycle_costs)
        line_details.append(cycle_details)
        if shift_norm < tol or abs(start_value - value) <= 1e-12 * (1.0 + abs(value)):
            break

        new_direction = shift / shift_norm
        before = oracle.n_f
        point, value, alpha, calls = line_minimize(
            oracle, point, new_direction
        )
        line_costs[-1].append(oracle.n_f - before)
        line_details[-1].append(calls)
        directions[int(np.argmax(gains))] = new_direction
        if cycle % 3 == 0:
            directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    return {"x": point, "f": value, "cycles": cycle, "n_f": oracle.n_f,
            "rows": rows, "line_costs": line_costs,
            "line_details": line_details}

def best_spatial_with_fixed_range(oracle, spatial_start, range_value):
    one_dimensional = Oracle(lambda s: oracle.value(np.array([s, range_value])))
    bracket = svenn(one_dimensional, spatial_start, 0.1, 0.3, 5.0)
    result = golden_section(one_dimensional, bracket["a"], bracket["b"], 1e-4)
    return {"svenn": bracket, "golden": result,
            "n_f": oracle.n_f}


def tikhonov_solve(observed, lam, iterations=300):
    lipschitz = 1.0 + 8.0 * lam
    step = 1.0 / lipschitz
    image = observed.copy()
    for _ in range(iterations):
        image -= step * energy_grad(image, observed, lam)
    return image

def find_lambda(noisy, clean):
    objective = lambda lam: mse(tikhonov_solve(noisy, lam, 300), clean)
    oracle = Oracle(objective)
    bracket = svenn(oracle, LAMBDA0, DELTA0, 1e-4, 50.0)
    bracket_calls = oracle.n_f
    oracle.reset()
    result = golden_section(oracle, bracket["a"], bracket["b"], 1e-3)
    return {"lambda": result["x"], "f": result["f"], "bracket": bracket,
            "svenn_calls": bracket_calls, "golden": result}

def gradient_method(observed, lam, kind, max_iter=200, tol=1e-6, reference=None):
    lipschitz = 1.0 + 8.0 * lam
    mu = 1.0
    step = 1.0 / lipschitz
    beta_hb = ((math.sqrt(lipschitz) - math.sqrt(mu)) /
               (math.sqrt(lipschitz) + math.sqrt(mu))) ** 2
    beta_nag = ((math.sqrt(lipschitz) - math.sqrt(mu)) /
                (math.sqrt(lipschitz) + math.sqrt(mu)))
    if kind == "hb_opt":
        beta = beta_hb
    elif kind == "hb_09":
        beta = 0.9
    elif kind == "nag_strong":
        beta = beta_nag
    else:
        beta = None

    image = observed.copy()
    previous = image.copy()
    trial = step
    rows = []
    n_f = 0
    n_grad = 0
    converged = False

    for k in range(1, max_iter + 1):
        if kind in ("nag_convex", "nag_strong"):
            momentum = (k - 1) / (k + 2) if kind == "nag_convex" else beta
            point = image + momentum * (image - previous)
            grad = energy_grad(point, observed, lam)
            n_grad += 1
            grad_norm = float(np.linalg.norm(grad))
            new_image = point - step * grad
        else:
            grad = energy_grad(image, observed, lam)
            n_grad += 1
            grad_norm = float(np.linalg.norm(grad))
            if kind == "armijo":
                value_here = energy(image, observed, lam)
                n_f += 1
                grad_sq = float(np.sum(grad * grad))
                trial_step = trial
                for _ in range(40):
                    candidate = image - trial_step * grad
                    candidate_value = energy(candidate, observed, lam)
                    n_f += 1
                    if candidate_value <= value_here - 1e-4 * trial_step * grad_sq:
                        break
                    trial_step *= 0.5
                new_image = candidate
                trial = trial_step * 1.5
                step = trial_step
            elif kind in ("hb_opt", "hb_09"):
                new_image = image - step * grad + beta * (image - previous)
            else:
                new_image = image - step * grad

        current_energy = energy(image, observed, lam)
        n_f += 1
        distance = (float(np.linalg.norm(image - reference))
                    if reference is not None else None)
        rows.append({"k": k - 1, "E": current_energy, "grad": grad_norm,
                     "h": step, "dist": distance,
                     "n_f": n_f, "n_grad": n_grad})
        if not np.isfinite(grad_norm) or not np.isfinite(current_energy):
            break
        if grad_norm <= tol:
            converged = True
            break
        previous, image = image, new_image

    return {"image": image, "iterations": max(0, len(rows) - 1),
            "steps_run": len(rows), "n_f": n_f, "n_grad": n_grad,
            "energy": energy(image, observed, lam),
            "gradient_norm": rows[-1]["grad"] if rows else float("inf"),
            "converged": converged, "rows": rows,
            "beta_hb": beta_hb, "beta_nag": beta_nag}

def full_variation_energy(image, observed, lam_tv, epsilon):
    gx, gy = grad_xy(image)
    variation = np.sqrt(gx * gx + gy * gy + epsilon * epsilon)
    fidelity = 0.5 * float(np.sum((image - observed) ** 2))
    return fidelity + lam_tv * float(np.sum(variation))

def full_variation_gradient(image, observed, lam_tv, epsilon):
    gx, gy = grad_xy(image)
    denominator = np.sqrt(gx * gx + gy * gy + epsilon * epsilon)
    return image - observed - lam_tv * divergence(gx / denominator, gy / denominator)

def isotropic_total_variation(image):
    gx, gy = grad_xy(image)
    return float(np.sum(np.sqrt(gx * gx + gy * gy)))

def full_variation_solve(observed, lam_tv, epsilon=5e-3,
                         max_iter=200, tol=1e-6):
    if lam_tv <= 0.0 or epsilon <= 0.0:
        raise ValueError("lambda_tv та epsilon мають бути додатними")

    lipschitz = 1.0 + 8.0 * lam_tv / epsilon
    mu = 1.0
    step = 1.0 / lipschitz
    beta = (math.sqrt(lipschitz) - math.sqrt(mu)) / (
        math.sqrt(lipschitz) + math.sqrt(mu)
    )

    image = observed.copy()
    previous = image.copy()
    rows = []
    converged = False
    for k in range(max_iter):
        point = image + beta * (image - previous)
        gradient = full_variation_gradient(point, observed, lam_tv, epsilon)
        gradient_norm = float(np.linalg.norm(gradient))
        new_image = point - step * gradient
        value = full_variation_energy(new_image, observed, lam_tv, epsilon)
        rows.append({"k": k, "E": value, "gradient_norm": gradient_norm})

        if not np.isfinite(gradient_norm) or not np.isfinite(value):
            break
        if gradient_norm <= tol:
            image = new_image
            converged = True
            break
        previous, image = image, new_image

    steps_run = len(rows)
    return {
        "image": image,
        "lambda_tv": lam_tv,
        "epsilon": epsilon,
        "L": lipschitz,
        "mu": mu,
        "kappa": lipschitz / mu,
        "step": step,
        "beta": beta,
        "iterations": max(0, steps_run - 1),
        "steps_run": steps_run,
        "n_f": steps_run,
        "n_grad": steps_run,
        "energy": full_variation_energy(image, observed, lam_tv, epsilon),
        "gradient_norm": rows[-1]["gradient_norm"] if rows else float("inf"),
        "converged": converged,
        "rows": rows,
        "MSE": None,
        "PSNR": None,
        "isotropic_TV": isotropic_total_variation(image),
    }

def operator_checks():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(9, 7))
    px = rng.normal(size=(9, 7))
    py = rng.normal(size=(9, 7))
    gx, gy = grad_xy(u)
    left = float(np.sum(gx * px + gy * py))
    right = -float(np.sum(u * divergence(px, py)))
    adjoint_error = abs(left - right)

    image = rng.random((16, 16))
    direction = rng.normal(size=(16, 16))
    observed = rng.random((16, 16))
    lam = 0.4
    step = 1e-6
    numeric = (energy(image + step * direction, observed, lam) -
               energy(image - step * direction, observed, lam)) / (2.0 * step)
    analytic = float(np.sum(energy_grad(image, observed, lam) * direction))
    relative_error = abs(numeric - analytic) / max(abs(analytic), 1e-15)
    return {"adjoint_error": adjoint_error, "finite_difference_relative_error": relative_error}


def project(image, lower=0.0, upper=1.0):
    return np.clip(image, lower, upper)

def violation(image, lower=0.0, upper=1.0):
    below = np.maximum(0.0, lower - image)
    above = np.maximum(0.0, image - upper)
    return {"max": float(max(np.max(below), np.max(above))),
            "count": int(np.count_nonzero((image < lower) | (image > upper)))}

def projected_gradient(raw, lam, max_iter=200, tol=1e-6, use_gradient_stop=False):
    lipschitz = 1.0 + 8.0 * lam
    step = 1.0 / lipschitz
    image = project(raw.copy())
    previous = image.copy()
    extrapolated = image.copy()
    rows = []
    converged = False

    for k in range(1, max_iter + 1):
        point = extrapolated if use_gradient_stop == "fista" else image
        grad = energy_grad(point, raw, lam)
        new_image = project(point - step * grad)
        gradient_map = float(np.linalg.norm(new_image - point) / step)
        grad_norm = float(np.linalg.norm(grad))
        v = violation(image)
        current_energy = energy(image, raw, lam)
        rows.append({"k": k - 1, "E": current_energy, "G": gradient_map,
                     "grad": grad_norm, "active": int(np.count_nonzero(
                         (image <= 1e-12) | (image >= 1.0 - 1e-12))),
                     "violation": v["max"]})

        if (grad_norm if use_gradient_stop is True else gradient_map) <= tol:
            converged = True
            image = new_image
            break

        if use_gradient_stop == "fista":
            beta = (k - 1) / (k + 2)
            extrapolated = new_image + beta * (new_image - previous)
            previous = new_image
        image = new_image

    return {"image": image, "iterations": max(0, len(rows) - 1),
            "steps_run": len(rows), "n_f": len(rows), "n_grad": len(rows),
            "energy": energy(image, raw, lam), "G": rows[-1]["G"],
            "grad": rows[-1]["grad"], "converged": converged, "rows": rows}

def penalty_value(image, lower=0.0, upper=1.0):
    below = np.maximum(0.0, lower - image)
    above = np.maximum(0.0, image - upper)
    return float(np.sum(below * below + above * above))

def penalty_grad(image, lower=0.0, upper=1.0):
    return (-2.0 * np.maximum(0.0, lower - image) +
            2.0 * np.maximum(0.0, image - upper))

def penalty_method(raw, lam, growth=10.0, outer=7, inner=100,
                   inner_tol=1e-6, tol=1e-8):
    lipschitz = 1.0 + 8.0 * lam
    image = raw.copy()
    r = 1.0
    n_f = 0
    n_grad = 0
    rows = []

    for outer_index in range(outer):
        penalized_lipschitz = lipschitz + 2.0 * r
        step = 1.0 / penalized_lipschitz
        previous = image.copy()
        inner_steps = 0
        for k in range(1, inner + 1):
            beta = (k - 1) / (k + 2)
            point = image + beta * (image - previous)
            grad = energy_grad(point, raw, lam) + r * penalty_grad(point)
            n_grad += 1
            new_image = point - step * grad
            inner_steps = k
            if float(np.linalg.norm(grad)) <= inner_tol:
                previous, image = image, new_image
                break
            previous, image = image, new_image

        penalty = penalty_value(image)
        v = violation(image)
        n_f += 1
        rows.append({"outer": outer_index, "r": r, "L_r": penalized_lipschitz,
                     "step": step, "inner_steps": inner_steps,
                     "E": energy(image, raw, lam), "P": penalty,
                     "rP": r * penalty, "violation": v["max"],
                     "outside": v["count"], "n_f": n_f, "n_grad": n_grad})
        if r * penalty <= tol:
            break
        r *= growth

    return {"image": image, "n_f": n_f, "n_grad": n_grad,
            "calls": n_f + n_grad, "energy": energy(image, raw, lam),
            "rows": rows}
