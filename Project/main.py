from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from methods import (
    BILATERAL_HI, BILATERAL_LO, DELTA0, EPS, LAMBDA0, NOISE_SIGMA,
    SEED, SIMPLEX_EDGE, SIZE, X0, Oracle, add_noise,
    best_spatial_with_fixed_range, bilateral_filter, clean_image,
    dichotomy, energy, energy_grad, find_lambda, full_variation_solve,
    gaussian_blur, golden_section, gradient_method,
    isotropic_total_variation, make_bilateral_objective, mse, nelder_mead,
    operator_checks, penalty_method, projected_gradient, psnr, powell,
    save_image, svenn, tikhonov_solve, violation, write_csv,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def run_lab1(noisy, clean):
    objective = lambda sigma: mse(gaussian_blur(noisy, sigma), clean)
    oracle = Oracle(objective)
    result = svenn(oracle, X0, DELTA0)
    write_csv(OUTPUTS / "lab1_svenn.csv", result["rows"])

    grid = np.linspace(0.05, 8.0, 200)
    values = [oracle.peek(sigma) for sigma in grid]
    plt.figure(figsize=(9, 5))
    plt.plot(grid, values, label="f(σ) = MSE")
    plt.scatter([row["x"] for row in result["rows"]],
                [row["f"] for row in result["rows"]],
                color="red", zorder=4, label="звернення Свенна")
    plt.axvspan(result["a"], result["b"], color="green", alpha=0.18,
                label=f"[{result['a']:.4f}; {result['b']:.4f}]")
    plt.xscale("log")
    plt.xlabel("σ — сила розмиття")
    plt.ylabel("MSE")
    plt.title("Лабораторна 1. Локалізація мінімуму методом Свенна")
    plt.grid(alpha=0.3)
    plt.legend()
    save_plot(OUTPUTS / "lab1_svenn.png")

    small = svenn(Oracle(objective), X0, DELTA0 / 10.0)
    large = svenn(Oracle(objective), X0, DELTA0 * 10.0)
    far = svenn(Oracle(objective), 3.0, DELTA0)
    experiments = {
        "delta_small": {"delta": DELTA0 / 10.0, "n_f": small["n_f"],
                        "a": small["a"], "b": small["b"], "p": small["p"]},
        "delta_large": {"delta": DELTA0 * 10.0, "n_f": large["n_f"],
                        "a": large["a"], "b": large["b"], "p": large["p"]},
        "x0_far": {"x0": 3.0, "n_f": far["n_f"], "a": far["a"],
                   "b": far["b"], "p": far["p"]},
    }
    return {"result": result, "experiments": experiments}

def run_lab2(noisy, clean, bracket):
    objective = lambda sigma: mse(gaussian_blur(noisy, sigma), clean)
    a, b = bracket["a"], bracket["b"]
    oracle_d = Oracle(objective)
    res_d = dichotomy(oracle_d, a, b, EPS)
    oracle_g = Oracle(objective)
    res_g = golden_section(oracle_g, a, b, EPS)
    write_csv(OUTPUTS / "lab2_dichotomy.csv", res_d["rows"])
    write_csv(OUTPUTS / "lab2_golden_section.csv", res_g["rows"])

    plt.figure(figsize=(8, 5))
    plt.semilogy([row["k"] for row in res_d["rows"]],
                 [max(row["L"], 1e-16) for row in res_d["rows"]],
                 label="Дихотомія")
    plt.semilogy([row["k"] for row in res_g["rows"]],
                 [max(row["L"], 1e-16) for row in res_g["rows"]],
                 label="Золотий переріз")
    plt.xlabel("Номер ітерації")
    plt.ylabel("Довжина інтервалу")
    plt.title("Лабораторна 2. Звуження інтервалу")
    plt.grid(alpha=0.3)
    plt.legend()
    save_plot(OUTPUTS / "lab2_convergence.png")

    restored = gaussian_blur(noisy, res_g["x"])
    save_image(restored, OUTPUTS / "lab2_restored.png")

    tight_d = dichotomy(Oracle(objective), a, b, 1e-6)
    tight_g = golden_section(Oracle(objective), a, b, 1e-6)
    try:
        dichotomy(Oracle(objective), a, b, EPS, delta=EPS)
        invalid_delta = "помилка не виявлена"
    except ValueError as error:
        invalid_delta = str(error)
    manual = []
    for sigma in (0.3, 0.5, 0.8, 1.5):
        image = gaussian_blur(noisy, sigma)
        manual.append({"sigma": sigma, "MSE": mse(image, clean),
                       "PSNR": psnr(image, clean)})

    return {"dichotomy": res_d, "golden": res_g,
            "MSE": mse(restored, clean), "PSNR": psnr(restored, clean),
            "experiments": {
                "eps_1e-6": {"dichotomy": tight_d, "golden": tight_g},
                "delta_equal_eps": invalid_delta,
                "manual_sigma": manual,
            }}

def run_lab3(noisy, clean, gaussian_sigma):
    initial = np.array([
        min(max(2.0 * gaussian_sigma, 0.5), 4.0),
        min(max(0.15 + 0.1 * NOISE_SIGMA, 0.02), 0.9),
    ])
    objective = make_bilateral_objective(noisy, clean)
    oracle_nm = Oracle(objective)
    result_nm = nelder_mead(oracle_nm, initial, SIMPLEX_EDGE)
    oracle_pw = Oracle(objective)
    result_pw = powell(oracle_pw, initial)
    write_csv(OUTPUTS / "lab3_nelder_mead.csv", result_nm["rows"])
    write_csv(OUTPUTS / "lab3_powell.csv", result_pw["rows"])

    best = result_nm if result_nm["f"] <= result_pw["f"] else result_pw
    restored = bilateral_filter(noisy, best["x"][0], best["x"][1])
    save_image(restored, OUTPUTS / "lab3_restored.png")

    small_nm = nelder_mead(Oracle(objective), initial, SIMPLEX_EDGE / 3.0)
    large_nm = nelder_mead(Oracle(objective), initial, SIMPLEX_EDGE * 3.0)
    strict_pw = powell(Oracle(objective), initial, tol=1e-7, max_cycles=50)
    fixed_range = best_spatial_with_fixed_range(
        Oracle(objective), float(best["x"][0]), float(best["x"][1])
    )

    ss = np.linspace(BILATERAL_LO[0], BILATERAL_HI[0], 40)
    sr = np.linspace(BILATERAL_LO[1], BILATERAL_HI[1], 40)
    grid = np.zeros((len(sr), len(ss)), dtype=float)
    preview_oracle = Oracle(objective)
    for row_index, range_sigma in enumerate(sr):
        for col_index, spatial_sigma in enumerate(ss):
            grid[row_index, col_index] = preview_oracle.peek(
                np.array([spatial_sigma, range_sigma])
            )

    levels = np.unique(np.quantile(grid, np.linspace(0.0, 0.8, 25)))
    if len(levels) < 3:
        levels = np.linspace(float(grid.min()), float(grid.max()) + 1e-12, 12)
    plt.figure(figsize=(9, 6))
    contour = plt.contourf(ss, sr, grid, levels=levels, alpha=0.8)
    plt.colorbar(contour, label="f(σₛ, σᵣ) = MSE")
    nm_path = np.array([row["x"] for row in result_nm["rows"]])
    pw_path = np.array([row["x"] for row in result_pw["rows"]])
    if len(nm_path):
        plt.plot(nm_path[:, 0], nm_path[:, 1], "o-", ms=3, lw=1,
                 label="Нелдер—Мід")
    if len(pw_path):
        plt.plot(pw_path[:, 0], pw_path[:, 1], "s-", ms=4, lw=1.2,
                 label="Пауелл")
    plt.scatter([best["x"][0]], [best["x"][1]], marker="*", s=180,
                color="black", label="найкращий знайдений параметр")
    plt.xlabel("σₛ — просторовий масштаб")
    plt.ylabel("σᵣ — фотометричний поріг")
    plt.title("Лабораторна 3. Лінії рівня та траєкторії методів")
    plt.legend()
    save_plot(OUTPUTS / "lab3_contours.png")

    first_cycle = result_pw["line_costs"][0] if result_pw["line_costs"] else []
    return {
        "initial": initial.tolist(), "nelder_mead": result_nm,
        "powell": result_pw, "best": best,
        "MSE": mse(restored, clean), "PSNR": psnr(restored, clean),
        "experiments": {
            "simplex_small": {"edge": SIMPLEX_EDGE / 3.0,
                               "n_f": small_nm["n_f"], "x": small_nm["x"],
                               "f": small_nm["f"]},
            "simplex_base": {"edge": SIMPLEX_EDGE, "n_f": result_nm["n_f"],
                             "x": result_nm["x"], "f": result_nm["f"]},
            "simplex_large": {"edge": SIMPLEX_EDGE * 3.0,
                               "n_f": large_nm["n_f"], "x": large_nm["x"],
                               "f": large_nm["f"]},
            "powell_tight": {"tol": 1e-7, "cycles": strict_pw["cycles"],
                             "n_f": strict_pw["n_f"], "x": strict_pw["x"],
                             "f": strict_pw["f"],
                             "reached_limit": strict_pw["cycles"] >= 50},
            "powell_first_cycle_line_calls": first_cycle,
            "powell_first_cycle_breakdown": (
                result_pw["line_details"][0] if result_pw["line_details"] else []
            ),
            "fixed_range": fixed_range,
        },
    }

def run_lab4(noisy, clean):
    search = find_lambda(noisy, clean)
    lam = search["lambda"]
    lipschitz = 1.0 + 8.0 * lam
    reference = tikhonov_solve(noisy, lam, 3000)
    exact_energy = energy(reference, noisy, lam)
    checks = operator_checks()

    kinds = ["gd", "armijo", "hb_opt", "hb_09", "nag_convex", "nag_strong"]
    labels = {
        "gd": "Спуск, сталий крок",
        "armijo": "Спуск, правило Армійо",
        "hb_opt": "Важка кулька, β оптимальне",
        "hb_09": "Важка кулька, β = 0,9",
        "nag_convex": "Нестеров, опуклий варіант",
        "nag_strong": "Нестеров, сильно опуклий",
    }
    results = {}
    for kind in kinds:
        results[kind] = gradient_method(noisy, lam, kind, reference=reference)
    table = []
    for kind, result in results.items():
        restored = np.clip(result["image"], 0.0, 1.0)
        table.append({"method": labels[kind], "iterations": result["iterations"],
                      "steps_run": result["steps_run"], "N_f": result["n_f"],
                      "N_grad": result["n_grad"], "total_calls": result["n_f"] + result["n_grad"],
                      "converged": result["converged"], "E": result["energy"],
                      "gradient_norm": result["gradient_norm"],
                      "PSNR": psnr(restored, clean)})
    write_csv(OUTPUTS / "lab4_methods.csv", table)

    best_method = results["nag_strong"]
    restored = np.clip(best_method["image"], 0.0, 1.0)
    save_image(restored, OUTPUTS / "lab4_restored.png")
    lambda_tv = max(0.02, 0.15 * lam)
    full_variation = full_variation_solve(noisy, lambda_tv)
    tv_restored = np.clip(full_variation["image"], 0.0, 1.0)
    full_variation["MSE"] = mse(tv_restored, clean)
    full_variation["PSNR"] = psnr(tv_restored, clean)
    full_variation["isotropic_TV"] = isotropic_total_variation(tv_restored)
    save_image(tv_restored, OUTPUTS / "lab4_full_variation_restored.png")
    write_csv(OUTPUTS / "lab4_full_variation.csv", full_variation["rows"])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis, image, title, score in (
        (axes[0], restored, "Квадратична модель Тихонова", psnr(restored, clean)),
        (axes[1], tv_restored, "Модель повної варіації", full_variation["PSNR"]),
    ):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"{title}\nPSNR = {score:.2f} дБ")
        axis.axis("off")
    fig.suptitle("Лабораторна 4. Вплив моделі регуляризації")
    save_plot(OUTPUTS / "lab4_model_comparison.png")

    plt.figure(figsize=(9, 5))
    for kind, result in results.items():
        values = [max(row["E"] - exact_energy, 1e-16) for row in result["rows"]]
        plt.semilogy(range(len(values)), values, label=labels[kind])
    plt.xlabel("Номер ітерації")
    plt.ylabel("E(Iₖ) − E*")
    plt.title("Лабораторна 4. Збіжність за енергією")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save_plot(OUTPUTS / "lab4_convergence_f.png")

    plt.figure(figsize=(9, 5))
    for kind, result in results.items():
        values = [max(row["dist"], 1e-16) for row in result["rows"]]
        plt.semilogy(range(len(values)), values, label=labels[kind])
    plt.xlabel("Номер ітерації")
    plt.ylabel("‖Iₖ − I*‖₂")
    plt.title("Лабораторна 4. Відстань до еталонного розв’язку")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save_plot(OUTPUTS / "lab4_convergence_dist.png")

    lambda_experiment = []
    lambda_values = (0.05, 0.143716, 0.24, 1.0, 5.0)
    for tested_lambda in lambda_values:
        solution = tikhonov_solve(noisy, tested_lambda, 10000)
        lambda_experiment.append({"lambda": tested_lambda,
                                  "PSNR": psnr(np.clip(solution, 0.0, 1.0), clean),
                                  "MSE": mse(np.clip(solution, 0.0, 1.0), clean)})
    write_csv(OUTPUTS / "lab4_lambda_quality.csv", lambda_experiment)
    plt.figure(figsize=(8, 5))
    plt.semilogx([row["lambda"] for row in lambda_experiment],
                 [row["PSNR"] for row in lambda_experiment], "o-")
    plt.axvline(lam, color="red", ls="--", label=f"знайдене λ* = {lam:.4f}")
    plt.xlabel("λ — вага регуляризації")
    plt.ylabel("PSNR, дБ")
    plt.title("Лабораторна 4. Вибір параметра регуляризації")
    plt.grid(alpha=0.3)
    plt.legend()
    save_plot(OUTPUTS / "lab4_lambda_psnr.png")

    high_lam = 6.0
    high_results = {kind: gradient_method(noisy, high_lam, kind, max_iter=3000)
                    for kind in ("gd", "hb_opt", "nag_strong")}
    high_summary = {kind: {"iterations": result["iterations"],
                           "steps_run": result["steps_run"],
                           "converged": result["converged"],
                           "gradient_norm": result["gradient_norm"]}
                    for kind, result in high_results.items()}

    diverging_step = 2.1 / lipschitz
    diverging_image = noisy.copy()
    divergence_rows = [{"k": 0, "E": energy(diverging_image, noisy, lam)}]
    for k in range(1, 51):
        diverging_image = diverging_image - diverging_step * energy_grad(
            diverging_image, noisy, lam
        )
        if k in (10, 20, 30, 40, 50):
            divergence_rows.append({"k": k, "E": energy(diverging_image, noisy, lam)})

    return {
        "lambda_search": search, "lambda": lam, "L": lipschitz,
        "kappa": lipschitz, "h": 1.0 / lipschitz,
        "energy_star": exact_energy, "methods": results, "method_table": table,
        "PSNR": psnr(restored, clean), "MSE": mse(restored, clean),
        "operator_checks": checks,
        "full_variation": full_variation,
        "experiments": {
            "lambda_quality": lambda_experiment,
            "high_condition_lambda_6": high_summary,
            "divergent_step": {"h": diverging_step, "rows": divergence_rows},
        },
    }

def run_lab5(raw, clean, lam):
    unconstrained = tikhonov_solve(raw, lam, 3000)
    outside_raw = violation(raw)
    outside_unconstrained = violation(unconstrained)
    save_image(unconstrained, OUTPUTS / "lab5_unconstrained.png")

    projection = projected_gradient(raw, lam)
    fista = projected_gradient(raw, lam, use_gradient_stop="fista")
    penalty = penalty_method(raw, lam)
    save_image(projection["image"], OUTPUTS / "lab5_restored.png")
    save_image(penalty["image"], OUTPUTS / "lab5_penalty_restored.png")
    write_csv(OUTPUTS / "lab5_projection.csv", projection["rows"])
    write_csv(OUTPUTS / "lab5_fista.csv", fista["rows"])
    write_csv(OUTPUTS / "lab5_penalty.csv", penalty["rows"])

    wrong_stop = projected_gradient(raw, lam, use_gradient_stop=True)
    penalty_2 = penalty_method(raw, lam, growth=2.0)
    penalty_100 = penalty_method(raw, lam, growth=100.0)

    plt.figure(figsize=(9, 5))
    plt.hist(raw.ravel(), bins=120, alpha=0.55, label="сире спостереження")
    plt.hist(unconstrained.ravel(), bins=120, alpha=0.55,
             label="безумовний розв’язок")
    plt.axvline(0.0, color="red", ls="--", label="межі [0, 1]")
    plt.axvline(1.0, color="red", ls="--")
    plt.xlabel("Яскравість пікселя")
    plt.ylabel("Кількість пікселів")
    plt.title("Лабораторна 5. Порушення меж яскравості")
    plt.legend()
    save_plot(OUTPUTS / "lab5_histogram.png")

    plt.figure(figsize=(9, 5))
    for name, result in (("Проєкція", projection), ("FISTA", fista)):
        values = [max(row["E"] - projection["energy"], 1e-16)
                  for row in result["rows"]]
        plt.semilogy(range(len(values)), values, label=name)
    plt.xlabel("Номер ітерації")
    plt.ylabel("E(Iₖ) − E(проєкція)")
    plt.title("Лабораторна 5. Збіжність проєкційних методів")
    plt.grid(alpha=0.3)
    plt.legend()
    save_plot(OUTPUTS / "lab5_convergence.png")

    active = np.full(projection["image"].shape, 128, dtype=np.uint8)
    active[projection["image"] <= 1e-12] = 0
    active[projection["image"] >= 1.0 - 1e-12] = 255
    Image.fromarray(active, mode="L").save(OUTPUTS / "lab5_active_constraints.png")

    growth = []
    for label, result in (("β = 2", penalty_2), ("β = 10", penalty),
                          ("β = 100", penalty_100)):
        last = result["rows"][-1]
        growth.append({"growth": label, "outer_iterations": len(result["rows"]),
                       "n_f": result["n_f"], "n_grad": result["n_grad"],
                       "total_calls": result["calls"],
                       "max_violation": last["violation"],
                       "outside_pixels": last["outside"], "rP": last["rP"]})

    return {
        "raw_violation": outside_raw,
        "unconstrained_violation": outside_unconstrained,
        "unconstrained_energy": energy(unconstrained, raw, lam),
        "projection": projection, "fista": fista, "penalty": penalty,
        "PSNR": psnr(projection["image"], clean),
        "MSE": mse(projection["image"], clean),
        "penalty_PSNR": psnr(np.clip(penalty["image"], 0.0, 1.0), clean),
        "wrong_stop": {"iterations": wrong_stop["iterations"],
                       "steps_run": wrong_stop["steps_run"],
                       "converged": wrong_stop["converged"],
                       "gradient_norm": wrong_stop["grad"],
                       "gradient_map": wrong_stop["G"]},
        "growth_experiment": growth,
    }

def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    clean = clean_image(SIZE)

    source_path = DATA / "00_clean.png"
    source_pixels = np.asarray(Image.open(source_path).convert("L"))
    rebuilt_pixels = np.rint(clean * 255.0).astype(np.uint8)
    source_matches = bool(np.array_equal(source_pixels, rebuilt_pixels))
    if not source_matches:
        raise ValueError("data/00_clean.png не відповідає зображенню з методички")

    raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)
    save_image(noisy, OUTPUTS / "00_noisy.png")
    noisy_mse = mse(noisy, clean)
    noisy_psnr = psnr(noisy, clean)

    lab1 = run_lab1(noisy, clean)
    lab2 = run_lab2(noisy, clean, lab1["result"])
    lab3 = run_lab3(noisy, clean, lab2["golden"]["x"])
    lab4 = run_lab4(noisy, clean)
    lab5 = run_lab5(raw, clean, lab4["lambda"])

    summary_rows = [
        {"Етап": "Зашумлене зображення", "Метод": "Гаусів шум",
         "Невідомих": "—", "Звернень": "—", "MSE": noisy_mse,
         "PSNR, дБ": noisy_psnr},
        {"Етап": "Лабораторна 2", "Метод": "Гаусів фільтр + золотий переріз",
         "Невідомих": 1, "Звернень": lab2["golden"]["n_f"],
         "MSE": lab2["MSE"], "PSNR, дБ": lab2["PSNR"]},
        {"Етап": "Лабораторна 3", "Метод": "Білатеральний фільтр + найкращий результат",
         "Невідомих": 2, "Звернень": min(lab3["nelder_mead"]["n_f"], lab3["powell"]["n_f"]),
         "MSE": lab3["MSE"], "PSNR, дБ": lab3["PSNR"]},
        {"Етап": "Лабораторна 4", "Метод": "Нестеров, сильно опуклий",
         "Невідомих": SIZE * SIZE, "Звернень": lab4["methods"]["nag_strong"]["n_f"] +
         lab4["methods"]["nag_strong"]["n_grad"], "MSE": lab4["MSE"],
         "PSNR, дБ": lab4["PSNR"]},
        {"Етап": "Лаб. 4, додатково", "Метод": "Нестеров, повна варіація",
         "Невідомих": SIZE * SIZE,
         "Звернень": lab4["full_variation"]["n_f"] +
         lab4["full_variation"]["n_grad"],
         "MSE": lab4["full_variation"]["MSE"],
         "PSNR, дБ": lab4["full_variation"]["PSNR"]},
        {"Етап": "Лабораторна 5", "Метод": "Проєкція градієнта",
         "Невідомих": SIZE * SIZE, "Звернень": lab5["projection"]["n_f"] +
         lab5["projection"]["n_grad"], "MSE": lab5["MSE"],
         "PSNR, дБ": lab5["PSNR"]},
    ]
    write_csv(OUTPUTS / "summary.csv", summary_rows)

if __name__ == "__main__":
    main()
