class AdaptiveSelectorEngine:
  

    @staticmethod
    def recommend(profile: dict, horizon: int) -> dict:
        score_es = 0.0
        score_reg = 0.0
        reasons = []

        n = profile["n"]
        nonlin = profile["nonlinearity_index"]
        r2_poly2 = profile.get("r2_poly2", 0.0)
        shift = profile["structural_shift_index"]
        f_t = profile["trend_strength"]
        f_s = profile["seasonality_strength"]
        outliers = profile["outlier_ratio"]
        is_stat = profile["is_stationary"]
        acf1 = profile["acf_lag1"]
        horizon_ratio = horizon / max(n, 1)

       
        if not is_stat and shift > 0.25 and (nonlin < 0.06 or r2_poly2 < 0.85):
            w = 2.8 * shift + (0.8 if acf1 > 0.8 else 0.0)
            score_es += w
            reasons.append(
                f"Нестаціонарність та злами тренду (Shift={shift:.2f}, R2_poly2={r2_poly2:.2f}) -> перевага локальної адаптації ES (+{w:.2f} ES)."
            )

       
        if nonlin >= 0.04 and r2_poly2 >= 0.85:
            w = min(3.5, 25.0 * nonlin)
            score_reg += w
            reasons.append(
                f"Стійкий нелінійний тренд (Nonlin={nonlin:.3f}, R2_poly2={r2_poly2:.2f}) -> поліноміальна регресія (+{w:.2f} Reg)."
            )

        if outliers >= 0.015:
            w = min(2.8, 45.0 * outliers)
            score_reg += w
            reasons.append(
                f"Виявлено точкові викиди ({outliers * 100:.1f}%) -> робастна регресія (Huber) стійкіша за ES (+{w:.2f} Reg)."
            )

        if f_s > 0.65:
            if shift < 0.25 and nonlin < 0.05 and n >= 80:
                score_reg += 1.25
                score_es += 0.95
                reasons.append(
                    f"Висока регулярна сезонність (F_S={f_s:.2f}) на довгому ряді -> компактний базис Фур'є в регресії (+1.25 Reg)."
                )
            elif shift < 0.25 and nonlin < 0.05:
                score_es += 1.15
                score_reg += 0.90
                reasons.append(
                    f"Висока регулярна сезонність (F_S={f_s:.2f}) -> ефективність Holt-Winters (+1.15 ES)."
                )
            else:
                score_reg += 1.20
                reasons.append(
                    f"Висока сезонність (F_S={f_s:.2f}) зі складним трендом -> гармоніки Фур'є в регресії (+1.20 Reg)."
                )

        if n < 60:
            score_es += 1.2
            reasons.append(f"Короткий ряд (N={n}) -> менший ризик перенавчання в ES (+1.20 ES).")
        elif horizon_ratio > 0.2 and f_t > 0.7 and shift < 0.2 and r2_poly2 >= 0.85:
            score_reg += 1.0
            reasons.append("Довгий горизонт зі стійким глобальним трендом -> глобальна регресія (+1.00 Reg).")

        if acf1 > 0.85 and (nonlin < 0.04 or r2_poly2 < 0.85):
            score_es += 1.0
            reasons.append(f"Сильна інерція (ACF1={acf1:.2f}) -> локальне експоненціальне згладжування (+1.00 ES).")

        diff = abs(score_es - score_reg)
        if diff < 0.20:
            recommended = "Hybrid / Ensemble (0.5*ES + 0.5*Reg)"
            primary_choice = "ExponentialSmoothing" if score_es >= score_reg else "Regression"
        elif score_es > score_reg:
            recommended = "ExponentialSmoothing"
            primary_choice = "ExponentialSmoothing"
        else:
            recommended = "Regression"
            primary_choice = "Regression"

        confidence = float(diff / (score_es + score_reg + 1e-9))

        return {
            "recommended_paradigm": recommended,
            "primary_choice": primary_choice,
            "score_es": round(score_es, 3),
            "score_reg": round(score_reg, 3),
            "confidence": round(confidence, 3),
            "reasons": reasons,
        }