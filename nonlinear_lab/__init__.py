from nonlinear_lab.direct_identification import (
    estimate_base_from_triplet,
    estimate_delay_from_quadruplet,
    estimate_mixed_from_quintet,
    fit_direct_identification,
    rolling_direct_identification,
    summarize_parameter_errors,
)
from nonlinear_lab.features import growth_rate, make_regression_df
from nonlinear_lab.lifecycle import find_lifecycle_stages, stagewise_analysis
from nonlinear_lab.models import (
    fixed_point,
    generate_base_process,
    generate_delay_process,
    generate_mixed_process,
    theoretical_coeffs,
)
from nonlinear_lab.plotting import (
    plot_phase_portrait,
    plot_rolling_coefficients,
    plot_series,
    plot_stepwise_frequency,
    save_figure,
)
from nonlinear_lab.plan_a_experiment import (
    add_multiplicative_noise,
    detect_degenerate_window,
    fit_structural_regression,
    identify_best_model,
    rolling_structural_regression,
    run_plan_a_experiment,
    run_plan_a_noise_sweep,
    save_plan_a_results,
)
from nonlinear_lab.plan_b_experiment import (
    estimate_characteristic_period,
    run_plan_b_experiment,
    save_plan_b_results,
    score_selected_features,
    summarize_plan_b_results,
)
from nonlinear_lab.plan_c_experiment import (
    compute_ranking_metrics,
    run_plan_c_experiment,
    save_plan_c_results,
)
from nonlinear_lab.plan_d_experiment import (
    active_features_from_coefficients,
    run_plan_d_experiment,
    save_plan_d_results,
    score_support,
    true_coefficients_for_case,
)
from nonlinear_lab.plan_e_experiment import (
    alpha_configs,
    classify_interpretability_window,
    run_plan_e_experiment,
    save_plan_e_results,
)
from nonlinear_lab.plan_f_experiment import (
    choose_component_count,
    run_plan_f_experiment,
    save_plan_f_results,
)
from nonlinear_lab.plan_g_experiment import (
    extract_window_diagnostics,
    oracle_regime_label,
    predict_regime_label,
    run_plan_g_experiment,
    save_plan_g_results,
)
from nonlinear_lab.plan_h_experiment import (
    classify_window_failure,
    compute_condition_number,
    compute_vif,
    run_plan_h_experiment,
    save_plan_h_results,
)
from nonlinear_lab.reporting import build_experiment_report, save_experiment_report
from nonlinear_lab.regression import (
    fit_enter_with_beta,
    rolling_window_regression,
    stepwise_frequency,
    stepwise_selection,
)

__all__ = [
    "estimate_base_from_triplet",
    "estimate_delay_from_quadruplet",
    "estimate_mixed_from_quintet",
    "fixed_point",
    "fit_direct_identification",
    "fit_enter_with_beta",
    "find_lifecycle_stages",
    "generate_base_process",
    "generate_delay_process",
    "generate_mixed_process",
    "growth_rate",
    "make_regression_df",
    "build_experiment_report",
    "plot_phase_portrait",
    "plot_rolling_coefficients",
    "plot_series",
    "plot_stepwise_frequency",
    "add_multiplicative_noise",
    "detect_degenerate_window",
    "fit_structural_regression",
    "identify_best_model",
    "rolling_direct_identification",
    "rolling_structural_regression",
    "rolling_window_regression",
    "run_plan_a_experiment",
    "run_plan_a_noise_sweep",
    "run_plan_b_experiment",
    "run_plan_c_experiment",
    "run_plan_d_experiment",
    "run_plan_e_experiment",
    "run_plan_f_experiment",
    "run_plan_g_experiment",
    "run_plan_h_experiment",
    "save_experiment_report",
    "save_figure",
    "save_plan_a_results",
    "save_plan_b_results",
    "save_plan_c_results",
    "save_plan_d_results",
    "save_plan_e_results",
    "save_plan_f_results",
    "save_plan_g_results",
    "save_plan_h_results",
    "alpha_configs",
    "classify_interpretability_window",
    "score_selected_features",
    "score_support",
    "compute_ranking_metrics",
    "compute_condition_number",
    "compute_vif",
    "classify_window_failure",
    "extract_window_diagnostics",
    "oracle_regime_label",
    "predict_regime_label",
    "stagewise_analysis",
    "summarize_plan_b_results",
    "stepwise_frequency",
    "stepwise_selection",
    "summarize_parameter_errors",
    "theoretical_coeffs",
    "estimate_characteristic_period",
    "active_features_from_coefficients",
    "choose_component_count",
    "true_coefficients_for_case",
]
