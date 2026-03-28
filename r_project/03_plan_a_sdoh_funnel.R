# =============================================================================
# 03_plan_a_sdoh_funnel.R
# Plan A — Access Friction Funnel
# Hard trend: SDoH barriers predict Is_True_Abandonment_Risk in ED/hospital
# encounters.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(broom)
  library(scales)
  library(ggplot2)
})

PROCESSED_DIR <- "data/processed"
PLOT_DIR      <- file.path(get0("OUTPUT_DIR", envir = .GlobalEnv, ifnotfound = "output"), "plots")
dir.create(PLOT_DIR, showWarnings = FALSE, recursive = TRUE)

enc <- readRDS(file.path(PROCESSED_DIR, "encounters_engineered.rds"))

# Restrict to ED or hospital encounters
ed_hosp <- enc |>
  filter(is_ed_visit_flag | is_hospital_admission_flag) |>
  mutate(
    abandonment_num = as.integer(is_true_abandonment_risk),
    sdoh_combo = paste0(
      "T:", as.integer(has_transport_need),
      " F:", as.integer(has_financial_strain),
      " H:", as.integer(has_housing_instability)
    )
  )

message("Plan A — ED/hospital rows: ", nrow(ed_hosp))

# ------ (1) Abandonment rate by each SDoH barrier ---------------------------

barrier_summary <- ed_hosp |>
  pivot_longer(
    cols      = c(has_transport_need, has_financial_strain, has_housing_instability),
    names_to  = "barrier",
    values_to = "present"
  ) |>
  group_by(barrier, present) |>
  summarise(
    n              = n(),
    abandon_rate   = mean(abandonment_num, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    barrier = recode(barrier,
      has_transport_need      = "Transport Need",
      has_financial_strain    = "Financial Strain",
      has_housing_instability = "Housing Instability"
    ),
    present = if_else(present, "Barrier Present", "No Barrier")
  )

p1 <- ggplot(barrier_summary, aes(x = barrier, y = abandon_rate, fill = present)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(
    aes(label = paste0(round(abandon_rate * 100, 1), "%\nn=", comma(n))),
    position = position_dodge(width = 0.7),
    vjust = -0.4, size = 3.2
  ) +
  scale_y_continuous(labels = percent_format(), limits = c(0, NA), expand = expansion(mult = c(0, 0.15))) +
  scale_fill_manual(values = c("Barrier Present" = "#d73027", "No Barrier" = "#4575b4")) +
  labs(
    title    = "Plan A — SDoH Barriers & Journey Abandonment Risk",
    subtitle = "ED / Hospital encounters only",
    x        = NULL,
    y        = "Abandonment Risk Rate",
    fill     = NULL,
    caption  = "Is_True_Abandonment_Risk = no follow-up AND not deceased"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")

ggsave(file.path(PLOT_DIR, "plan_a_sdoh_abandonment_bars.png"), p1,
       width = 9, height = 6, dpi = 150)

# ------ (2) SDoH combo heat-map ---------------------------------------------

combo_summary <- ed_hosp |>
  group_by(sdoh_combo, has_transport_need, has_financial_strain, has_housing_instability) |>
  summarise(
    n            = n(),
    abandon_rate = mean(abandonment_num, na.rm = TRUE),
    .groups      = "drop"
  )

p2 <- ggplot(
    combo_summary,
    aes(
      x    = factor(has_financial_strain, labels = c("No Financial", "Financial Strain")),
      y    = factor(has_transport_need,   labels = c("No Transport", "Transport Need")),
      fill = abandon_rate,
      size = n
    )
  ) +
  geom_point(shape = 21, stroke = 0.4, color = "white") +
  facet_wrap(~ factor(has_housing_instability,
                      labels = c("No Housing Issue", "Housing Instability")),
             ncol = 2) +
  scale_fill_gradient(low = "#ffffcc", high = "#d73027",
                      labels = percent_format(), name = "Abandon\nRate") +
  scale_size_area(max_size = 20, name = "Encounters") +
  labs(
    title    = "Plan A — Abandonment Risk by SDoH Combination",
    subtitle = "Dot size = encounter volume | Faceted by Housing Instability",
    x        = NULL,
    y        = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank())

ggsave(file.path(PLOT_DIR, "plan_a_sdoh_combo_heatmap.png"), p2,
       width = 11, height = 6, dpi = 150)

# ------ (3) Logistic regression — odds ratios --------------------------------

model_data <- ed_hosp |>
  select(abandonment_num, has_transport_need,
         has_financial_strain, has_housing_instability) |>
  drop_na()

if (nrow(model_data) > 100) {
  fit <- glm(
    abandonment_num ~ has_transport_need + has_financial_strain + has_housing_instability,
    data   = model_data,
    family = binomial()
  )

  or_df <- tidy(fit, exponentiate = TRUE, conf.int = TRUE) |>
    filter(term != "(Intercept)") |>
    mutate(
      term = recode(term,
        has_transport_needTRUE      = "Transport Need",
        has_financial_strainTRUE    = "Financial Strain",
        has_housing_instabilityTRUE = "Housing Instability"
      )
    )

  p3 <- ggplot(or_df, aes(x = estimate, y = reorder(term, estimate))) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.25, linewidth = 0.8) +
    geom_point(size = 4, color = "#d73027") +
    geom_text(aes(label = sprintf("OR=%.2f", estimate)), hjust = -0.3, size = 3.5) +
    labs(
      title    = "Plan A — Logistic Regression Odds Ratios",
      subtitle = "Outcome: Is_True_Abandonment_Risk | ED/Hospital encounters",
      x        = "Odds Ratio (95% CI)",
      y        = NULL
    ) +
    theme_minimal(base_size = 13)

  ggsave(file.path(PLOT_DIR, "plan_a_logistic_or.png"), p3,
         width = 8, height = 5, dpi = 150)
}

message("Plan A plots saved to ", PLOT_DIR)
message("03_plan_a_sdoh_funnel.R complete.")
