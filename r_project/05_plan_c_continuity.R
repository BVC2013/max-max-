# =============================================================================
# 05_plan_c_continuity.R
# Plan C — Continuity Breakdown (Provider Fragmentation)
# Hard trend: cumulative care transfers → ED escalation + longer encounters.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
  library(ggplot2)
  library(survival)
  library(survminer)
})

PROCESSED_DIR <- "data/processed"
PLOT_DIR      <- file.path(get0("OUTPUT_DIR", envir = .GlobalEnv, ifnotfound = "output"), "plots")
dir.create(PLOT_DIR, showWarnings = FALSE, recursive = TRUE)

enc <- readRDS(file.path(PROCESSED_DIR, "encounters_engineered.rds"))

# Bin transfer count
continuity <- enc |>
  filter(!is.na(cumulative_care_transfers)) |>
  mutate(
    transfer_bin = case_when(
      cumulative_care_transfers == 0 ~ "0",
      cumulative_care_transfers == 1 ~ "1",
      cumulative_care_transfers == 2 ~ "2",
      TRUE                           ~ "3+"
    ),
    transfer_bin = factor(transfer_bin, levels = c("0", "1", "2", "3+")),
    ed_num       = as.integer(is_ed_visit_flag)
  )

message("Plan C — rows: ", nrow(continuity))

# ------ (1) ED escalation rate × transfers ----------------------------------

transfer_summary <- continuity |>
  group_by(transfer_bin) |>
  summarise(
    n          = n(),
    ed_rate    = mean(ed_num, na.rm = TRUE),
    avg_dur    = mean(encounter_duration_hours, na.rm = TRUE),
    .groups    = "drop"
  )

p1 <- ggplot(transfer_summary, aes(x = transfer_bin)) +
  geom_col(aes(y = ed_rate), fill = "#d73027", alpha = 0.85, width = 0.55) +
  geom_line(aes(y = ed_rate, group = 1), color = "#d73027", linewidth = 1) +
  geom_line(
    aes(y = avg_dur / max(avg_dur, na.rm = TRUE) * max(ed_rate, na.rm = TRUE), group = 1),
    color = "#1a9641", linewidth = 1.2, linetype = "dashed"
  ) +
  geom_text(aes(y = ed_rate, label = paste0(round(ed_rate * 100, 1), "%\nn=", comma(n))),
            vjust = -0.4, size = 3.5) +
  scale_y_continuous(
    labels   = percent_format(),
    name     = "ED Escalation Rate",
    sec.axis = sec_axis(
      ~ . / max(transfer_summary$ed_rate, na.rm = TRUE) * max(transfer_summary$avg_dur, na.rm = TRUE),
      name   = "Avg Encounter Duration (hours)",
      labels = label_number(suffix = " h")
    )
  ) +
  labs(
    title    = "Plan C — Care Transfer Count vs ED Escalation & Duration",
    subtitle = "Bars = ED rate | Dashed green = avg encounter duration",
    x        = "Cumulative Care Transfers"
  ) +
  theme_minimal(base_size = 13) +
  theme(axis.title.y.right = element_text(color = "#1a9641"))

ggsave(file.path(PLOT_DIR, "plan_c_transfer_escalation.png"), p1,
       width = 9, height = 6, dpi = 150)

# ------ (2) Ridge-plot: duration by transfer bin ----------------------------

if (requireNamespace("ggridges", quietly = TRUE)) {
  library(ggridges)

  p2 <- continuity |>
    filter(!is.na(encounter_duration_hours), encounter_duration_hours > 0) |>
    ggplot(aes(x = encounter_duration_hours, y = transfer_bin, fill = transfer_bin)) +
    geom_density_ridges(alpha = 0.7, scale = 1.2) +
    coord_cartesian(xlim = c(0, quantile(
      continuity$encounter_duration_hours, 0.97, na.rm = TRUE
    ))) +
    scale_fill_manual(values = c("0" = "#4575b4", "1" = "#fee090",
                                 "2" = "#f46d43", "3+" = "#d73027")) +
    labs(
      title    = "Plan C — Duration Distribution by Care Transfer Count",
      subtitle = "Higher transfer counts shift duration rightward",
      x        = "Encounter Duration (hours)",
      y        = "Cumulative Care Transfers",
      fill     = NULL
    ) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "none")

  ggsave(file.path(PLOT_DIR, "plan_c_duration_ridges.png"), p2,
         width = 10, height = 6, dpi = 150)
}

# ------ (3) Kaplan-Meier: time to ED escalation by transfer stratum ----------

# Journey-level dataset: time until first ED encounter
km_data <- enc |>
  filter(!is.na(diagnosis_value), !is.na(days_since_last_visit) | visit_number == 1) |>
  group_by(patient_durable_key, diagnosis_value) |>
  summarise(
    transfer_stratum   = cut(max(cumulative_care_transfers, na.rm = TRUE),
                             breaks = c(-Inf, 0, 1, Inf),
                             labels = c("0 transfers", "1 transfer", "2+ transfers")),
    had_ed             = any(is_ed_visit_flag, na.rm = TRUE),
    days_to_event      = if_else(
      any(is_ed_visit_flag, na.rm = TRUE),
      min(cumulative_days_in_journey[is_ed_visit_flag], na.rm = TRUE),
      max(cumulative_days_in_journey, na.rm = TRUE)
    ),
    .groups            = "drop"
  ) |>
  filter(!is.na(transfer_stratum), days_to_event >= 0)

if (nrow(km_data) > 200) {
  km_fit <- survfit(
    Surv(days_to_event, had_ed) ~ transfer_stratum,
    data = km_data
  )

  km_plot <- ggsurvplot(
    km_fit,
    data         = km_data,
    conf.int     = TRUE,
    risk.table   = TRUE,
    fun          = "event",
    palette      = c("#4575b4", "#f46d43", "#d73027"),
    legend.title = "Transfer Stratum",
    title        = "Plan C — Cumulative ED Escalation by Transfer Stratum",
    xlab         = "Days in Journey",
    ylab         = "Cumulative Probability of ED Escalation"
  )

  ggsave(file.path(PLOT_DIR, "plan_c_km_ed_escalation.png"),
         km_plot$plot, width = 10, height = 7, dpi = 150)
}

message("Plan C plots saved to ", PLOT_DIR)
message("05_plan_c_continuity.R complete.")
