# =============================================================================
# 04_plan_b_off_hours.R
# Plan B — Off-Hours Throughput Strain
# Hard trend: ED rate and encounter duration vary by hour-of-arrival.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
  library(ggplot2)
})

PROCESSED_DIR <- "data/processed"
PLOT_DIR      <- file.path(get0("OUTPUT_DIR", envir = .GlobalEnv, ifnotfound = "output"), "plots")
dir.create(PLOT_DIR, showWarnings = FALSE, recursive = TRUE)

enc <- readRDS(file.path(PROCESSED_DIR, "encounters_engineered.rds"))

# Keep only rows with admit hour information
hourly <- enc |>
  filter(!is.na(admit_hour)) |>
  mutate(
    hour_bucket = case_when(
      admit_hour < 7               ~ "Night (0–6)",
      admit_hour >= 17             ~ "Evening (17–23)",
      TRUE                         ~ "Day (7–16)"
    ),
    hour_bucket = factor(hour_bucket,
                         levels = c("Night (0–6)", "Day (7–16)", "Evening (17–23)"))
  )

message("Plan B — rows with admit hour: ", nrow(hourly))

# ------ (1) ED rate by hour of day ------------------------------------------

hourly_summary <- hourly |>
  group_by(admit_hour) |>
  summarise(
    n          = n(),
    ed_rate    = mean(is_ed_visit_flag, na.rm = TRUE),
    avg_dur    = mean(encounter_duration_hours, na.rm = TRUE),
    is_off     = first(is_off_hours),
    .groups    = "drop"
  )

p1 <- ggplot(hourly_summary, aes(x = admit_hour)) +
  geom_col(aes(y = ed_rate, fill = is_off), alpha = 0.85) +
  geom_line(aes(y = avg_dur / max(avg_dur, na.rm = TRUE) * max(ed_rate, na.rm = TRUE)),
            color = "#1a9641", linewidth = 1.2, na.rm = TRUE) +
  geom_point(aes(y = avg_dur / max(avg_dur, na.rm = TRUE) * max(ed_rate, na.rm = TRUE)),
             color = "#1a9641", size = 2, na.rm = TRUE) +
  annotate("rect", xmin = -0.5, xmax = 6.5,  ymin = -Inf, ymax = Inf,
           alpha = 0.08, fill = "#d73027") +
  annotate("rect", xmin = 16.5, xmax = 23.5, ymin = -Inf, ymax = Inf,
           alpha = 0.08, fill = "#d73027") +
  scale_x_continuous(breaks = 0:23) +
  scale_y_continuous(
    labels   = percent_format(),
    name     = "ED Visit Rate",
    sec.axis = sec_axis(
      ~ . / max(hourly_summary$ed_rate, na.rm = TRUE) * max(hourly_summary$avg_dur, na.rm = TRUE),
      name   = "Avg Encounter Duration (hours)",
      labels = label_number(suffix = " h")
    )
  ) +
  scale_fill_manual(values = c("TRUE" = "#d73027", "FALSE" = "#4575b4"),
                    labels = c("TRUE" = "Off-Hours", "FALSE" = "Regular Hours"),
                    name   = NULL) +
  labs(
    title    = "Plan B — ED Rate & Duration by Hour of Arrival",
    subtitle = "Bars = ED rate | Green line = avg encounter duration | Red shading = off-hours",
    x        = "Admit Hour (0–23)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top", axis.title.y.right = element_text(color = "#1a9641"))

ggsave(file.path(PLOT_DIR, "plan_b_hourly_dual_axis.png"), p1,
       width = 13, height = 6, dpi = 150)

# ------ (2) Duration box-plot: off-hours vs regular -------------------------

p2 <- hourly |>
  filter(!is.na(encounter_duration_hours), encounter_duration_hours > 0) |>
  ggplot(aes(x = hour_bucket, y = encounter_duration_hours, fill = hour_bucket)) +
  geom_violin(trim = TRUE, alpha = 0.6) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3,
               fill = "white", color = "black") +
  coord_cartesian(ylim = c(0, quantile(
    hourly$encounter_duration_hours, 0.99, na.rm = TRUE
  ))) +
  scale_fill_manual(values = c(
    "Night (0–6)"    = "#d73027",
    "Day (7–16)"     = "#4575b4",
    "Evening (17–23)"= "#f46d43"
  )) +
  labs(
    title    = "Plan B — Encounter Duration by Time-of-Arrival Bucket",
    subtitle = "Diamond = mean | Box = IQR | Violin = distribution (clipped at 99th pctile)",
    x        = NULL,
    y        = "Encounter Duration (hours)",
    fill     = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave(file.path(PLOT_DIR, "plan_b_duration_violin.png"), p2,
       width = 9, height = 6, dpi = 150)

# ------ (3) Heat-map: day-of-week × hour-of-day ∝ ED rate ------------------

dow_hour <- hourly |>
  mutate(day_of_week = wday(encounter_date, label = TRUE, abbr = TRUE)) |>
  group_by(day_of_week, admit_hour) |>
  summarise(ed_rate = mean(is_ed_visit_flag, na.rm = TRUE), n = n(), .groups = "drop")

p3 <- ggplot(dow_hour, aes(x = admit_hour, y = day_of_week, fill = ed_rate)) +
  geom_tile(color = "white", linewidth = 0.3) +
  scale_fill_gradientn(
    colors = c("#ffffcc", "#fd8d3c", "#d73027"),
    labels = percent_format(),
    name   = "ED Rate"
  ) +
  scale_x_continuous(breaks = seq(0, 23, by = 3)) +
  labs(
    title    = "Plan B — ED Rate Heat-Map by Day-of-Week & Hour",
    subtitle = "Darker cells = higher proportion of ED encounters",
    x        = "Admit Hour",
    y        = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank())

ggsave(file.path(PLOT_DIR, "plan_b_dow_hour_heatmap.png"), p3,
       width = 12, height = 5, dpi = 150)

message("Plan B plots saved to ", PLOT_DIR)
message("04_plan_b_off_hours.R complete.")
