# =============================================================================
# 06_plan_d_geography.R
# Plan D — Local Care Mismatch Map (Geography + Digital Engagement)
# Hard trend: block-group map of ED rate, abandonment risk, MyChart inactivity.
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

# =============================================================================
# Block-group level aggregation
# =============================================================================

message("Plan D — aggregating block-group statistics...")

block_stats <- enc |>
  filter(geo_data_available, !is.na(centlat), !is.na(centlon)) |>
  group_by(census_block_fips_clean, centlat, centlon,
           population_value, is_rural_census_block) |>
  summarise(
    n_encounters      = n(),
    ed_rate           = mean(is_ed_visit_flag, na.rm = TRUE),
    abandon_rate      = mean(is_true_abandonment_risk, na.rm = TRUE),
    mychart_inactive  = 1 - mean(my_chart_active, na.rm = TRUE),
    .groups           = "drop"
  ) |>
  filter(n_encounters >= 5) |>
  mutate(
    risk_score = (
      scales::rescale(ed_rate, to = c(0, 1)) +
      scales::rescale(abandon_rate, to = c(0, 1)) +
      scales::rescale(mychart_inactive, to = c(0, 1))
    ) / 3
  )

message("Plan D — block groups with sufficient data: ", nrow(block_stats))

# ------ (1) Dot map: Kansas block groups ------------------------------------

p1 <- ggplot(block_stats, aes(x = centlon, y = centlat)) +
  geom_point(
    aes(color = ed_rate, size = abandon_rate,
        alpha = if_else(is_rural_census_block, 1.0, 0.5)),
    shape = 16
  ) +
  scale_color_gradientn(
    colors = c("#ffffcc", "#fd8d3c", "#d73027"),
    labels = percent_format(),
    name   = "ED Visit Rate"
  ) +
  scale_size_area(max_size = 10, name = "Abandon Risk Rate",
                  labels = percent_format()) +
  scale_alpha_identity() +
  coord_fixed(ratio = 1.3) +
  labs(
    title    = "Plan D — Kansas Block-Group Risk Map",
    subtitle = "Dot color = ED rate | Dot size = abandonment risk | Opacity = rural (opaque) vs urban",
    x        = "Longitude",
    y        = "Latitude",
    caption  = "Source: SVH encounters + US Census TIGER block groups"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank())

ggsave(file.path(PLOT_DIR, "plan_d_kansas_dot_map.png"), p1,
       width = 12, height = 7, dpi = 150)

# ------ (2) Composite risk score: rural vs urban ----------------------------

p2 <- ggplot(block_stats, aes(
    x    = factor(is_rural_census_block, labels = c("Urban", "Rural")),
    y    = risk_score,
    fill = factor(is_rural_census_block, labels = c("Urban", "Rural"))
  )) +
  geom_violin(trim = TRUE, alpha = 0.65) +
  geom_boxplot(width = 0.12, outlier.shape = NA) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 4, fill = "white") +
  scale_fill_manual(values = c("Urban" = "#4575b4", "Rural" = "#d73027")) +
  labs(
    title    = "Plan D — Composite Risk Score: Rural vs Urban Block Groups",
    subtitle = "Risk score = avg of normalised ED rate, abandonment rate, MyChart inactivity",
    x        = NULL,
    y        = "Composite Risk Score (0–1)",
    fill     = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave(file.path(PLOT_DIR, "plan_d_rural_urban_risk.png"), p2,
       width = 8, height = 6, dpi = 150)

# ------ (3) Scatter: MyChart inactivity vs ED rate --------------------------

p3 <- ggplot(block_stats, aes(x = mychart_inactive, y = ed_rate)) +
  geom_point(
    aes(color = is_rural_census_block, size = n_encounters),
    alpha = 0.7
  ) +
  geom_smooth(method = "lm", se = TRUE, color = "black", linewidth = 0.8) +
  scale_x_continuous(labels = percent_format()) +
  scale_y_continuous(labels = percent_format()) +
  scale_color_manual(values = c("TRUE" = "#d73027", "FALSE" = "#4575b4"),
                     labels = c("TRUE" = "Rural", "FALSE" = "Urban"),
                     name   = "Block Type") +
  scale_size_area(max_size = 12, name = "Encounters") +
  labs(
    title    = "Plan D — MyChart Inactivity vs ED Rate by Block Group",
    subtitle = "Each point = one census block group | Line = OLS fit",
    x        = "MyChart Inactivity Rate",
    y        = "ED Visit Rate"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(PLOT_DIR, "plan_d_mychart_vs_ed_scatter.png"), p3,
       width = 10, height = 6, dpi = 150)

# ------ (4) Top-10 highest risk clusters ------------------------------------

top_clusters <- block_stats |>
  arrange(desc(risk_score)) |>
  slice_head(n = 10) |>
  mutate(label = paste0(round(centlat, 3), "°N, ", round(abs(centlon), 3), "°W"))

p4 <- ggplot(top_clusters, aes(
    x    = reorder(label, risk_score),
    y    = risk_score,
    fill = is_rural_census_block
  )) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(ymin = ed_rate, ymax = abandon_rate), width = 0.2, color = "grey30") +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#d73027", "FALSE" = "#4575b4"),
                    labels = c("TRUE" = "Rural", "FALSE" = "Urban")) +
  scale_y_continuous(labels = percent_format()) +
  labs(
    title    = "Plan D — Top 10 Highest-Risk Block Groups",
    subtitle = "Bars = composite risk score | Error bars = ED rate (low) to abandon rate (high)",
    x        = "Block Group (lat/lon centroid)",
    y        = "Composite Risk Score",
    fill     = "Block Type"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(PLOT_DIR, "plan_d_top10_clusters.png"), p4,
       width = 11, height = 6, dpi = 150)

message("Plan D plots saved to ", PLOT_DIR)
message("06_plan_d_geography.R complete.")
