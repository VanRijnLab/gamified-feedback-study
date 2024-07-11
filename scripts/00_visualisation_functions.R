require(ggplot2)
require(dplyr)
require(data.table)
require(tidyr)
require(lme4)
require(lmerTest)
require(flextable)



# ggplot2 theme -----------------------------------------------------------

theme_paper <- theme_classic(base_size = 12) + 
  theme(axis.text = element_text(colour = "black"),
        panel.grid.major.y = element_line())

col_condition <- c("#e69f00", # Control
                   "#56b4e9", # Points  
                   "#009e73" # Progress bar
)



# Data plotting -----------------------------------------------------------

plot_data <- function (d, y_mean, y_se, y_lab) {
  
  dodge_width <- .1
  
  y_mean <- enquo(y_mean)
  y_se <- enquo(y_se)
  
  p <- ggplot(d, aes(x = block, y = !!y_mean, group = exp_order)) +
    geom_line(aes(lty = exp_order), position = position_dodge(width = dodge_width)) +
    geom_errorbar(aes(ymin = !!y_mean - !!y_se, ymax = !!y_mean + !!y_se, colour = condition),
                  width = 0,
                  alpha = .5,
                  position = position_dodge(width = dodge_width)) +
    geom_point(aes(colour = condition, pch = condition),
               size = 2,
               position = position_dodge(width = dodge_width)) +
    scale_colour_manual(values = col_condition) +
    guides(lty = "none") +
    labs(x = "Block",
         y = y_lab,
         colour = "Condition",
         pch = "Condition") +
    theme_paper
  
  p
  
}

plot_model_fit <- function(m, d_m, exp_trans = FALSE, y_lims = NULL, y_breaks = NULL, y_lab = "", plot_title = "") {
  m_fit <- tidyr::expand(d_m, nesting(block, condition, gamified, gamified_first, exp_group, gamified_first_c, exp_group_c))
  
  m_fit <- cbind(m_fit, pred_val = predict(m, newdata = m_fit, re.form = NA, type = "response")) |>
    mutate(exp_order = case_when(
      gamified_first == 0 & exp_group == "score" ~ "Control—Score",
      gamified_first == 0 & exp_group == "both" ~ "Control—Both",
      gamified_first == 1 & exp_group == "score" ~ "Score—Control",
      gamified_first == 1 & exp_group == "both" ~ "Both—Control",
    )) |>
    mutate(type = ifelse(gamified, "Gamified", "Control"))
  
  if (exp_trans == TRUE) {
    m_fit$pred_val <- exp(m_fit$pred_val)
  }
  
  print(m_fit)
  
  dodge_width <- .1
  
  p <- ggplot(m_fit, aes(x = block, y = pred_val, group = exp_order)) +
    geom_line(aes(lty = exp_order),
              position = position_dodge(width = dodge_width)) +
    geom_point(aes(colour = condition, pch = condition), 
               size = 2,
               position = position_dodge(width = dodge_width)) +
    scale_y_continuous(limits = y_lims, breaks = y_breaks) +
    scale_colour_manual(values = col_condition) +
    guides(lty = "none") +
    labs(x = "Block",
         y = y_lab,
         colour = "Condition",
         pch = "Condition",
         title = plot_title) +
    theme_paper
  
  p
  
}


print_model_table <- function(m) {
  m_coef <- as.data.frame(summary(m)$coefficients)
  data.table::setDT(m_coef, keep.rownames = TRUE)
  m_coef$rn <- c("Intercept (Control)",
                 "Gamification",
                 "Type of gamification on Control (Progress bar - Points)",
                 "Type of gamification (Progress bar - Points)",
                 "Block on Control (2 - 1)",
                 "Block on gamification (1 - 2)",
                 "Type of gamification on Control; block difference (Block 2 - Block 1)",
                 "Type of gamification; block difference (Block 1 - Block 2)")
  
  # Switch order of rows
  m_coef <- m_coef[c(1, 2, 4, 3, 6, 5, 8, 7)]
  
  
  if(summary(m)$objClass == "glmerMod") {
    # Format p-values
    m_coef$`Pr(>|z|)` <- format.pval(round(m_coef$`Pr(>|z|)`, 3), eps = .001, digits = 3, flag = "0")
    m_coef$`Pr(>|z|)` <- sub('^(<)?0[.]', '\\1.', m_coef$`Pr(>|z|)`) # Remove leading zero
    
    
    if(summary(m)$family == "Gamma") {
      return(
        flextable::flextable(m_coef) |>
          flextable::colformat_double(j = c("Estimate", "Std. Error"), digits = 3) |>
          flextable::colformat_double(j = c("t value"), digits = 2) |>
          flextable::align(j = "Pr(>|z|)", align = "right") |>
          flextable::set_header_labels(rn = "Effect",
                                       `Std. Error` = "SE",
                                       `t value` = "t-value",
                                       `Pr(>|z|)` = "p-value") |>
          flextable::autofit()
      )
    }
    
    return(
      flextable::flextable(m_coef) |>
        flextable::colformat_double(j = c("Estimate", "Std. Error"), digits = 2) |>
        flextable::colformat_double(j = c("z value"), digits = 2) |>
        flextable::align(j = "Pr(>|z|)", align = "right") |>
        flextable::set_header_labels(rn = "Effect",
                                     `Std. Error` = "SE",
                                     `z value` = "z-value",
                                     `Pr(>|z|)` = "p-value") |>
        flextable::autofit()
    )
  }
  
  # Format p-values
  m_coef$`Pr(>|t|)` <- format.pval(round(m_coef$`Pr(>|t|)`, 3), eps = .001, digits = 3, flag = "0")
  m_coef$`Pr(>|t|)` <- sub('^(<)?0[.]', '\\1.', m_coef$`Pr(>|t|)`) # Remove leading zero
  
  return(
    flextable::flextable(m_coef) |>
      flextable::colformat_double(j = c("Estimate", "Std. Error"), digits = 3) |>
      flextable::colformat_double(j = c("df", "t value"), digits = 2) |>
      flextable::align(j = "Pr(>|t|)", align = "right") |>
      flextable::set_header_labels(rn = "Effect",
                                   `Std. Error` = "SE",
                                   `t value` = "t-value",
                                   `Pr(>|t|)` = "p-value") |>
      flextable::autofit()
  )
}
