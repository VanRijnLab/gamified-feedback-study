library(data.table)
library(dplyr)
library(ggplot2)
library(patchwork)

# Plotting function
plot_model_bf <- function(hypotheses, print_plot = FALSE) {
  
  d <- hypotheses |>
    mutate(lab = case_when(
      Hypothesis == "(gamifiedTRUE) = 0" ~ "Gamification",
      Hypothesis == "(gamifiedFALSE:exp_group_c) = 0" ~ "Type of gamification on Control",
      Hypothesis == "(gamifiedTRUE:exp_group_c) = 0" ~ "Type of gamification",
      Hypothesis == "(gamifiedFALSE:gamified_first_c) = 0" ~ "Block on Control",
      Hypothesis == "(gamifiedTRUE:gamified_first_c) = 0" ~ "Block on gamification",
      Hypothesis == "(gamifiedFALSE:exp_group_c:gamified_first_c) = 0" ~ "Type of gamification on Control; block difference",
      Hypothesis == "(gamifiedTRUE:exp_group_c:gamified_first_c) = 0" ~ "Type of gamification; block difference",
      # Hypotheses for models only looking at gamified conditions:
      Hypothesis == "(exp_group_c) = 0" ~ "Type of gamification",
      Hypothesis == "(gamified_first_c) = 0" ~ "Block on gamification",
      Hypothesis == "(exp_group_c:gamified_first_c) = 0" ~ "Type of gamification; block difference"
    ))

  # Set hypothesis order for plotting
  d$lab <- factor(d$lab, levels = rev(c(
    "Gamification",
    "Type of gamification",
    "Type of gamification on Control",
    "Block on gamification",
    "Block on Control",
    "Type of gamification; block difference",
    "Type of gamification on Control; block difference"
  )))
  
  # Limit Bayes factors to within the range 1/1000 - 1000
  d$Evid.Ratio <- pmin(pmax(d$Evid.Ratio, 1/1000), 1000)
  
  # Create plot
  p <- ggplot(d, aes(x = 1/Evid.Ratio, y = lab, colour = 1/Evid.Ratio)) +
    facet_grid(~ outcome) +
    geom_rect(aes(xmin = 1/1000, xmax = 1000, ymin = -Inf, ymax = Inf), fill = "grey80", colour = NA) +
    geom_rect(aes(xmin = 1/100, xmax = 100, ymin = -Inf, ymax = Inf), fill = "grey85", colour = NA) +
    geom_rect(aes(xmin = 1/30, xmax = 30, ymin = -Inf, ymax = Inf), fill = "grey90", colour = NA) +
    geom_rect(aes(xmin = 1/10, xmax = 10, ymin = -Inf, ymax = Inf), fill = "grey95", colour = NA) +
    geom_rect(aes(xmin = 1/3, xmax = 3, ymin = -Inf, ymax = Inf), fill = "grey100", colour = NA) +
    geom_vline(xintercept = 1, linetype = 2) +
    geom_segment(xend = log10(1)) +
    geom_point() +
    scale_colour_gradient2(low = muted("red"), mid = "grey", high = muted("blue"),
                           midpoint = 1, 
                           transform = "log10",
                           limits = c(1/100, 100),
                           oob = scales::squish) +
    scale_x_log10(breaks = c(1/1000, 1/100, 1/30, 1/10, 1/3, 1, 3, 10, 30, 100, 1000),
                  labels = c("< 1/1000", "1/100", "1/30", "1/10", "1/3", "1", "3", "10", "30", "100", "> 1000"),
                  limits = c(1/1e3, 1e3)) +
    guides(colour = "none") +
    labs(x = expression("Evidence for effect (" * BF[10] * ")"),
      y = NULL) +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.line.y = element_blank(),
          strip.background = element_blank()
    )
  
  if (print_plot) {
    print(p)
  }
  
  return (p)
  
}


# Load hypothesis test data -----------------------------------------------

h_learn_acc <- fread(here("output", "hypothesis_tests", "h_learn_acc.csv"))
h_learn_rt <- fread(here("output", "hypothesis_tests", "h_learn_rt.csv"))
h_learn_score <- fread(here("output", "hypothesis_tests", "h_learn_score.csv"))
h_learn_trials <- fread(here("output", "hypothesis_tests", "h_learn_trials.csv"))
h_learn_words <- fread(here("output", "hypothesis_tests", "h_learn_words.csv"))

h_test_competence <- fread(here("output", "hypothesis_tests", "h_test_competence.csv"))
h_test_enjoyment <- fread(here("output", "hypothesis_tests", "h_test_enjoyment.csv"))
h_test_value <- fread(here("output", "hypothesis_tests", "h_test_value.csv"))
h_test_goalsetting <- fread(here("output", "hypothesis_tests", "h_test_goalsetting.csv"))
h_test_performwell <- fread(here("output", "hypothesis_tests", "h_test_performwell.csv"))
h_test_relevance <- fread(here("output", "hypothesis_tests", "h_test_relevance.csv"))

h_test_acc <- fread(here("output", "hypothesis_tests", "h_test_acc.csv"))
h_test_rt <- fread(here("output", "hypothesis_tests", "h_test_rt.csv"))



# PRACTICE
learn_acc <- h_learn_acc |> mutate(outcome = "Accuracy")
learn_rt <- h_learn_rt |> mutate(outcome = "Reaction time (correct responses)")
learn_score <- h_learn_score |> mutate(outcome = "Total score achieved")
learn_trials <- h_learn_trials |> mutate(outcome = "Trials completed")
learn_words <- h_learn_words |> mutate(outcome = "Words practiced")

hypotheses_learn <- rbind(learn_acc, learn_rt, learn_score, learn_trials, learn_words)
bf_learn <- plot_model_bf(hypotheses_learn)
bf_learn

# SURVEY
survey_competence <- h_test_competence |> mutate(outcome = "Feelings of competence")
survey_enjoyment <- h_test_enjoyment |> mutate(outcome = "Enjoyment")
survey_value <- h_test_value |> mutate(outcome = "Task value")
survey_goalsetting <- h_test_goalsetting |> mutate(outcome = "Goal setting")
survey_performwell <- h_test_performwell |> mutate(outcome = "Trying to perform well")
survey_relevance <- h_test_relevance |> mutate(outcome = "Perceived relevance (gamified only)")

hypotheses_survey <- rbind(survey_competence, survey_enjoyment, survey_value, survey_goalsetting, survey_performwell, survey_relevance)
hypotheses_survey$outcome <- factor(hypotheses_survey$outcome, levels = c("Feelings of competence", "Enjoyment", "Task value", "Goal setting", "Trying to perform well", "Perceived relevance (gamified only)"))
bf_survey <- plot_model_bf(hypotheses_survey)
bf_survey

# POSTTEST
test_acc <- h_test_acc |> mutate(outcome = "Accuracy")
test_rt <- h_test_rt |> mutate(outcome = "Reaction time (correct responses)")

hypotheses_test <- rbind(test_acc, test_rt)
bf_test <- plot_model_bf(hypotheses_test)
bf_test


# Create combined plot
((bf_survey + plot_spacer() + plot_layout(widths = c(6, 0))) /
    (bf_learn + plot_spacer() + plot_layout(widths = c(5, 1))) / 
  (bf_test + plot_spacer() + plot_layout(widths = c(2, 4)))) +
  plot_annotation(tag_levels = list(c("Motivation", "Learning behaviours during practice",
                                      "Learning outcomes on posttest"))) & 
  theme(plot.tag.position = c(0, 1),
        plot.tag = element_text(hjust = 0))

ggsave(here("output", "hypothesis_tests_bayes.png"), width = 15, height = 7.5, dpi = 300)
