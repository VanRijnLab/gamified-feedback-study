library(here)
library(dplyr)
library(tidyr)
library(forcats)


# Helper functions --------------------------------------------------------

add_condition_labels <- function (data) {
  condition_labels <- tibble(condition = c("control", "score", "both"),
                             condition_label = factor(
                               c("Control", "Points", "Progress bar"),
                               levels = c("Control", "Points", "Progress bar")
                             )
  )
  
  left_join(data, condition_labels, by = "condition")
}

anonymise_subjects <- function (data) {
  set.seed(0)
  data |>
    mutate(subject = fct_anon(subject, prefix = "subj"))
}

# Load data ---------------------------------------------------------------

# Trial-level data: practice sessions
d_learn_full <- read.csv(here("data", "raw", "07_Pract_PPxWxT.csv"))

# Trial-level data: delayed recall test
d_test_full <- read.csv(here("data", "raw", "07_Test_PPxWxT.csv"))

# Participant-level data: practice, test, questionnaire
d_agg_full <- read.csv(here("data", "raw", "07_PractTestQuestion_PP_lf.csv"))

# Participant-level data: preferences
d_preference <- read.csv(here("data", "raw", "07_PractTestQuestion_PP_wf.csv"))



# Format columns ----------------------------------------------------------

# Learning data
d_learn <- d_learn_full |>
  add_condition_labels() |>
  transmute(subject = as.character(GlobalIDpract),
            exp_group = ExpGroup,
            gamified_first = as.logical(GamifiedFirst),
            block = as.factor(Block),
            condition = condition_label,
            gamified = condition != "Control",
            trial = trial_index,
            study_trial = as.logical(study_trial),
            fact = paste(stimulus_list, fact_id, sep = "_"),
            start_time,
            correct = as.logical(correct),
            rt = onsetRT,
            feedback_score
  ) |>
  anonymise_subjects() |>
  arrange(subject, block, trial)

# Ensure that all participants completed exactly two blocks of practice
stopifnot(
  d_learn |>
    group_by(subject, condition) |>
    filter(trial == 3) |> # First trial in a learning session
    group_by(subject) |>
    filter(n() != 2) |>
    nrow() == 0
)


# Test data
d_test <- d_test_full |>
  add_condition_labels() |>
  filter(conditionUnpr != "Unpracticed") |> # Only keep items that were practiced
  transmute(
    subject = as.character(GlobalIDpract),
    exp_group = ExpGroup,
    gamified_first = as.logical(GamifiedFirst),
    block = as.factor(Block),
    condition = condition_label,
    gamified = condition != "Control",
    trial = Trial.Number,
    fact = french_word,
    correct = as.logical(LD_Correct),
    rt = Reaction.Time
  ) |>
  anonymise_subjects() |>
  arrange(subject, block, trial)


# Survey data
d_survey <- d_agg_full |>
  rename(condition = Condition) |>
  add_condition_labels() |>
  filter(condition != "Unpracticed") |>
  filter(ExpGroup != "") |>
  rowwise() |>
  mutate(perception_relevance = mean(c(Clear, Valuable, HogeScore))) |>
  ungroup() |>
  transmute(
    subject = as.character(GlobalIDpract),
    exp_group = ExpGroup,
    gamified_first = as.logical(GamifiedFirst),
    block = as.factor(Block),
    condition = condition_label,
    gamified = condition != "Control",
    motivation_enjoyment = Enjoyment,
    motivation_competence = Competence,
    motivation_value = Value,
    perception_goalsetting = Doel,
    perception_goalstress = Stressvol,
    perception_performwell = GoedPresteren,
    perception_distraction = Distracting,
    perception_relevance
  ) |>
  left_join(select(d_preference, GlobalIDpract, Preference), by = c("subject" = "GlobalIDpract")) |>
  rename(preference = Preference) |>
  anonymise_subjects()

d_survey_long <- d_survey |>
  pivot_longer(cols = motivation_enjoyment:perception_relevance,
               names_to = c("category", "question"),
               names_sep = "\\_",
               values_to = "response")

# Save data ---------------------------------------------------------------

saveRDS(d_learn, here("data", "processed", "d_learn.rds"))
saveRDS(d_test, here("data", "processed", "d_test.rds"))
saveRDS(d_survey_long, here("data", "processed", "d_survey.rds"))