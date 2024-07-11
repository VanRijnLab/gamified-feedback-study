.PHONY: all clean data learn test survey

all: learn test survey

clean:
	rm -rf output/* data/processed/*


# Data preparation
data: scripts/01_prepare_data.R
	Rscript scripts/01_prepare_data.R

# Learning session analysis
learn: scripts/02_analyse_learning_session.Rmd
	Rscript -e "rmarkdown::render('scripts/02_analyse_learning_session.Rmd', output_format = 'all')"
	rm -rf output/02_analyse_learning_session_files
	mv scripts/02_analyse_learning_session.nb.html scripts/02_analyse_learning_session.html scripts/02_analyse_learning_session.md scripts/02_analyse_learning_session_files output
	
# Posttest analysis
test: scripts/03_analyse_posttest.Rmd
	Rscript -e "rmarkdown::render('scripts/03_analyse_posttest.Rmd', output_format = 'all')"
	rm -rf output/03_analyse_posttest_files
	mv scripts/03_analyse_posttest.nb.html scripts/03_analyse_posttest.html scripts/03_analyse_posttest.md scripts/03_analyse_posttest_files output
	
# Survey analysis
survey: scripts/04_analyse_survey.Rmd
	Rscript -e "rmarkdown::render('scripts/04_analyse_survey.Rmd', output_format = 'all')"
	rm -rf output/04_analyse_survey_files
	mv scripts/04_analyse_survey.nb.html scripts/04_analyse_survey.html scripts/04_analyse_survey.md scripts/04_analyse_survey_files output