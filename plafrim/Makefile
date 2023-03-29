.PHONY: clean preprocess regression sigtest scanreports regress-and-test

HOST=$(shell hostname | grep -oh plafrim)
USER=$(shell whoami)

ifeq ($(HOST), plafrim)
# Within Plafrim
	HOST=$(shell hostname)

	DATA_ROOT=/beegfs/$(USER)/MASC-MEG
	REPORT_ROOT=/beegfs/$(USER)/MEG-analysis/reports
else
# Local (outside plafrim)
	HOST=$(shell hostname)
	DATA_ROOT=../../data
	REPORT_ROOT=../../reports
endif

clean:
	rm -f jobs/*.sl jobs/*.sh jobs/config/*.txt stdout/* stderr/*

pyinstall: requirements.txt
	guix install python-numpy python-pandas python-scipy; python3 -m pip install -r requirements.txt

whereami:
	echo "$(USER) at $(HOST)"; echo "Data source: $(DATA_ROOT)"; echo "Reports source $(REPORT_ROOT)"

### PREPROCESS

preprocess: $(DATA_ROOT)/meg/*.npy

$(DATA_ROOT)/meg/*.npy: jobs/preprocess.sl
	sbatch jobs/preprocess.sl

jobs/preprocess.sl: jobs/config/subject_list.txt
	python3 src/generate_job.py preprocess -r $(DATA_ROOT)/raw -o $(DATA_ROOT)/meg/

### REGRESSION

regression: REPORT_ROOT/*/*/r2s.npy

REPORT_ROOT/*/*/r2s.npy: jobs/regression*.sl
	chmod +x ./jobs/regression*; ./jobs/regression-main.sh

jobs/regression*.sl: jobs/config/feature_list.txt jobs/config/subject_list.txt
	python3 src/generate_job.py regression -o $(REPORT_ROOT) --meg_data $(DATA_ROOT)/meg --feat_data $(DATA_ROOT)/vectors

### SIGTEST

sigtest: REPORT_ROOT/*/*/sig.npy

REPORT_ROOT/*/*/sig.npy: jobs/sigtest*.sl
	chmod +x ./jobs/sigtest*; ./jobs/sigtest-main.sh

jobs/sigtest*.sl: jobs/config/reg_reports.txt
	python3 src/generate_job.py sigtest -o $(REPORT_ROOT)


### REGRESSION AND SIGTEST

regress-and-test: jobs/regress-test*.sl
	chmod +x ./jobs/regress-test*; ./jobs/regress-test-main.sh

jobs/regress-test*.sl: jobs/config/feature_list.txt jobs/config/subject_list.txt
	python3 src/generate_job.py regress-test -o $(REPORT_ROOT) --meg_data $(DATA_ROOT)/meg --feat_data $(DATA_ROOT)/vectors


### FILE LISTS (for convenience, to track exprimentations and data sources)

scanreports: jobs/config/reg_reports.txt jobs/config/concat_reports.txt

jobs/config/feature_list.txt:
	./jobs/config/gen_feature_list.sh $(DATA_ROOT)/vectors ./jobs/config/feature_list.txt

jobs/config/subject_list.txt:
	./jobs/config/gen_subject_list.sh $(DATA_ROOT)/meg ./jobs/config/subject_list.txt

jobs/config/reg_reports.txt jobs/config/concat_reports.txt:
	./jobs/config/scan_reports.sh $(REPORT_ROOT) ./jobs/config/reg_reports.txt  ./jobs/config/concat_reports.txt
