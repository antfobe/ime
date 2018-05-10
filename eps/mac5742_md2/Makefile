CODE_DIR=contextswitch

default: run

run: hyper_perf
	$(MAKE) -C $(CODE_DIR) bench

all: hyper_perf
	$(MAKE) -C $(CODE_DIR) all 

clean: hyper_perf
	$(MAKE) -C $(CODE_DIR) clean

license:
	-less LICENSE

.PHONY: hyper_perf
