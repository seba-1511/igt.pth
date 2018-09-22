
.PHONY: all

all:
	python tests/test_igt_wrapper.py

test:
	python tests/test_igt.py
	python tests/test_heavyball_igt.py
	python tests/test_nesterov_igt.py
