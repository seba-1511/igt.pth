
.PHONY: all

all:
	python tests/test_wrapper_cnn.py

test:
	python tests/test_igt.py
	python tests/test_heavyball_igt.py
	python tests/test_nesterov_igt.py
	python tests/test_igt_wrapper.py
	python tests/test_adam_igt.py
