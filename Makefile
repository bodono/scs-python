PYTHON = python3.12
.PHONY: test

test:
	# replicate github CI job for packaging
	rm -rf env/
	$(PYTHON) -m venv env
	env/bin/pip install setuptools
	env/bin/python setup.py install
	env/bin/pip install pytest
	pytest