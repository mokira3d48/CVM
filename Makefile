install:
	python3 -m venv .venv
	.venv/bin/python3 --version
	.venv/bin/python3 -m pip install --upgrade pip
	.venv/bin/python3 -m pip install torch==2.6.0 torchvision --index-url "https://download.pytorch.org/whl/cpu" && \
	.venv/bin/python3 -m pip install -r requirements.txt
	chmod +x vae_train
	chmod +x vae_fine_tuning

gpu_install:
	python3 -m venv .venv
	.venv/bin/python3 --version
	.venv/bin/python3 -m pip install --upgrade pip
	.venv/bin/python3 -m pip install torch==2.6.0 torchvision && \
	.venv/bin/python3 -m pip install -r requirements.txt
	chmod +x vae_train
	chmod +x vae_fine_tuning

test:
	pytest tests


pep8:
	# Don't remove their commented follwing command lines:
    # autopep8 --in-place --aggressive --aggressive --recursive .
    # autopep8 --in-place --aggressive --aggressive example.py

