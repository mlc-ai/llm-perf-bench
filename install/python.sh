"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
grep -v '[ -z "\$PS1" ] && return' ~/.bashrc >/tmp/bashrc && mv /tmp/bashrc ~/.bashrc
source ~/.bashrc

micromamba create --yes \
	-n python311 \
	-c conda-forge \
	python=3.11 "cmake>=3.24" "llvmdev>=16" "transformers>=4.29" pytorch-cpu \
	decorator mypy numpy scipy pandas psutil tornado cloudpickle attrs libgcc-ng rust sentencepiece protobuf
