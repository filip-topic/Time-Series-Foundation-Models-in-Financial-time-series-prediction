# MSc_dissertation

Version of gluonts has to be 0.14.0 in the requirements.txt in lag-llama submodule


Problems when installing "gluonts" package for the DeepAR model. Solved by running:

pip install numpy==1.23.5
pip install mxnet -f https://dist.mxnet.io/python/cpu

-there was "error building numpy wheel", but commands above solve it
