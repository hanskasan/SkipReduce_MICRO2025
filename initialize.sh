apt-get update
apt-get install --assume-yes vim
python -m pip install --upgrade pip

pip install 'accelerate>=0.26.0'
pip install datasets
pip install evaluate
pip install scikit-learn
pip install pillow==10.2.0

# pip install torchvision==0.20.0
# pip install torchao
# pip install bitsandbytes

apt install --assume-yes build-essential devscripts debhelper fakeroot

huggingface-cli login --token "<YOUR TOKEN>"
# git config --global user.email "<YOUR EMAIL>"
# git config --global user.name "<YOUR USERNAME>"
