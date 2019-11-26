cd comet

bash scripts/setup/get_atomic_data.sh
bash scripts/setup/get_conceptnet_data.sh
bash scripts/setup/get_model_files.sh

pip install tensorflow
pip install ftfy==5.1
pip install spacy
python -m spacy download en
pip install tensorboardX
pip install tqdm
pip install pandas
pip install ipython

python scripts/data/make_atomic_data_loader.py
python scripts/data/make_conceptnet_data_loader.py

echo Please manually download the pretrained model from: https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB and untar it into comet/pretrained_model
