

Installation has not been thoroughly tested on different systems, but on Ubuntu with virtual environments this worked for me:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 

cd src/cell_models/
nrnivmodl
cd ../..

python3 make_all_figures.py