wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
echo "export PANDAS_DB_PATH=/home/ubuntu/data" >> ~/.bashrc
source ~/.bashrc
git clone https://github.com/nielsrolf/pandas_db.git
