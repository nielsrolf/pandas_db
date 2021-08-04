cd /home/ubuntu/pandas_db
git pull
pip install -e .
sudo kill -9 $(sudo lsof -t -i:8050)
pandasdb metrics2
