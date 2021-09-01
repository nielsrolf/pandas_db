echo "Restarting..."
wget https://pandasdb-ddsp-demo.s3.eu-central-1.amazonaws.com/.local_db.csv -O $PANDAS_DB_PATH/.local_db.csv
mkdir $PANDAS_DB_PATH/tuning
wget https://pandasdb-ddsp-demo.s3.eu-central-1.amazonaws.com/tuning/migrated.csv -O $PANDAS_DB_PATH/tuning/migrated.csv
pip install -e .
python ec2/migrate_pandasdb.py
cp pandas_db/views.json $PANDAS_DB_PATH/.pandas_db_views.json
sudo kill -9 $(sudo lsof -t -i:8050)
# pandasdb metrics2 &
python reports/thesis.py &
