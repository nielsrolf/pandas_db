cd /home/ubuntu/pandas_db
git log > oldlog
git pull
git log > newlog
if [ "$(diff oldlog newlog)" != "" ]; then
    source ec2/restart.sh
fi
