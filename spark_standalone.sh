#!/usr/bin/env bash
duty=${1}
set -e
sudo apt-get update;
sudo add-apt-repository -y ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install -y openjdk-8-jre-headless scala openssh-server openssh-client syslinux-utils python3-pip socat;
wget https://archive.apache.org/dist/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz;
tar xvf spark-2.4.5-bin-hadoop2.7.tgz;
sudo mv spark-2.4.5-bin-hadoop2.7 /usr/local/spark;
sudo chown -R $USER:$USER /usr/local/spark
echo export PATH="$PATH:/usr/local/spark/bin" >> ~/.bashrc;
echo export SPARK_HOME="/usr/local/spark" >> ~/.bashrc;
cp /usr/local/spark/conf/spark-env.sh.template /usr/local/spark/conf/spark-env.sh;
cp /usr/local/spark/conf/slaves.template /usr/local/spark/conf/slaves;

sudo python3.7 -m pip install --upgrade --force-reinstall setuptools

sudo python3.7 -m pip install -r ./requirements.txt;

HADOOP_HOME=/local/hadoop
sudo mkdir -p $HADOOP_HOME
sudo chown -R $USER:$USER $HADOOP_HOME
HOST_LIST_PATH=/local/gphost_list
JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::")

wget https://archive.apache.org/dist/hadoop/core/hadoop-2.7.3/hadoop-2.7.3.tar.gz
tar -xvf hadoop-2.7.3.tar.gz 
cp -r hadoop-2.7.3/. $HADOOP_HOME/.
cp $HOST_LIST_PATH $HADOOP_HOME/etc/hadoop/slaves

echo "master" | tee $HADOOP_HOME/etc/hadoop/workers
echo "export HADOOP_HOME=$HADOOP_HOME" | sudo tee -a ~/.bashrc
echo "export HADOOP_PREFIX=$HADOOP_HOME" | sudo tee -a ~/.bashrc
echo "export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin" | sudo tee -a ~/.bashrc
source ~/.bashrc
echo "export JAVA_HOME=$JAVA_HOME" | tee -a $HADOOP_HOME/etc/hadoop/hadoop-env.sh
cp core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml
cp yarn-site.xml /local/hadoop/etc/hadoop/yarn-site.xml
cp hdfs-site.xml /local/hadoop/etc/hadoop/hdfs-site.xml
# Spark ips configs
ips=($(ip -4 addr | grep -oP '(?<=inet\s)\d+(\.\d+){3}'))
for ip in "${ips[@]}"
do
    if [[ $ip == *"10."* ]]; then
        echo "export LOCAL_IP=$ip" | sudo tee -a ~/.bashrc;
        export LOCAL_IP=$ip
    fi
done
echo "export master_ip=$(gethostip -d master);" | sudo tee -a ~/.bashrc;
export master_ip=$(gethostip -d master);
echo "export SPARK_MASTER_HOST=$master_ip" | sudo tee -a /usr/local/spark/conf/spark-env.sh;
echo "export SPARK_LOCAL_IP=$LOCAL_IP" | sudo tee -a /usr/local/spark/conf/spark-env.sh;
echo "export PYSPARK_PYTHON=python3.7" | sudo tee -a /usr/local/spark/conf/spark-env.sh;

if [ "$duty" = "m" ]; then
	bash /usr/local/spark/sbin/start-master.sh
  $HADOOP_PREFIX/bin/hdfs namenode -format "spark_cluster"
  $HADOOP_PREFIX/sbin/hadoop-daemon.sh --script hdfs start namenode
  # $HADOOP_PREFIX/sbin/yarn-daemon.sh start resourcemanager
  # $HADOOP_PREFIX/sbin/yarn-daemons.sh start nodemanager
	sudo nohup socat TCP-LISTEN:8081,fork TCP:${LOCAL_IP}:8080 > /dev/null 2>&1 &
	sudo nohup socat TCP-LISTEN:4041,fork TCP:${LOCAL_IP}:4040 > /dev/null 2>&1 &
  sudo nohup socat TCP-LISTEN:7078,fork TCP:${LOCAL_IP}:7077 > /dev/null 2>&1 &
elif [ "$duty" = "s" ]; then
	bash /usr/local/spark/sbin/start-slave.sh $master_ip:7077
	sudo nohup socat TCP-LISTEN:8082,fork TCP:${LOCAL_IP}:8081 > /dev/null 2>&1 &	
  $HADOOP_PREFIX/sbin/hadoop-daemons.sh --script hdfs start datanode
  $HADOOP_PREFIX/sbin/yarn-daemons.sh start nodemanager
fi


# gpfdist -d /mnt/criteo/unload -p 8101 -l /mnt/nfs/logs/gpfdist_logs/$WORKER_NUMBER &
# check hadoop: http://localhost:50070/
# hdfs dfsadmin -report
# modify pg_hba.conf on master and workers;
# host     all         all         10.10.1.1/24       trust