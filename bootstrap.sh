#!/bin/bash
duty=${1}
JUPYTER_PASSWORD=${2:-"root"}
set -e
sudo apt-get update;
sudo add-apt-repository -y ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install -y openjdk-8-jre-headless scala openssh-server openssh-client syslinux-utils python3-pip socat;

awk 'NR>1 {print $NF}' /etc/hosts | grep -v 'master' > /local/host_list
# docker
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get -y install docker-ce docker-ce-cli containerd.io
# spark
wget https://archive.apache.org/dist/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz;
tar xvf spark-2.4.5-bin-hadoop2.7.tgz;
sudo mv spark-2.4.5-bin-hadoop2.7 /usr/local/spark;
echo export PATH="$PATH:/usr/local/spark/bin" > ~/.bashrc;
echo export SPARK_HOME="/usr/local/spark" >> ~/.bashrc;
sudo cp /usr/local/spark/conf/spark-env.sh.template /usr/local/spark/conf/spark-env.sh;
sudo cp /usr/local/spark/conf/slaves.template /usr/local/spark/conf/slaves;

pip3 install -r /local/repository/requirements.txt;



# setup hadoop
HADOOP_HOME=/local/hadoop
mkdir $HADOOP_HOME
HOST_LIST_PATH=/local/host_list

JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::")
awk 'NR>1 {print $NF}' /etc/hosts | grep -v 'master' > $HOST_LIST_PATH


cd /mnt
wget https://archive.apache.org/dist/hadoop/core/hadoop-2.7.3/hadoop-2.7.3.tar.gz
tar -xvf hadoop-2.7.3.tar.gz
cp -r /mnt/hadoop-2.7.3/. $HADOOP_HOME/.

sudo cp $HOST_LIST_PATH $HADOOP_HOME/etc/hadoop/slaves
echo "master" | sudo tee $HADOOP_HOME/etc/hadoop/workers
echo "export HADOOP_HOME=$HADOOP_HOME" | sudo tee -a ~/.bashrc
echo "export HADOOP_PREFIX=$HADOOP_HOME" | sudo tee -a ~/.bashrc
echo "export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin" | sudo tee -a ~/.bashrc
source ~/.bashrc
echo "export JAVA_HOME=$JAVA_HOME" | sudo tee -a $HADOOP_HOME/etc/hadoop/hadoop-env.sh
cp /local/repository/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml
cp /local/repository/yarn-site.xml /local/hadoop/etc/hadoop/yarn-site.xml
cp /local/repository/hdfs-site.xml /local/hadoop/etc/hadoop/hdfs-site.xml
# Spark ips configs
ips=($(ip -4 addr | grep -oP '(?<=inet\s)\d+(\.\d+){3}'))
for ip in "${ips[@]}"
do
    if [[ $ip == *"10."* ]]; then
        echo export LOCAL_IP=$ip >> ~/._bashrc;
        LOCAL_IP=$ip
    fi
done


master_ip=$(gethostip -d master);
echo "export master_ip=$master_ip" | sudo tee -a ~/.bashrc
echo "export SPARK_MASTER_HOST=$master_ip" | sudo tee -a /usr/local/spark/conf/spark-env.sh;
echo "export SPARK_LOCAL_IP=$LOCAL_IP" | sudo tee -a /usr/local/spark/conf/spark-env.sh;
echo "export PYSPARK_PYTHON=python3.6" | sudo tee -a /usr/local/spark/conf/spark-env.sh;



# Jupyter extension configs
sudo /usr/local/bin/jupyter contrib nbextension install --system ;
sudo /usr/local/bin/jupyter nbextensions_configurator enable --system ;
sudo /usr/local/bin/jupyter nbextension enable code_prettify/code_prettify --system ;
sudo /usr/local/bin/jupyter nbextension enable execute_time/ExecuteTime --system ;
sudo /usr/local/bin/jupyter nbextension enable collapsible_headings/main --system ;
sudo /usr/local/bin/jupyter nbextension enable freeze/main --system ;
sudo /usr/local/bin/jupyter nbextension enable spellchecker/main --system ;

# Jupyter password
mkdir -p ~/.jupyter;
HASHED_PASSWORD=$(python3.6 -c "from notebook.auth import passwd; print(passwd('$JUPYTER_PASSWORD'))");
echo "c.NotebookApp.password = u'$HASHED_PASSWORD'" >~/.jupyter/jupyter_notebook_config.py;
echo "c.NotebookApp.open_browser = False" >>~/.jupyter/jupyter_notebook_config.py;


cp ~/._bashrc /local/.bashrc
cp ~/._bashrc /etc/profile.d/spark.sh
source ~/.bashrc
# Running Spark deamons
if [ "$duty" = "m" ]; then
	sudo bash /usr/local/spark/sbin/start-master.sh
  $HADOOP_PREFIX/bin/hdfs namenode -format "spark_cluster"
  $HADOOP_PREFIX/sbin/hadoop-daemon.sh --script hdfs start namenode
  # $HADOOP_PREFIX/sbin/yarn-daemon.sh start resourcemanager
  # $HADOOP_PREFIX/sbin/yarn-daemons.sh start nodemanager
	sudo nohup socat TCP-LISTEN:8081,fork TCP:${LOCAL_IP}:8080 > /dev/null 2>&1 &
	sudo nohup socat TCP-LISTEN:4041,fork TCP:${LOCAL_IP}:4040 > /dev/null 2>&1 &
  sudo nohup socat TCP-LISTEN:8089,fork TCP:${LOCAL_IP}:8088 > /dev/null 2>&1 &
	sudo nohup docker run --init -p 3000:3000 -v "/:/home/project:cached" theiaide/theia-python:next > /dev/null 2>&1 &
	sudo nohup jupyter notebook --no-browser --allow-root --ip 0.0.0.0 --notebook-dir=/ > /dev/null 2>&1 &


elif [ "$duty" = "s" ]; then
	sudo bash /usr/local/spark/sbin/start-slave.sh $master_ip:7077
	sudo nohup socat TCP-LISTEN:8082,fork TCP:${LOCAL_IP}:8081 > /dev/null 2>&1 &
  $HADOOP_PREFIX/sbin/hadoop-daemons.sh --script hdfs start datanode
  $HADOOP_PREFIX/sbin/yarn-daemons.sh start nodemanager
fi
echo "Bootstraping complete"



sudo nohup socat TCP-LISTEN:8083,fork TCP:${LOCAL_IP}:8082 > /dev/null 2>&1 &
