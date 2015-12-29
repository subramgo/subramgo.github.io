---
layout: post
title: Analtyics Node Setup
comments: true
---

Analtyics Node 
-------------

Recently I had setup a cluster for analytics purpose.

> The machine has the following softwares installed.
> 
- Java 7
- Base R
- R-Studio
- H2O Cluster

R Studio can be accessed through browser

    http://<ip address>:8787/


In order to access H2O cluster, ssh tunnelling has be enabled from the client. Install H2O section has the details of tunnelling. Curently there are two 


Install Java
-----

    sudo apt-get install python-software-properties
    sudo add-apt-repository ppa:webupd8team/java
    sudo apt-get update


Oracle JDK 7

    sudo apt-get install oracle-java7-installer

Install R
-------------------

    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 
    echo "deb http://cran.cnr.berkeley.edu/bin/linux/ubuntu precise/" >> /etc/apt/sources.list
    sudo apt-get update 
	sudo apt-get install r-base

Type R to check installation


Install R Studio Server
--------------------------

64 bit version

    $ sudo apt-get install gdebi-core
    $ sudo apt-get install libapparmor1 # Required only for Ubuntu, not Debian
    $ wget http://download2.rstudio.org/rstudio-server-0.98.994-amd64.deb
    $ sudo gdebi rstudio-server-0.98.994-amd64.deb

Check by accessing http://<Ip address>:8787/
Use the linux username and password

Install H2o
---

Download H2o

http://s3.amazonaws.com/h2o-release/h2o/rel-kramer/1/index.html?_sm_au_=iVV4RHKSHKjqQMBh



    unzip h2o-2.4.6.1.zip`

Run h2o


    java -Xmx2g -jar h20.jar
    http://localhost:54322/

To access H2o from a remote machine. Setup a ssh tunnel in remote machine


    ssh -L 55555:localhost:54321 tyconet@10.47.86.77

Access from remote machine using

    http://localhost:55555

Install H2o cluster
----

Unzip h2o zip file in all the nodes of the cluster
Create a nodes.txt file with the ip address of nodes, in this case it will only one entry

    <ip address>:54321
    <ip address>:54321
    
Start the H2o cluster in all the nodes

    java -Xmx4g -jar h2o.jar -flatfile nodes.txt -port 54321


