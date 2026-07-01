FROM cd-docker-hub.sxy5.artifactory.cd-cloud-artifact.tools.huawei.com/driveinsight/training/pytorch2.7.1-cuda12.4-py3.10-base:latest


COPY ./requirements.txt /home/ma-user/requirements.txt


RUN pip3 install --trusted-host cmc.centralrepo.rnd.huawei.com -i https://cmc.centralrepo.rnd.huawei.com/pypi/simple --no-cache-dir -r /home/ma-user/requirements.txt && rm -rf /home/ma-user/requirements.txt