# mmlm-rag
pip install colpali-engine
pip install transformers==4.53.1
# install and start milvus-standalone
wget https://github.com/milvus-io/milvus/releases/download/v2.6.0-rc1/milvus-standalone-docker-compose.yml -O docker-compose.yml

sudo docker compose up -d

Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done

# install and start attu
docker run -itd --restart=always -p 8000:3000 -e HOST_URL=http://localhost:8000 -e MILVUS_URL=your machine IP(obtained from ifconfig -a and find "enp" IP address ):19530 zilliz/attu:latest