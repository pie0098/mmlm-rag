# mmlm-rag

This project uses Milvus as the vector database and Attu as the visualization and management tool.

## Install Dependencies
> Note: Although `colpali-engine` and `transformers` may have conflicts, they are used in different Python files in this project, so they do not affect each other.
```bash
pip install colpali-engine
pip install transformers==4.53.1
```

## Deploy Milvus Standalone

1. Download the Milvus Standalone Docker Compose file:

   ```bash
   wget https://github.com/milvus-io/milvus/releases/download/v2.6.0-rc1/milvus-standalone-docker-compose.yml -O docker-compose.yml
   ```

2. Start Milvus services:

   ```bash
   sudo docker compose up -d
   ```

   This will automatically create and start the following services:

   - milvus-etcd
   - milvus-minio
   - milvus-standalone

## Deploy Attu (Milvus Dashboard)

1. Get your machine's IP address (for configuring `MILVUS_URL`):

   ```bash
   ifconfig -a
   ```

   Find the IP address associated with the network interface starting with "enp", for example: 192.168.1.100

2. Start Attu:

   ```bash
   docker run -itd --restart=always \
     -p 8000:3000 \
     -e HOST_URL=http://localhost:8000 \
     -e MILVUS_URL=YOUR_MACHINE_IP:19530 \
     zilliz/attu:latest
   ```

   Example:

   ```bash
   -e MILVUS_URL=192.168.1.100:19530
   ```

3. After starting, open your browser and visit [http://localhost:8000](http://localhost:8000) to access the Attu dashboard.

---

For more information, please refer to the official documentation or contact the project maintainer.