from pymilvus import DataType
import numpy as np
import concurrent.futures

class MilvusColbertRetriever:
    def __init__(self, milvus_client, collection_name, dim=128):
        self.collection_name = collection_name
        self.client = milvus_client
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)
        self.dim = dim

    def create_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema.add_field(field_name="seq_id", datatype=DataType.INT16)
        schema.add_field(field_name="doc_id", datatype=DataType.INT64)
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

    def create_index(self):
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="vector"
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="HNSW",
            metric_type="IP",
            params={
                "M": 16,
                "efConstruction": 500,
            },
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def create_scalar_index(self):
        self.client.release_collection(collection_name=self.collection_name)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="doc_id",
            index_name="int32_index",
            index_type="INVERTED",
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def search(self, data, topk):
        # data=query
        # "params": {} means no extra parameters are passed to the HNSW index algorithm
        search_params = {"metric_type": "IP", "params": {}}
        # len(results)=22(query.length), each query vector (1*128) has top50 similar doc vector (1*128)
        results = self.client.search(
            self.collection_name,
            data,
            limit=int(10),
            output_fields=["vector", "doc_id", "doc"],
            search_params=search_params,
        )
        docs = set()
        # For each row vector of a query (22 in total), deduplicate the doc_id results of its top50
        for r_id in range(len(results)):
            for r in range(len(results[r_id])):
                docs.add(results[r_id][r]["entity"]["doc"])

        scores = []

        def rerank_single_doc(doc, data, client, collection_name):
            # get doc_id related vectors
            doc_colbert_vecs = client.query(
                collection_name=collection_name,
                filter=f"doc=='{doc}'",
                output_fields=["seq_id", "vector", "doc"],
                limit=400, # return the first 400 row vectors
            )
            # stack these vectors
            doc_vecs = np.vstack(
                [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
            )
            # Perform ColBERT late interaction calculation for a single query-doc_id
            score = np.dot(data, doc_vecs.T).max(1).sum()
            return (score, doc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc, data, self.client, self.collection_name
                ): doc
                for doc in docs
            }
            for future in concurrent.futures.as_completed(futures):
                score, doc = future.result()
                scores.append((score, doc))
        # sort by the score in each tuple in scores
        scores.sort(key=lambda x: x[0], reverse=True)
        # if the total exceeds topk, take the topk; otherwise, return all
        if len(scores) >= topk:
            return scores[:topk]
        else:
            return scores

    def insert(self, data):
        # The input is the embedding of an image/pdf page, 1064*128
        # data["colbert_vecs"] is a list, 1064*128, each row in the list is a 1*128 torch.tensor
        colbert_vecs = [vec for vec in data["colbert_vecs"]]
        # seq_length = 1064
        seq_length = len(colbert_vecs)
        # Generate the same doc_id for 1064 row vectors, docs is the file path
        # repeat data["doc_id"] for seq_length times, [doc_id, ..., doc_id];
        doc_ids = [data["doc_id"] for i in range(seq_length)]
        # seq_ids = [0,1,2,...,1059]
        seq_ids = list(range(seq_length))
        # repeat data["filepath"] seq_length times
        docs = [data["filepath"] for i in range(seq_length)]

        self.client.insert(
            self.collection_name,
            [
                {
                    "vector": colbert_vecs[i],
                    "seq_id": seq_ids[i],
                    "doc_id": doc_ids[i],
                    "doc": docs[i],
                }
                for i in range(seq_length)
            ],
        ) 