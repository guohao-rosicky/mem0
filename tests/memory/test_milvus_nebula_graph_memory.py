import unittest

from mem0.configs.base import MemoryConfig

from mem0.memory.milvus_nebulagraph_memory import MilvusNebulaGraph as MemoryGraph
from mem0.memory.milvus_nebulagraph_memory import VertexModel, EdgeModel

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")


import uuid
import time

import logging

import os
import numpy as np

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TestMilvusNebulaGraphMemory(unittest.TestCase):

    def getDB(self):
        LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
        LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-8b")
        LLM_API_KEY = os.environ.get("LLM_API_KEY", "yndVGzNS07o39SvG99F3E486E6Bf48Ac93Eb3407D45e8294")
        LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://llm.api.zyuncs.com/v1")
        LLM_MAX_TOKENS = os.environ.get("LLM_MAX_TOKENS", "2000")

        EMBEDDER_PROVIDER = os.environ.get("EMBEDDER_PROVIDER", "openai")
        EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "360-qwen3-embedding-8b")
        EMBEDDER_API_KEY = os.environ.get("EMBEDDER_API_KEY", "yndVGzNS07o39SvG99F3E486E6Bf48Ac93Eb3407D45e8294")
        EMBEDDER_BASE_URL = os.environ.get("EMBEDDER_BASE_URL", "http://llm.api.zyuncs.com/v1")

        # 配置参数
        dim = 2048
        self.dim = dim
        space_name = "test_vector_graph10"

        nebula_cfg = {
            "nebula_address": "dev01v.xstore.corp.qihoo.net",
            "nebula_port": 9669,
            "nebula_max_connection_pool_size": 10,
            "nebula_user": "root",
            "nebula_password": "nebula",
            "nebula_space": space_name,
            "nebula_partition_num": 1,
            "nebula_replica_factor": 1
        }
        milvus_cfg = {
            "milvus_uri": "http://dev01v.xstore.corp.qihoo.net:19530",
            "milvus_collection_name": space_name,
            "milvus_vector_dimension": dim
        }

        graph_config = {**nebula_cfg, **milvus_cfg}

        SERVER_CONFIG = {
            "version": "v1.1",
            "vector_store": {
                "provider": "milvus",
                "config": {
                    "collection_name": "test_mem0_3",
                    "embedding_model_dims": dim,
                    "url": "http://dev01v.xstore.corp.qihoo.net:19530",
                    "token": "",
                    "metric_type": "L2",
                }
            },
            "graph_store": {
                "provider": "milvus_nebula_graph",
                "config": graph_config,
            },
            "llm": {
                "provider": LLM_PROVIDER,
                "config": {
                    "api_key": LLM_API_KEY,
                    "temperature": 0.2,
                    "model": LLM_MODEL,
                    "max_tokens": int(LLM_MAX_TOKENS)
                }
            },
            "embedder": {
                "provider": EMBEDDER_PROVIDER,
                "config": {
                    "api_key": EMBEDDER_API_KEY,
                    "model": EMBEDDER_MODEL
                }
            },
            "history_db_path": "/tmp/mem0-test-server/history.db",
        }

        cc = MemoryConfig(**SERVER_CONFIG)

        logger.info(f"parse config {cc.graph_store.config}")

        return MemoryGraph(cc)


    def testA(self):

        logger.info("....start")

        memoryGraph = self.getDB()

        vgdb = memoryGraph.getVectorGraphDB()

        logger.info(f"my_test")

        logger.info("step 1")

        timestamp = time.time()

        ss = str(uuid.uuid4()).replace("-", "")[:14]

        source_v = None
        dest_v = None

        for i in range(2):
            # 添加示例数据
            mock_vectors = np.random.rand(self.dim).tolist()  # 随机向量

            entity_id = str(uuid.uuid4()).replace("-", "")[:16]

            v1 = VertexModel(
                entity_id=entity_id,
                name=f"{ss}_{i}",
                project_id="project_01",
                member_id="user_04",
                user_id="guohao1",
                agent_id="myagent",
                mentions=1,
                created=int(timestamp),
                embedding=mock_vectors
            )

            vgdb.add_vertex(v1)

            if i == 0:
                source_v = v1
                logger.info(f"add source vertex {v1} successful")
            else:
                dest_v = v1
                logger.info(f"add dest vertex {v1} successful")

        e1 = EdgeModel(
            name=f"like",
            mentions=1,
            created=int(timestamp),
            updated_at=int(timestamp),
        )

        vgdb.add_edge(source_v, e1, dest_v)
        logger.info(f"add edge {e1} successful")

        logger.info("step 2")

        mock_vectors = np.random.rand(self.dim).tolist()  # 随机向量

        project_id = "project_01"
        member_id = "user_04"
        user_id = ""
        agent_id = ""
        threshold = 0.1

        vid_list = vgdb.vector_search(mock_vectors, project_id=project_id, member_id=member_id, user_id=user_id,
                                      agent_id=agent_id, threshold=threshold)

        logger.debug("forward...........")
        r1 = vgdb.graph_search_forward(vid_list)
        logger.debug("reverse...........")
        r2 = vgdb.graph_search_reverse(vid_list)

        if r1:
            for e in r1:
                logger.debug(f"forward rel: {e}")
                rrr = vgdb.graph_search_ref(source_name=e["source"], relationship=e["relationship"],
                                            dest_name=e["destination"],
                                            project_id=project_id, member_id=member_id, user_id=user_id, agent_id=agent_id)

                for rr in rrr:
                    vgdb.delete_edge(rr["__source_entity_id__"], rr["__destination_entity_id__"],
                                     rr["__relationship_rank__"])

        if r2:
            for e in r2:
                logger.debug(f"reverse rel: {e}")
                rrr = vgdb.graph_search_ref(source_name=e["source"], relationship=e["relationship"],
                                            dest_name=e["destination"],
                                            project_id=project_id, member_id=member_id, user_id=user_id, agent_id=agent_id)

        vgdb.release()
