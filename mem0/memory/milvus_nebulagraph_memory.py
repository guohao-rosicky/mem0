import logging

from mem0.memory.utils import format_entities

try:
    from nebula3.gclient.net import ConnectionPool
    from nebula3.Config import Config
except ImportError:
    raise ImportError("nebula3-python is not installed. Please install it using pip install nebula3-python")

try:
    from pymilvus import (
        FieldSchema,
        CollectionSchema,
        DataType,
        Collection,
        utility,
        MilvusClient
    )
except ImportError:
    raise ImportError("pymilvus is not installed. Please install it using pip install pymilvus")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory

import uuid
import time

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MilvusNebulaGraph:
    def __init__(self, config):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config
        )
        self._vector_graph_db = VectorGraphDB(self.config.graph_store.config, self.embedding_model)

        self.llm_provider = "openai_structured"
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider

        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)
        self.user_id = None
        self.threshold = 0.7

    def getVectorGraphDB(self):
        return self._vector_graph_db

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)

        # TODO: Batch queries with APOC plugin
        # TODO: Add more filter support
        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)

        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "destination": item[2]})

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    def delete_all(self, filters):
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", "")
        project_id = filters.get("project_id", "")
        member_id = filters.get("member_id", "")
        self._vector_graph_db.delete_all(user_id=user_id, agent_id=agent_id, project_id=project_id, member_id=member_id)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.
         Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'contexts': The base data store response for each memory.
                - 'entities': A list of strings representing the nodes and relationships
        """

        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", "")
        project_id = filters.get("project_id", "")
        member_id = filters.get("member_id", "")
        results = self._vector_graph_db.graph_search_all(project_id, member_id, user_id, agent_id, limit)
        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Establish relations among the extracted nodes."""

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            # Add the custom prompt line if configured
            system_content = system_content.replace("CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}")
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities.get("tool_calls"):
            entities = extracted_entities["tool_calls"][0].get("arguments", {}).get("entities", [])

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes among and their respective incoming and outgoing relations."""
        result_relations = []

        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", "")
        project_id = filters.get("project_id", "")
        member_id = filters.get("member_id", "")

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)

            # 先查向量相似度
            vector_result = self._vector_graph_db.vector_search(n_embedding, project_id, member_id, user_id, agent_id,
                                                                limit, self.threshold)
            vector_result = vector_result[::-1]

            # 先从一端找图上的点和关系
            graph_result1 = self._vector_graph_db.graph_search_forward(vector_result)

            # 再找另一端找图上的点和关系
            graph_result2 = self._vector_graph_db.graph_search_reverse(vector_result)

            # 合并 graph1 + graph2
            graph_result = []
            graph_result.extend(graph_result1)
            graph_result.extend(graph_result2)

            m = {}  # distinct
            ans = []
            for e in graph_result:
                if e["source"] in m:
                    continue
                m[e["source"]] = e
                ans.append(e)
                if len(ans) >= limit:
                    break

            result_relations.extend(ans)

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, user_identity)

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        # Clean entities formatting
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        user_id = filters["user_id"]
        project_id = filters.get("project_id", "")
        member_id = filters.get("member_id", "")
        agent_id = filters.get("agent_id", "")
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            ref_list = self._vector_graph_db.graph_search_ref(source, relationship, destination, project_id, member_id,
                                                              user_id, agent_id)
            self._vector_graph_db.delete_edges(ref_list)
            results.append(ref_list)

        return results

    # to_be_added: [{'source': 'user_id:_guohao,_agent_id:_agent01', 'relationship': 'is_a', 'destination': 'vegetarian'},
    # {'source': 'user_id:_guohao,_agent_id:_agent01', 'relationship': 'has_allergy', 'destination': 'nut_allergy'}]
    # entity_type_map: {'guohao': 'person', 'vegetarian': 'diet', 'nut_allergy': 'allergy'}
    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", "")
        project_id = filters.get("project_id", "")
        member_id = filters.get("member_id", "")

        results = []

        itmes = self._vector_graph_db.entry_type_to_vertex_and_edge(to_be_added=to_be_added, project_id=project_id,
                                                                    member_id=member_id, user_id=user_id,
                                                                    agent_id=agent_id)

        for item in itmes:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # types
            source_type = entity_type_map.get(source, "__User__")

            destination_type = entity_type_map.get(destination, "__User__")

            # embeddings
            source.embedding = source_embedding = self.embedding_model.embed(source)
            destination.embedding = dest_embedding = self.embedding_model.embed(destination)

            # search for the nodes with the closest embeddings
            threshold = 0.9
            source_node_search_result = self._vector_graph_db.vector_search(source_embedding, project_id, member_id,
                                                                            user_id, agent_id, 1, threshold)
            destination_node_search_result = self._vector_graph_db.vector_search(dest_embedding, project_id, member_id,
                                                                                 user_id, agent_id, 1, threshold)

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                # 1. 添加源vertex，覆盖它(entity_id 相同，提及次数加一)
                self._vector_graph_db.add_vertex_mentions(source)
                # 2. 添加目的vertex
                self._vector_graph_db.add_vertex(destination)
                # 3. 添加关系
                self._vector_graph_db.add_edge(source, destination, relationship)

            elif destination_node_search_result and not source_node_search_result:
                # 1. 添加源vertex
                self._vector_graph_db.add_vertex(source)
                # 2. 添加目的vertex，覆盖它(entity_id 相同，提及次数加一)
                self._vector_graph_db.add_vertex_mentions(destination)
                # 3. 添加关系
                self._vector_graph_db.add_edge(source, destination, relationship)

            elif source_node_search_result and destination_node_search_result:
                # 1. 添加源vertex，覆盖它(entity_id 相同，提及次数加一)
                self._vector_graph_db.add_vertex_mentions(source)
                # 2. 添加目的vertex，覆盖它(entity_id 相同，提及次数加一)
                self._vector_graph_db.add_vertex_mentions(destination)
                # 3. 添加关系
                self._vector_graph_db.add_edge(source, destination, relationship)

            else:  # 都没有找到历史记录的情况下，直接添加记录即可
                # 1. 添加源vertex
                self._vector_graph_db.add_vertex(source)
                # 2. 添加目的vertex
                self._vector_graph_db.add_vertex(destination)
                # 3. 添加关系
                self._vector_graph_db.add_edge(source, destination, relationship)

            result = self._vector_graph_db.vector_search(query_vector=source, project_id=project_id, member_id=member_id,
                                                    user_id=user_id, agent_id=agent_id)
            results.append(result)
        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    # Reset is not defined in base.py
    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        return self._vector_graph_db.delete_all()


# 定义两种类型，点和边（节点和关系）
class VertexModel(BaseModel):
    entity_id: Optional[str] = None
    name: str
    project_id: str
    member_id: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    embedding: Optional[list] = None
    mentions: Optional[int] = None
    created: Optional[int] = None


class EdgeModel(BaseModel):
    name: str
    mentions: Optional[int] = None
    created: Optional[int] = None
    updated_at: Optional[int] = None


class VectorGraphDB:
    """
    集成 Nebula Graph（图数据库）与 Milvus（向量数据库）的协同系统
    功能：
    1. 在 Nebula 中存储实体关系图
    2. 在 Milvus 中存储实体的向量特征
    3. 支持基于向量的相似性搜索与图关系查询联动
    """

    def __init__(self, config, embedding_model):
        """
        初始化两个数据库的连接
        """
        # Nebula Graph 客户端配置
        nebula_pool = ConnectionPool()
        cfg = Config()
        cfg.max_connection_pool_size = config.nebula_max_connection_pool_size
        success = nebula_pool.init([(config.nebula_address, config.nebula_port)], cfg)
        if not success:
            raise ConnectionError("Failed to initialize Nebula connection pool")

        self._nebula_session = nebula_pool.get_session(config.nebula_user, config.nebula_password)
        self._nebula_pool = nebula_pool
        self._nebula_space = config.nebula_space
        self._nebula_partition_num = config.nebula_partition_num
        self._nebula_replica_factor = config.nebula_replica_factor
        self._create_nebula_graph_space_if_need()

        # Milvus 客户端配置，todo：创建索引
        self._milvus_client = MilvusClient(uri=config.milvus_uri)
        self._collection_name = config.milvus_collection_name
        self._vector_dimension = config.milvus_vector_dimension
        self._create_milvus_collection_if_need()

        self.threshold = 0.7

        self.embedding_model = embedding_model

    def _create_milvus_collection_if_need(self):
        """确保Milvus集合存在，若不存在则创建"""
        if not self._milvus_client.has_collection(self._collection_name):
            # 定义字段
            fields = [
                # 主键字段，自增ID
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                # 用于标识向量对应的实体（图上的点的id）
                FieldSchema(name="entity_id", dtype=DataType.VARCHAR, max_length=50),
                # 其它的id
                FieldSchema(name="project_id", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="member_id", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="agent_id", dtype=DataType.VARCHAR, max_length=50),
                # 向量字段
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._vector_dimension),
                # Json字段，保留用作其它使用
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]

            # 创建集合
            schema = CollectionSchema(fields, enable_dynamic_field=True,
                                      description="配合图数据库存储的向量索引")

            index = self._milvus_client.prepare_index_params(
                field_name="embedding", metric_type="COSINE", index_type="AUTOINDEX", index_name="vector_index"
            )

            self._milvus_client.create_collection(collection_name=self._collection_name, schema=schema,
                                                  index_params=index)
            logger.info(f"create collection {self._collection_name} successful.")

    def _create_nebula_graph_space_if_need(self):
        """确保Nebula graph space存在，若不存在则创建，首次创建space时间需要比较长，创建space、tag、edge type 都需要等待心跳"""
        result = self._nebula_session.execute(f"DESCRIBE SPACE {self._nebula_space}")
        if not result.is_succeeded():
            # Create the space
            r = self._nebula_session.execute_py(
                f"CREATE SPACE IF NOT EXISTS {self._nebula_space}(partition_num={self._nebula_partition_num}, replica_factor={self._nebula_replica_factor}, vid_type=FIXED_STRING(30));")
            if r.is_succeeded():
                logger.info(f"create nebula graph space {self._nebula_space} successful")
                time.sleep(
                    20)  # two cycles of heartbeat, by default of a NebulaGraph cluster, we will need to sleep 20s
                # Create the schemas
                self._nebula_session.execute_py(
                    f"USE {self._nebula_space};" +
                    "CREATE TAG IF NOT EXISTS vp(entity_id string, name string, vtype string, user_id string, agent_id string, project_id string, member_id string, mentions int, created TIMESTAMP);" +
                    "CREATE EDGE IF NOT EXISTS rp(name string, mentions int, created TIMESTAMP, updated_at TIMESTAMP);" +

                    "CREATE TAG INDEX `vp_index_0` on `vp`();"
                    "CREATE TAG INDEX `entity_id_index` on `vp`(`entity_id`(256)) ;" +
                    "CREATE TAG INDEX `name_index` on `vp`(`name`(256)) ;" +
                    "CREATE TAG INDEX `vtype_index` on `vp`(`vtype`(256)) ;" +
                    "CREATE TAG INDEX `user_id_index` on `vp`(`user_id`(256)) ;" +
                    "CREATE TAG INDEX `agent_id_index` on `vp`(`agent_id`(256)) ;" +
                    "CREATE TAG INDEX `member_id_index` on `vp`(`member_id`(256)) ;" +
                    "CREATE TAG INDEX `project_id_index` on `vp`(`project_id`(256)) ;"
                )
                # insert data need to sleep after create schema
                time.sleep(20)

                logger.info("create nebula graph tag&edge successful")
        self._nebula_session.execute_py(
            f"USE {self._nebula_space};"
        )

    # 转换为对象
    def entry_type_to_vertex_and_edge(self, to_be_added: list, project_id: str, member_id: str, user_id: str,
                                      agent_id: str) -> list:
        timestamp = int(time.time())
        result = []
        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            source_vertex = VertexModel(
                entity_id=self._get_uniq_id(),
                name=source,
                project_id=project_id,
                member_id=member_id,
                user_id=user_id,
                agent_id=agent_id,
                mentions=1,
                created=timestamp,
            )
            destination_vertex = VertexModel(
                entity_id=self._get_uniq_id(),
                name=destination,
                project_id=project_id,
                member_id=member_id,
                user_id=user_id,
                agent_id=agent_id,
                mentions=1,
                created=timestamp,
            )
            edge = EdgeModel(
                name=relationship,
                mentions=1,
                created=timestamp,
                updated_at=timestamp,
            )

            result.append({
                "source": source_vertex,
                "relationship": edge,
                "destination": destination_vertex,
            })

        return result

    def add_vertex(self, vertex: VertexModel):
        # 添加到graph
        args = vertex.model_dump()
        self._nebula_session.execute_py(
            f"INSERT VERTEX vp(entity_id, name, user_id, agent_id, project_id, member_id, mentions, created) VALUES '{args["entity_id"]}':($entity_id, $name, $user_id, $agent_id, $project_id, $member_id, $mentions, $created)",
            args
        )

        # 添加到vector
        items = [{
            "entity_id": args["entity_id"],
            "project_id": args["project_id"],
            "member_id": args["member_id"],
            "user_id": args["user_id"],
            "agent_id": args["agent_id"],
            "embedding": args["embedding"],
            "name": args["name"],
            "created": args["created"],
            "metadata": {
                "name": args["name"],
            }
        }]
        self._milvus_client.insert(collection_name=self._collection_name, data=items)

    def add_vertex_mentions(self, vertex: VertexModel):
        self._nebula_session.execute_py(
            f"UPDATE VERTEX ON vp \"{vertex.entity_id}\" SET mentions = mentions + 1 YIELD mentions"
        )

    def delete_vertex(self, entity_id: str):
        self._nebula_session.execute_py(f"DELETE VERTEX \"{entity_id}\"")
        self.delete_vector_from_entity_id([entity_id])

    def add_edge(self, source: VertexModel, edge: EdgeModel, destination: VertexModel):
        args = edge.model_dump()
        rank = int(time.time() * 1000)
        query = f"INSERT EDGE rp(name, mentions, created, updated_at) VALUES \"{source.entity_id}\"->\"{destination.entity_id}\"@{rank}:($name, $mentions, $created, $updated_at)"
        r = self._nebula_session.execute_py(query, args)
        logger.debug(f"add edge {edge}, query: {query}, result: {r}")

    def delete_edge(self, source_id: str, dest_id: str, rank: int):
        query = f"DELETE EDGE rp \"{source_id}\" -> \"{dest_id}\"@{rank}"
        delete_edge = self._nebula_session.execute_py(query)
        logger.debug(f"delete edge query: {query}, result: {delete_edge}")

    def delete_edges(self, ref_list: list):
        for rr in ref_list:
            self.delete_edge(rr["__source_entity_id__"], rr["__destination_entity_id__"], rr["__relationship_rank__"])

    def delete_all(self, project_id: str, member_id: str, user_id: str, agent_id: str) -> list:
        res = self.graph_search_all(project_id, member_id, user_id, agent_id)
        if res:
            for re in res:
                self.delete_vertex(re["__source_entity_id__"])
                self.delete_vertex(re["__destination_entity_id__"])
            self.delete_edges(res)
        return res

    def _get_uniq_id(self) -> str:
        s = str(uuid.uuid4()).replace("-", "")
        reversed_s = ''.join(reversed(s))
        return reversed_s[:16]

    def search_vector_by_ids(self, ids: list) -> dict:
        f = "entity_id in " + str(ids)
        res = self._milvus_client.query(
            collection_name=self._collection_name,
            filter=f,
            output_fields=["name", "entity_id"]
        )

        m = {}

        for re in res:
            m[re["entity_id"]] = re["name"]

        return m

    # 按照向量相似度降序排列
    def search_vector_by_ids_ordered(self, ids: list, query_vector: list) -> List[dict]:
        f = "entity_id in " + str(ids)
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "radius": 0,
                "range_filter": 1
            }
        }
        results = self._milvus_client.search(
            collection_name=self._collection_name,
            data=[query_vector],
            filter=f,
            anns_field="embedding",
            search_params=search_params,
            output_fields=["name", "entity_id"]
        )
        res = []
        for result in results:
            for e in result:
                res.append({
                    "name": e["name"],
                    "entity_id": e["entity_id"],
                    "distance": e["distance"],
                })
        return res[::-1]

    def vector_search(self, query_vector: list, project_id: str, member_id: str, user_id: str, agent_id: str,
                      top_k: int = 3, threshold=0.9) -> list:
        """
        执行向量搜索并返回关联的图数据
        param query_vector: 查询向量
        """
        # 1. Milvus向量搜索
        filter_list = []
        if project_id:
            filter_list.append(f" project_id == '{project_id}' ")

        if member_id:
            filter_list.append(f" member_id == '{member_id}' ")

        if user_id:
            filter_list.append(f" user_id == '{user_id}' ")

        if agent_id:
            filter_list.append(f" agent_id == '{agent_id}' ")

        filter = "" if len(filter_list) == 0 else " and ".join(filter_list)

        logger.debug(f"filter: {filter} filter_list: ({len(filter_list)}) {filter_list}")

        search_params = {
            "metric_type": "COSINE",
            "params": {
                "radius": threshold,
                "range_filter": 1
            }
        }

        search_results = self._milvus_client.search(
            collection_name=self._collection_name,
            data=[query_vector],
            filter=filter,
            anns_field="embedding",
            search_params=search_params,
            limit=top_k,
            output_fields=["name", "entity_id"]
        )

        vids = []
        for hits in search_results:
            for hit in hits:
                vids.append(hit["entity_id"])
        logger.debug("search vector results: %s, vids: %s", search_results, vids)
        return vids

    def graph_search_all(self, project_id: str, member_id: str, user_id: str, agent_id: str, limit: int = 0) -> list:
        project_id_param = f"project_id: '{project_id}'," if project_id else ""
        member_id_param = f"member_id: '{member_id}'," if member_id else ""
        user_id_param = f"user_id: '{user_id}'," if user_id else ""
        agent_id_param = f"agent_id: '{agent_id}'," if agent_id else ""

        source_param = project_id_param + member_id_param + user_id_param + agent_id_param
        source_param = source_param.strip(",")
        dest_param = project_id_param + member_id_param + user_id_param + agent_id_param
        dest_param = dest_param.strip(",")

        limit_param = "" if limit == 0 else f" LIMIT '{limit}'"

        query = f"""
                MATCH (src:vp {{ {source_param} }})
                -[e:rp]->
                (dst:vp {{ {dest_param} }})
                RETURN
                    src AS source,
                    dst AS destination,
                    e AS rel
                {limit_param}
                """

        graph_data = self._nebula_session.execute_py(query)
        results = []
        if graph_data and graph_data.is_succeeded():
            rp = graph_data.as_primitive()
            for r in rp:
                results.append({
                    "source": r["source"]["tags"]["vp"]["name"],
                    "destination": r["destination"]["tags"]["vp"]["name"],
                    "relationship": r["rel"]["props"]["name"],
                    "__source_entity_id__": r["source"]["tags"]["vp"]["entity_id"],
                    "__destination_entity_id__": r["destination"]["tags"]["vp"]["entity_id"],
                    "__relationship_rank__": r["rel"]["rank"],
                })

        logger.debug(f"results: {results}, query: {query}")

        return results

    def graph_search_ref(self, source_name: str, relationship: str, dest_name: str, project_id: str, member_id: str,
                         user_id: str, agent_id: str) -> list:

        project_id_param = f"project_id: '{project_id}'," if project_id else ""
        member_id_param = f"member_id: '{member_id}'," if member_id else ""
        user_id_param = f"user_id: '{user_id}'," if user_id else ""
        agent_id_param = f"agent_id: '{agent_id}'," if agent_id else ""

        source_param = f"name: '{source_name}'," + project_id_param + member_id_param + user_id_param + agent_id_param
        source_param = source_param.strip(",")
        dest_param = f"name: '{dest_name}'," + project_id_param + member_id_param + user_id_param + agent_id_param
        dest_param = dest_param.strip(",")

        query = f"""
        MATCH (src:vp {{ {source_param} }})
        -[e:rp {{ name: '{relationship}' }}]->
        (dst:vp {{ {dest_param} }})
        RETURN
            src AS source,
            dst AS destination,
            e AS rel
        """

        graph_data = self._nebula_session.execute_py(query)
        results = []
        if graph_data and graph_data.is_succeeded():
            rp = graph_data.as_primitive()
            for r in rp:
                results.append({
                    "source": r["source"]["tags"]["vp"]["name"],
                    "destination": r["destination"]["tags"]["vp"]["name"],
                    "relationship": r["rel"]["props"]["name"],
                    "__source_entity_id__": r["source"]["tags"]["vp"]["entity_id"],
                    "__destination_entity_id__": r["destination"]["tags"]["vp"]["entity_id"],
                    "__relationship_rank__": r["rel"]["rank"],
                })

        logger.debug(f"results: {results}, query: {query}")

        return results

    def delete_vector_from_entity_id(self, entity_id: list):
        f = "entity_id in " + str(entity_id)
        self._milvus_client.delete(collection_name=self._collection_name,
                                   filter=f)

    def graph_search_forward(self, vid_list: list) -> list:
        return self._graph_search(vid_list, "forward")

    def graph_search_reverse(self, vid_list: list) -> list:
        return self._graph_search(vid_list, "reverse")

    def _graph_search(self, vid_list: list, t: str) -> list:
        params = {
            "vids": vid_list,
        }
        query = f"""
               MATCH (src:vp)-[e:rp]->(dst:vp)
               WHERE id(src) in $vids
               RETURN src as source, e AS rel, dst as destination
               """

        if t == "reverse":
            query = f"""
                   MATCH (src:vp)-[e:rp]->(dst:vp)
                   WHERE id(dst) in $vids
                   RETURN src as source, e AS rel, dst as destination
                   """

        graph_data = self._nebula_session.execute_py(query, params)
        results = []
        if graph_data and graph_data.is_succeeded():
            rp = graph_data.as_primitive()
            for r in rp:
                results.append({
                    "source": r["source"]["tags"]["vp"]["name"],
                    "destination": r["destination"]["tags"]["vp"]["name"],
                    "relationship": r["rel"]["props"]["name"],
                    "__source_entity_id__": r["source"]["tags"]["vp"]["entity_id"],
                    "__destination_entity_id__": r["destination"]["tags"]["vp"]["entity_id"],
                    "__relationship_rank__": r["rel"]["rank"],
                })

        logger.debug(f"search graph params: {params}, origin data: {graph_data}, results: {results}")

        return results

    # 返回指定类型为key的dict
    def get_graph_map(self, results: list, t: str) -> dict:
        m = {}
        if t == "source":
            for result in results:
                m[result["__source_entity_id__"]] = result
        elif t == "destination":
            for result in results:
                m[result["__destination_entity_id__"]] = result

        return m

    def get_graph_entity_ids(self, results: list) -> list:
        m = []
        for result in results:
            m.append(result["__source_entity_id__"])
            m.append(result["__destination_entity_id__"])

        mm = []
        mm.append(set(m))
        return mm

    def flushDB(self):
        # todo: reset

        pass

    def release(self):
        """清理资源"""
        self._nebula_pool.close()
        self._milvus_client.close()

    def __del__(self):
        self.release()
