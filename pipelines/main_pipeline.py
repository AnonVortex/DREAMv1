import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException
import httpx
import uvicorn
from config import settings  # Ensure this config.py is in the same folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PipelineAggregator")

# Endpoints for each module are loaded from environment variables
INGESTION_URL = settings.INGESTION_URL
PERCEPTION_URL = settings.PERCEPTION_URL
INTEGRATION_URL = settings.INTEGRATION_URL
ROUTING_URL = settings.ROUTING_URL
SPECIALIZED_URL = settings.SPECIALIZED_URL
META_URL = settings.META_URL
MEMORY_URL = settings.MEMORY_URL
AGGREGATION_URL = settings.AGGREGATION_URL
FEEDBACK_URL = settings.FEEDBACK_URL
MONITORING_URL = settings.MONITORING_URL
GRAPH_RL_URL = settings.GRAPH_RL_URL
COMM_URL = settings.COMM_URL

app = FastAPI(title="HMAS Pipeline Aggregator", version="1.0")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/run_pipeline")
async def run_pipeline():
    """
    Orchestrates the entire HMAS pipeline by sequentially calling each module's endpoint.
    """
    async with httpx.AsyncClient() as client:
        # 1. Ingestion: Upload a dummy file
        logger.info("Calling Ingestion...")
        ingestion_response = await client.post(INGESTION_URL, files={"file": ("dummy.txt", b"dummy content", "text/plain")})
        if ingestion_response.status_code != 200:
            raise HTTPException(status_code=ingestion_response.status_code, detail="Ingestion failed")
        ingestion_data = ingestion_response.json()
        logger.info(f"Ingestion output: {ingestion_data}")
        file_path = ingestion_data.get("file_path")
        if not file_path:
            raise HTTPException(status_code=500, detail="Ingestion did not return file_path")
        
        # 2. Perception: Process the file
        logger.info("Calling Perception...")
        perception_payload = {"image_path": file_path}
        perception_response = await client.post(PERCEPTION_URL, json=perception_payload)
        if perception_response.status_code != 200:
            raise HTTPException(status_code=perception_response.status_code, detail="Perception failed")
        perception_data = perception_response.json()
        logger.info(f"Perception output: {perception_data}")

        # 3. Integration: Fuse features
        logger.info("Calling Integration...")
        integration_response = await client.post(INTEGRATION_URL, json=perception_data)
        if integration_response.status_code != 200:
            raise HTTPException(status_code=integration_response.status_code, detail="Integration failed")
        integration_data = integration_response.json()
        logger.info(f"Integration output: {integration_data}")

        # 4. Routing: Decide specialized processing
        logger.info("Calling Routing...")
        routing_response = await client.post(ROUTING_URL, json=integration_data)
        if routing_response.status_code != 200:
            raise HTTPException(status_code=routing_response.status_code, detail="Routing failed")
        routing_data = routing_response.json()
        logger.info(f"Routing output: {routing_data}")

        # 5. Specialized: Process based on routing
        logger.info("Calling Specialized...")
        specialized_response = await client.post(SPECIALIZED_URL, json=routing_data)
        if specialized_response.status_code != 200:
            raise HTTPException(status_code=specialized_response.status_code, detail="Specialized failed")
        specialized_data = specialized_response.json()
        logger.info(f"Specialized output: {specialized_data}")

        # 6. Meta: Evaluate specialized output
        logger.info("Calling Meta...")
        meta_payload = {"specialized_output": specialized_data}
        meta_response = await client.post(META_URL, json=meta_payload)
        if meta_response.status_code != 200:
            raise HTTPException(status_code=meta_response.status_code, detail="Meta failed")
        meta_data = meta_response.json()
        logger.info(f"Meta output: {meta_data}")

        # 7. Memory: Archive meta output
        logger.info("Calling Memory...")
        memory_payload = {"meta_output": meta_data}
        memory_response = await client.post(MEMORY_URL, json=memory_payload)
        if memory_response.status_code != 200:
            raise HTTPException(status_code=memory_response.status_code, detail="Memory failed")
        memory_data = memory_response.json()
        logger.info(f"Memory output: {memory_data}")

        # 8. Aggregation: Produce final decision
        logger.info("Calling Aggregation...")
        aggregation_response = await client.post(AGGREGATION_URL, json=memory_data)
        if aggregation_response.status_code != 200:
            raise HTTPException(status_code=aggregation_response.status_code, detail="Aggregation failed")
        aggregation_data = aggregation_response.json()
        logger.info(f"Aggregation output: {aggregation_data}")

        # 9. Feedback: Provide system feedback
        logger.info("Calling Feedback...")
        feedback_payload = {"final_decision": aggregation_data.get("final_decision")}
        feedback_response = await client.post(FEEDBACK_URL, json=feedback_payload)
        if feedback_response.status_code != 200:
            raise HTTPException(status_code=feedback_response.status_code, detail="Feedback failed")
        feedback_data = feedback_response.json()
        logger.info(f"Feedback output: {feedback_data}")

        # 10. Monitoring: Get system diagnostics
        logger.info("Calling Monitoring...")
        monitoring_response = await client.get(MONITORING_URL)
        if monitoring_response.status_code != 200:
            raise HTTPException(status_code=monitoring_response.status_code, detail="Monitoring failed")
        monitoring_data = monitoring_response.json()
        logger.info(f"Monitoring output: {monitoring_data}")

        # 11. Graph RL: (Optional) Trigger Graph RL training/inference
        logger.info("Calling Graph RL...")
        graph_rl_response = await client.post(GRAPH_RL_URL, json={})
        if graph_rl_response.status_code != 200:
            raise HTTPException(status_code=graph_rl_response.status_code, detail="Graph RL failed")
        graph_rl_data = graph_rl_response.json()
        logger.info(f"Graph RL output: {graph_rl_data}")

        # 12. Communication: (Optional) Optimize inter-agent communication
        logger.info("Calling Communication Optimization...")
        comm_response = await client.post(COMM_URL, json={})
        if comm_response.status_code != 200:
            raise HTTPException(status_code=comm_response.status_code, detail="Communication failed")
        comm_data = comm_response.json()
        logger.info(f"Communication output: {comm_data}")

        final_output = {
            "ingestion": ingestion_data,
            "perception": perception_data,
            "integration": integration_data,
            "routing": routing_data,
            "specialized": specialized_data,
            "meta": meta_data,
            "memory": memory_data,
            "aggregation": aggregation_data,
            "feedback": feedback_data,
            "monitoring": monitoring_data,
            "graph_rl": graph_rl_data,
            "communication": comm_data,
        }
        return final_output

if __name__ == "__main__":
    uvicorn.run("main_pipeline:app", host="0.0.0.0", port=9000, reload=True)
