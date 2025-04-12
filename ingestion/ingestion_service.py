import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import json
import asyncio
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
import aiofiles
import numpy as np
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class DataFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    BINARY = "binary"
    TEXT = "text"

class DataSource(str, Enum):
    API = "api"
    FILE = "file"
    STREAM = "stream"
    DATABASE = "database"
    SENSOR = "sensor"

class DataSchema(BaseModel):
    name: str
    fields: Dict[str, str]  # field_name: field_type
    constraints: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class DataValidationRule(BaseModel):
    field: str
    rule_type: str  # e.g., "range", "regex", "enum", "custom"
    parameters: Dict[str, Any]
    error_message: str

class DataTransformation(BaseModel):
    field: str
    transformation_type: str  # e.g., "normalize", "encode", "aggregate"
    parameters: Optional[Dict[str, Any]] = None

class IngestionConfig(BaseModel):
    source: DataSource
    format: DataFormat
    schema: DataSchema
    validation_rules: Optional[List[DataValidationRule]] = None
    transformations: Optional[List[DataTransformation]] = None
    routing: Optional[Dict[str, str]] = None  # topic/queue mapping
    batch_size: Optional[int] = 1000
    timeout: Optional[float] = 30.0

class DataValidator:
    def __init__(self, rules: List[DataValidationRule]):
        self.rules = rules
        
    def validate(self, data: Dict[str, Any]) -> List[str]:
        """Validate data against rules and return error messages."""
        errors = []
        
        for rule in self.rules:
            if rule.field not in data:
                errors.append(f"Missing field: {rule.field}")
                continue
                
            value = data[rule.field]
            
            if rule.rule_type == "range":
                min_val = rule.parameters.get("min")
                max_val = rule.parameters.get("max")
                if min_val is not None and value < min_val:
                    errors.append(rule.error_message)
                if max_val is not None and value > max_val:
                    errors.append(rule.error_message)
                    
            elif rule.rule_type == "regex":
                import re
                pattern = rule.parameters["pattern"]
                if not re.match(pattern, str(value)):
                    errors.append(rule.error_message)
                    
            elif rule.rule_type == "enum":
                allowed_values = rule.parameters["values"]
                if value not in allowed_values:
                    errors.append(rule.error_message)
                    
            elif rule.rule_type == "custom":
                # Execute custom validation function
                func_name = rule.parameters["function"]
                try:
                    if not globals()[func_name](value):
                        errors.append(rule.error_message)
                except Exception as e:
                    errors.append(f"Custom validation error: {str(e)}")
                    
        return errors

class DataTransformer:
    def __init__(self, transformations: List[DataTransformation]):
        self.transformations = transformations
        
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformations to data."""
        result = data.copy()
        
        for transform in self.transformations:
            if transform.field not in result:
                continue
                
            value = result[transform.field]
            
            if transform.transformation_type == "normalize":
                min_val = transform.parameters.get("min", 0)
                max_val = transform.parameters.get("max", 1)
                value = (value - min_val) / (max_val - min_val)
                
            elif transform.transformation_type == "encode":
                encoding = transform.parameters.get("encoding", "one_hot")
                if encoding == "one_hot":
                    # Implement one-hot encoding
                    categories = transform.parameters["categories"]
                    encoded = {f"{transform.field}_{cat}": 1 if value == cat else 0
                             for cat in categories}
                    result.update(encoded)
                    del result[transform.field]
                    
            elif transform.transformation_type == "aggregate":
                window = transform.parameters.get("window", 10)
                operation = transform.parameters.get("operation", "mean")
                if isinstance(value, list):
                    if operation == "mean":
                        value = np.mean(value[-window:])
                    elif operation == "sum":
                        value = np.sum(value[-window:])
                        
            result[transform.field] = value
            
        return result

class DataRouter:
    def __init__(self, kafka_bootstrap_servers: str):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
    async def route(self, data: Dict[str, Any], routing_config: Dict[str, str]):
        """Route data to appropriate topics/queues."""
        futures = []
        
        for condition, topic in routing_config.items():
            # Evaluate routing condition
            try:
                if eval(condition, {"data": data}):
                    future = self.producer.send(topic, data)
                    futures.append(future)
            except Exception as e:
                logger.error(f"Error evaluating routing condition: {str(e)}")
                
        # Wait for all messages to be sent
        for future in futures:
            try:
                future.get(timeout=10)
            except KafkaError as e:
                logger.error(f"Error sending message to Kafka: {str(e)}")

class IngestionManager:
    def __init__(self):
        self.configs: Dict[str, IngestionConfig] = {}
        self.validators: Dict[str, DataValidator] = {}
        self.transformers: Dict[str, DataTransformer] = {}
        self.router = DataRouter(os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
        
    def register_config(self, config_id: str, config: IngestionConfig):
        """Register an ingestion configuration."""
        self.configs[config_id] = config
        
        if config.validation_rules:
            self.validators[config_id] = DataValidator(config.validation_rules)
            
        if config.transformations:
            self.transformers[config_id] = DataTransformer(config.transformations)
            
    async def process_data(
        self,
        config_id: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """Process incoming data according to configuration."""
        if config_id not in self.configs:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration {config_id} not found"
            )
            
        config = self.configs[config_id]
        
        # Handle both single items and batches
        items = data if isinstance(data, list) else [data]
        results = []
        errors = []
        
        for item in items:
            try:
                # Validate
                if config_id in self.validators:
                    validation_errors = self.validators[config_id].validate(item)
                    if validation_errors:
                        errors.append({
                            "data": item,
                            "errors": validation_errors
                        })
                        continue
                        
                # Transform
                if config_id in self.transformers:
                    item = self.transformers[config_id].transform(item)
                    
                # Route
                if config.routing:
                    background_tasks.add_task(
                        self.router.route,
                        item,
                        config.routing
                    )
                    
                results.append(item)
                
            except Exception as e:
                errors.append({
                    "data": item,
                    "error": str(e)
                })
                
        return {
            "processed": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors
        }
        
    async def process_file(
        self,
        config_id: str,
        file: UploadFile,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """Process data from uploaded file."""
        if config_id not in self.configs:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration {config_id} not found"
            )
            
        config = self.configs[config_id]
        
        # Read and parse file based on format
        try:
            if config.format == DataFormat.CSV:
                df = pd.read_csv(file.file)
                data = df.to_dict('records')
            elif config.format == DataFormat.JSON:
                content = await file.read()
                data = json.loads(content)
            elif config.format == DataFormat.PARQUET:
                df = pd.read_parquet(file.file)
                data = df.to_dict('records')
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {config.format}"
                )
                
            return await self.process_data(config_id, data, background_tasks)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file: {str(e)}"
            )

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing ingestion service...")
    try:
        ingestion_manager = IngestionManager()
        app.state.ingestion_manager = ingestion_manager
        logger.info("Ingestion service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ingestion service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down ingestion service...")

app = FastAPI(title="HMAS Ingestion Service", lifespan=lifespan)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/config/{config_id}")
@limiter.limit("20/minute")
async def register_config(
    request: Request,
    config_id: str,
    config: IngestionConfig
):
    """Register an ingestion configuration."""
    try:
        request.app.state.ingestion_manager.register_config(config_id, config)
        return {"status": "success", "config_id": config_id}
    except Exception as e:
        logger.error(f"Error registering configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/{config_id}")
@limiter.limit("100/minute")
async def ingest_data(
    request: Request,
    config_id: str,
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    background_tasks: BackgroundTasks
):
    """Ingest data using specified configuration."""
    try:
        result = await request.app.state.ingestion_manager.process_data(
            config_id,
            data,
            background_tasks
        )
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/file/{config_id}")
@limiter.limit("50/minute")
async def ingest_file(
    request: Request,
    config_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks
):
    """Ingest data from file using specified configuration."""
    try:
        result = await request.app.state.ingestion_manager.process_file(
            config_id,
            file,
            background_tasks
        )
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500) 