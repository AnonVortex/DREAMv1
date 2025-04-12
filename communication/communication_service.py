import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from datetime import datetime
import json
import asyncio
import aio_pika
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
import redis
from redis.client import Redis

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"
    EVENT = "event"
    BROADCAST = "broadcast"
    DIRECT = "direct"

class MessagePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProtocolType(str, Enum):
    SYNC = "sync"
    ASYNC = "async"
    PUBSUB = "pubsub"
    REQUEST_REPLY = "request_reply"

class Message(BaseModel):
    id: str = Field(default_factory=lambda: datetime.now().isoformat())
    type: MessageType
    sender: str
    receiver: Optional[str] = None
    topic: Optional[str] = None
    content: Any
    priority: MessagePriority = MessagePriority.MEDIUM
    protocol: ProtocolType
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    ttl: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class CommunicationConfig(BaseModel):
    protocols: List[ProtocolType]
    message_types: List[MessageType]
    rate_limits: Dict[MessageType, int]
    retry_policy: Optional[Dict[str, Any]] = None
    timeout: int = 30

class MessageBroker:
    def __init__(self, rabbitmq_url: str):
        self.url = rabbitmq_url
        self.connection = None
        self.channel = None
        self.exchange = None
        self.queues: Dict[str, aio_pika.Queue] = {}
        
    async def connect(self):
        """Connect to RabbitMQ."""
        try:
            self.connection = await aio_pika.connect_robust(self.url)
            self.channel = await self.connection.channel()
            self.exchange = await self.channel.declare_exchange(
                "hmas_exchange",
                aio_pika.ExchangeType.TOPIC
            )
            logger.info("Connected to RabbitMQ")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise
            
    async def close(self):
        """Close connection."""
        if self.connection:
            await self.connection.close()
            
    async def declare_queue(self, name: str) -> aio_pika.Queue:
        """Declare a queue."""
        queue = await self.channel.declare_queue(
            name,
            durable=True,
            auto_delete=False
        )
        self.queues[name] = queue
        return queue
        
    async def publish(
        self,
        message: Message,
        routing_key: str
    ):
        """Publish message to exchange."""
        try:
            message_body = json.dumps(message.dict()).encode()
            await self.exchange.publish(
                aio_pika.Message(
                    body=message_body,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    priority=len(MessagePriority) - list(MessagePriority).index(message.priority)
                ),
                routing_key=routing_key
            )
        except Exception as e:
            logger.error(f"Failed to publish message: {str(e)}")
            raise
            
    async def subscribe(
        self,
        queue_name: str,
        callback: callable,
        routing_key: str = "#"
    ):
        """Subscribe to messages."""
        try:
            if queue_name not in self.queues:
                queue = await self.declare_queue(queue_name)
            else:
                queue = self.queues[queue_name]
                
            await queue.bind(self.exchange, routing_key)
            
            async def process_message(message: aio_pika.IncomingMessage):
                async with message.process():
                    msg_dict = json.loads(message.body.decode())
                    await callback(Message(**msg_dict))
                    
            await queue.consume(process_message)
            
        except Exception as e:
            logger.error(f"Failed to subscribe: {str(e)}")
            raise

class ProtocolHandler:
    def __init__(self, broker: MessageBroker, redis_client: Redis):
        self.broker = broker
        self.redis = redis_client
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
    async def handle_sync(self, message: Message) -> Optional[Message]:
        """Handle synchronous communication."""
        try:
            if message.type == MessageType.QUERY:
                # Store pending request
                future = asyncio.Future()
                self.pending_requests[message.id] = future
                
                # Publish message
                await self.broker.publish(
                    message,
                    f"query.{message.receiver}"
                )
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(
                        future,
                        timeout=30
                    )
                    return response
                except asyncio.TimeoutError:
                    logger.error(f"Request {message.id} timed out")
                    raise HTTPException(
                        status_code=408,
                        detail="Request timed out"
                    )
                    
            elif message.type == MessageType.RESPONSE:
                # Resolve pending request
                if message.correlation_id in self.pending_requests:
                    future = self.pending_requests.pop(message.correlation_id)
                    future.set_result(message)
                    
        except Exception as e:
            logger.error(f"Error in sync handler: {str(e)}")
            raise
            
    async def handle_async(self, message: Message):
        """Handle asynchronous communication."""
        try:
            routing_key = f"async.{message.receiver}" if message.receiver else "async.broadcast"
            await self.broker.publish(message, routing_key)
        except Exception as e:
            logger.error(f"Error in async handler: {str(e)}")
            raise
            
    async def handle_pubsub(self, message: Message):
        """Handle publish-subscribe communication."""
        try:
            if not message.topic:
                raise HTTPException(
                    status_code=400,
                    detail="Topic is required for pub/sub"
                )
                
            routing_key = f"topic.{message.topic}"
            await self.broker.publish(message, routing_key)
        except Exception as e:
            logger.error(f"Error in pubsub handler: {str(e)}")
            raise

class CommunicationManager:
    def __init__(self):
        # Initialize connections
        rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        self.broker = MessageBroker(rabbitmq_url)
        self.redis = redis.from_url(redis_url)
        self.protocol_handler = ProtocolHandler(self.broker, self.redis)
        
        self.configs: Dict[str, CommunicationConfig] = {}
        
    async def initialize(self):
        """Initialize connections and subscriptions."""
        await self.broker.connect()
        
    async def shutdown(self):
        """Cleanup connections."""
        await self.broker.close()
        
    def register_config(self, config_id: str, config: CommunicationConfig):
        """Register communication configuration."""
        self.configs[config_id] = config
        
    async def send_message(
        self,
        config_id: str,
        message: Message,
        background_tasks: BackgroundTasks
    ) -> Optional[Message]:
        """Send message using appropriate protocol."""
        if config_id not in self.configs:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration {config_id} not found"
            )
            
        config = self.configs[config_id]
        
        try:
            # Rate limiting
            rate_limit = config.rate_limits.get(message.type, 1000)
            current = int(self.redis.get(f"rate:{message.sender}") or 0)
            
            if current >= rate_limit:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded"
                )
                
            self.redis.incr(f"rate:{message.sender}")
            self.redis.expire(f"rate:{message.sender}", 60)  # Reset after 1 minute
            
            # Handle message based on protocol
            if message.protocol == ProtocolType.SYNC:
                return await self.protocol_handler.handle_sync(message)
            elif message.protocol == ProtocolType.ASYNC:
                await self.protocol_handler.handle_async(message)
            elif message.protocol == ProtocolType.PUBSUB:
                await self.protocol_handler.handle_pubsub(message)
                
            return None
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing communication service...")
    try:
        comm_manager = CommunicationManager()
        await comm_manager.initialize()
        app.state.comm_manager = comm_manager
        logger.info("Communication service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize communication service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down communication service...")
    await app.state.comm_manager.shutdown()

app = FastAPI(title="HMAS Communication Service", lifespan=lifespan)

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
    config: CommunicationConfig
):
    """Register a communication configuration."""
    try:
        request.app.state.comm_manager.register_config(config_id, config)
        return {"status": "success", "config_id": config_id}
    except Exception as e:
        logger.error(f"Error registering configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send/{config_id}")
@limiter.limit("1000/minute")
async def send_message(
    request: Request,
    config_id: str,
    message: Message,
    background_tasks: BackgroundTasks
):
    """Send a message."""
    try:
        response = await request.app.state.comm_manager.send_message(
            config_id,
            message,
            background_tasks
        )
        return {"status": "success", "response": response.dict() if response else None}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500) 