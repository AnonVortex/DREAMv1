import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from datetime import datetime, timedelta
import json
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class AggregationType(str, Enum):
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    CUSTOM = "custom"

class TimeWindow(str, Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    CUSTOM = "custom"

class DataSource(str, Enum):
    METRICS = "metrics"
    LOGS = "logs"
    EVENTS = "events"
    TRACES = "traces"
    CUSTOM = "custom"

class AggregationConfig(BaseModel):
    aggregation_type: AggregationType
    data_source: DataSource
    time_window: TimeWindow
    custom_window_seconds: Optional[int] = None
    group_by: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    custom_function: Optional[str] = None

class DataPoint(BaseModel):
    timestamp: datetime
    source: DataSource
    value: Any
    metadata: Optional[Dict[str, Any]] = None

class AggregationResult(BaseModel):
    aggregation_id: str
    config_id: str
    result: Dict[str, Any]
    start_time: datetime
    end_time: datetime
    metadata: Optional[Dict[str, Any]] = None

class TimeSeriesAggregator:
    def __init__(self):
        self.aggregation_functions = {
            AggregationType.COUNT: self._count,
            AggregationType.SUM: self._sum,
            AggregationType.AVERAGE: self._average,
            AggregationType.MIN: self._min,
            AggregationType.MAX: self._max,
            AggregationType.MEDIAN: self._median,
            AggregationType.PERCENTILE: self._percentile
        }
        
    async def aggregate(
        self,
        data_points: List[DataPoint],
        config: AggregationConfig
    ) -> Dict[str, Any]:
        """Aggregate data points according to configuration."""
        # Filter data points
        filtered_points = self._filter_points(data_points, config.filters)
        
        # Group data points
        grouped_points = self._group_points(filtered_points, config.group_by)
        
        # Calculate aggregations
        results = {}
        for group, points in grouped_points.items():
            if config.aggregation_type == AggregationType.CUSTOM:
                results[group] = self._custom_aggregate(
                    points,
                    config.custom_function
                )
            else:
                results[group] = self.aggregation_functions[config.aggregation_type](
                    points
                )
                
        return results
        
    def _filter_points(
        self,
        points: List[DataPoint],
        filters: Optional[Dict[str, Any]]
    ) -> List[DataPoint]:
        """Filter data points based on criteria."""
        if not filters:
            return points
            
        filtered = []
        for point in points:
            if self._matches_filters(point, filters):
                filtered.append(point)
                
        return filtered
        
    def _group_points(
        self,
        points: List[DataPoint],
        group_by: Optional[List[str]]
    ) -> Dict[str, List[DataPoint]]:
        """Group data points by specified fields."""
        if not group_by:
            return {"all": points}
            
        groups = {}
        for point in points:
            key = tuple(
                point.metadata.get(field, "unknown")
                for field in group_by
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(point)
            
        return groups
        
    def _matches_filters(self, point: DataPoint, filters: Dict[str, Any]) -> bool:
        """Check if point matches filter criteria."""
        for key, value in filters.items():
            if key not in point.metadata:
                return False
            if point.metadata[key] != value:
                return False
        return True
        
    def _count(self, points: List[DataPoint]) -> int:
        """Count number of data points."""
        return len(points)
        
    def _sum(self, points: List[DataPoint]) -> float:
        """Calculate sum of data points."""
        return sum(float(p.value) for p in points)
        
    def _average(self, points: List[DataPoint]) -> float:
        """Calculate average of data points."""
        if not points:
            return 0.0
        return self._sum(points) / len(points)
        
    def _min(self, points: List[DataPoint]) -> float:
        """Find minimum value in data points."""
        if not points:
            return float("inf")
        return min(float(p.value) for p in points)
        
    def _max(self, points: List[DataPoint]) -> float:
        """Find maximum value in data points."""
        if not points:
            return float("-inf")
        return max(float(p.value) for p in points)
        
    def _median(self, points: List[DataPoint]) -> float:
        """Calculate median of data points."""
        if not points:
            return 0.0
        values = sorted(float(p.value) for p in points)
        mid = len(values) // 2
        if len(values) % 2 == 0:
            return (values[mid - 1] + values[mid]) / 2
        return values[mid]
        
    def _percentile(self, points: List[DataPoint], percentile: float = 95) -> float:
        """Calculate percentile of data points."""
        if not points:
            return 0.0
        values = sorted(float(p.value) for p in points)
        k = (len(values) - 1) * (percentile / 100)
        f = np.floor(k)
        c = np.ceil(k)
        if f == c:
            return values[int(k)]
        d0 = values[int(f)] * (c - k)
        d1 = values[int(c)] * (k - f)
        return d0 + d1
        
    def _custom_aggregate(self, points: List[DataPoint], function_name: str) -> Any:
        """Apply custom aggregation function."""
        # Implement custom aggregation logic
        return None

class WindowManager:
    def __init__(self):
        self.window_sizes = {
            TimeWindow.MINUTE: timedelta(minutes=1),
            TimeWindow.HOUR: timedelta(hours=1),
            TimeWindow.DAY: timedelta(days=1),
            TimeWindow.WEEK: timedelta(weeks=1),
            TimeWindow.MONTH: timedelta(days=30)
        }
        
    def get_window_bounds(
        self,
        window: TimeWindow,
        custom_seconds: Optional[int] = None
    ) -> tuple[datetime, datetime]:
        """Calculate time window bounds."""
        end_time = datetime.now()
        
        if window == TimeWindow.CUSTOM and custom_seconds is not None:
            window_size = timedelta(seconds=custom_seconds)
        else:
            window_size = self.window_sizes[window]
            
        start_time = end_time - window_size
        return start_time, end_time

class AggregationManager:
    def __init__(self):
        self.time_series_aggregator = TimeSeriesAggregator()
        self.window_manager = WindowManager()
        self.configs: Dict[str, AggregationConfig] = {}
        self.data_points: List[DataPoint] = []
        
    def register_config(self, config_id: str, config: AggregationConfig):
        """Register an aggregation configuration."""
        self.configs[config_id] = config
        
    def add_data_point(self, point: DataPoint):
        """Add a new data point."""
        self.data_points.append(point)
        
    async def aggregate_data(
        self,
        config_id: str,
        background_tasks: BackgroundTasks
    ) -> AggregationResult:
        """Aggregate data according to configuration."""
        if config_id not in self.configs:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration {config_id} not found"
            )
            
        config = self.configs[config_id]
        start_time, end_time = self.window_manager.get_window_bounds(
            config.time_window,
            config.custom_window_seconds
        )
        
        # Filter points within time window
        window_points = [
            p for p in self.data_points
            if start_time <= p.timestamp <= end_time
            and p.source == config.data_source
        ]
        
        try:
            result = await self.time_series_aggregator.aggregate(
                window_points,
                config
            )
            
            return AggregationResult(
                aggregation_id=f"agg_{datetime.now().isoformat()}",
                config_id=config_id,
                result=result,
                start_time=start_time,
                end_time=end_time,
                metadata={
                    "point_count": len(window_points),
                    "aggregation_type": config.aggregation_type,
                    "data_source": config.data_source
                }
            )
            
        except Exception as e:
            logger.error(f"Error aggregating data: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing aggregation service...")
    try:
        aggregation_manager = AggregationManager()
        app.state.aggregation_manager = aggregation_manager
        logger.info("Aggregation service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize aggregation service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down aggregation service...")

app = FastAPI(title="HMAS Aggregation Service", lifespan=lifespan)

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
    config: AggregationConfig
):
    """Register an aggregation configuration."""
    try:
        request.app.state.aggregation_manager.register_config(config_id, config)
        return {"status": "success", "config_id": config_id}
    except Exception as e:
        logger.error(f"Error registering configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data")
@limiter.limit("1000/minute")
async def add_data_point(
    request: Request,
    point: DataPoint
):
    """Add a new data point."""
    try:
        request.app.state.aggregation_manager.add_data_point(point)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error adding data point: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/aggregate/{config_id}")
@limiter.limit("50/minute")
async def aggregate_data(
    request: Request,
    config_id: str,
    background_tasks: BackgroundTasks
):
    """Aggregate data using specified configuration."""
    try:
        result = await request.app.state.aggregation_manager.aggregate_data(
            config_id,
            background_tasks
        )
        return result.dict()
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error aggregating data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8700) 