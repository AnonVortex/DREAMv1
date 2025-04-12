import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Adaptation configurations
DYNAMIC_SCALING_ENABLED = True
LOAD_BALANCING_ENABLED = True
SELF_MODIFICATION_ENABLED = True
EVOLUTIONARY_ENABLED = True

# Resource monitoring settings
MONITORING_INTERVAL = 5  # seconds
CPU_THRESHOLD = 80  # percentage
MEMORY_THRESHOLD = 80  # percentage
DISK_THRESHOLD = 80  # percentage
NETWORK_THRESHOLD = 1000  # MB/s

# Scaling settings
MIN_INSTANCES = 1
MAX_INSTANCES = 10
SCALE_UP_THRESHOLD = 70  # percentage
SCALE_DOWN_THRESHOLD = 30  # percentage
COOLDOWN_PERIOD = 300  # seconds

# Load balancing settings
LOAD_BALANCING_STRATEGY = "round_robin"  # or "least_loaded"
MAX_QUEUE_LENGTH = 100
TASK_TIMEOUT = 30  # seconds

# Self-modification settings
MODIFICATION_HISTORY_SIZE = 100
ROLLBACK_ENABLED = True
MAX_MODIFICATION_DEPTH = 5

# Evolutionary settings
POPULATION_SIZE = 10
GENERATION_LIMIT = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
FITNESS_THRESHOLD = 0.9

# Docker settings
DOCKER_NETWORK = "hmas_network"
DOCKER_REGISTRY = os.getenv("DOCKER_REGISTRY", "localhost:5000")
IMAGE_PREFIX = "hmas_"

# Performance settings
MAX_RESPONSE_TIME = 1.0  # seconds
MIN_THROUGHPUT = 10  # requests/second 