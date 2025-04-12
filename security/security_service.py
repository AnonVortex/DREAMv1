import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import json
import jwt
import bcrypt
import asyncio
import secrets
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
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

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class UserRole(str, Enum):
    ADMIN = "admin"
    AGENT = "agent"
    SERVICE = "service"
    USER = "user"

class SecurityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(str, Enum):
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_TOKEN = "invalid_token"
    MALFORMED_REQUEST = "malformed_request"

class User(BaseModel):
    username: str
    email: EmailStr
    role: UserRole
    disabled: bool = False
    security_level: SecurityLevel = SecurityLevel.MEDIUM

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime

class TokenData(BaseModel):
    username: str
    role: UserRole
    security_level: SecurityLevel

class SecurityEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: secrets.token_urlsafe(16))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str
    severity: SecurityLevel
    source_ip: str
    user: Optional[str]
    details: Dict[str, Any]
    resolved: bool = False

class ThreatAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: secrets.token_urlsafe(16))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    threat_type: ThreatType
    severity: SecurityLevel
    source: str
    details: Dict[str, Any]
    mitigated: bool = False

class SecurityManager:
    def __init__(self):
        self.users_db: Dict[str, UserInDB] = {}
        self.security_events: List[SecurityEvent] = []
        self.threat_alerts: List[ThreatAlert] = []
        self.blacklisted_tokens: set = set()
        
    async def create_user(self, username: str, email: str, password: str, role: UserRole) -> User:
        """Create a new user with hashed password."""
        if username in self.users_db:
            raise HTTPException(status_code=400, detail="Username already registered")
            
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode(), salt)
        
        user = UserInDB(
            username=username,
            email=email,
            role=role,
            hashed_password=hashed_password.decode()
        )
        self.users_db[username] = user
        return User(**user.dict(exclude={'hashed_password'}))
        
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user credentials."""
        user = self.users_db.get(username)
        if not user:
            return None
            
        if not bcrypt.checkpw(password.encode(), user.hashed_password.encode()):
            return None
            
        return User(**user.dict(exclude={'hashed_password'}))
        
    async def create_access_token(self, user: User) -> Token:
        """Create JWT access token."""
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        expires_at = datetime.utcnow() + expires_delta
        
        token_data = {
            "sub": user.username,
            "role": user.role,
            "security_level": user.security_level,
            "exp": expires_at
        }
        
        access_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        
        return Token(
            access_token=access_token,
            expires_at=expires_at
        )
        
    async def verify_token(self, token: str) -> TokenData:
        """Verify JWT token and return token data."""
        try:
            if token in self.blacklisted_tokens:
                raise HTTPException(
                    status_code=401,
                    detail="Token has been revoked"
                )
                
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            role = payload.get("role")
            security_level = payload.get("security_level")
            
            if not all([username, role, security_level]):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token payload"
                )
                
            return TokenData(
                username=username,
                role=UserRole(role),
                security_level=SecurityLevel(security_level)
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate token"
            )
            
    async def revoke_token(self, token: str):
        """Revoke a JWT token."""
        self.blacklisted_tokens.add(token)
        
    async def log_security_event(
        self,
        event_type: str,
        severity: SecurityLevel,
        source_ip: str,
        user: Optional[str],
        details: Dict[str, Any]
    ) -> SecurityEvent:
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user=user,
            details=details
        )
        self.security_events.append(event)
        
        if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            logger.warning(f"High severity security event: {event.dict()}")
            
        return event
        
    async def create_threat_alert(
        self,
        threat_type: ThreatType,
        severity: SecurityLevel,
        source: str,
        details: Dict[str, Any]
    ) -> ThreatAlert:
        """Create a new threat alert."""
        alert = ThreatAlert(
            threat_type=threat_type,
            severity=severity,
            source=source,
            details=details
        )
        self.threat_alerts.append(alert)
        
        if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            logger.warning(f"High severity threat alert: {alert.dict()}")
            
        return alert
        
    async def get_security_events(
        self,
        severity: Optional[SecurityLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Get filtered security events."""
        filtered_events = self.security_events
        
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]
            
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
            
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
            
        return filtered_events[-limit:]
        
    async def get_threat_alerts(
        self,
        threat_type: Optional[ThreatType] = None,
        severity: Optional[SecurityLevel] = None,
        mitigated: Optional[bool] = None,
        limit: int = 100
    ) -> List[ThreatAlert]:
        """Get filtered threat alerts."""
        filtered_alerts = self.threat_alerts
        
        if threat_type:
            filtered_alerts = [a for a in filtered_alerts if a.threat_type == threat_type]
            
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
            
        if mitigated is not None:
            filtered_alerts = [a for a in filtered_alerts if a.mitigated == mitigated]
            
        return filtered_alerts[-limit:]

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing security service...")
    try:
        security_manager = SecurityManager()
        app.state.security_manager = security_manager
        logger.info("Security service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize security service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down security service...")

app = FastAPI(title="HMAS Security Service", lifespan=lifespan)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    security_manager: SecurityManager = Depends(lambda: app.state.security_manager)
) -> User:
    """Get current user from token."""
    token_data = await security_manager.verify_token(token)
    user = security_manager.users_db.get(token_data.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.disabled:
        raise HTTPException(status_code=400, detail="User is disabled")
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@app.post("/users")
@limiter.limit("10/minute")
async def create_user(
    request: Request,
    username: str,
    email: EmailStr,
    password: str,
    role: UserRole,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new user."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    try:
        user = await request.app.state.security_manager.create_user(
            username,
            email,
            password,
            role
        )
        return user
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/token")
@limiter.limit("20/minute")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """Login to get access token."""
    try:
        user = await request.app.state.security_manager.authenticate_user(
            form_data.username,
            form_data.password
        )
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password"
            )
            
        token = await request.app.state.security_manager.create_access_token(user)
        return token
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/token/revoke")
@limiter.limit("10/minute")
async def revoke_token(
    request: Request,
    token: str,
    current_user: User = Depends(get_current_active_user)
):
    """Revoke an access token."""
    try:
        await request.app.state.security_manager.revoke_token(token)
        return {"status": "success", "message": "Token revoked"}
    except Exception as e:
        logger.error(f"Error revoking token: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/security/events")
@limiter.limit("50/minute")
async def log_security_event(
    request: Request,
    event_type: str,
    severity: SecurityLevel,
    details: Dict[str, Any],
    current_user: User = Depends(get_current_active_user)
):
    """Log a security event."""
    try:
        event = await request.app.state.security_manager.log_security_event(
            event_type,
            severity,
            request.client.host,
            current_user.username,
            details
        )
        return event
    except Exception as e:
        logger.error(f"Error logging security event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/security/threats")
@limiter.limit("50/minute")
async def create_threat_alert(
    request: Request,
    threat_type: ThreatType,
    severity: SecurityLevel,
    details: Dict[str, Any],
    current_user: User = Depends(get_current_active_user)
):
    """Create a threat alert."""
    try:
        alert = await request.app.state.security_manager.create_threat_alert(
            threat_type,
            severity,
            request.client.host,
            details
        )
        return alert
    except Exception as e:
        logger.error(f"Error creating threat alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/security/events")
@limiter.limit("100/minute")
async def get_security_events(
    request: Request,
    severity: Optional[SecurityLevel] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user)
):
    """Get security events."""
    try:
        events = await request.app.state.security_manager.get_security_events(
            severity,
            start_time,
            end_time,
            limit
        )
        return events
    except Exception as e:
        logger.error(f"Error getting security events: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/security/threats")
@limiter.limit("100/minute")
async def get_threat_alerts(
    request: Request,
    threat_type: Optional[ThreatType] = None,
    severity: Optional[SecurityLevel] = None,
    mitigated: Optional[bool] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user)
):
    """Get threat alerts."""
    try:
        alerts = await request.app.state.security_manager.get_threat_alerts(
            threat_type,
            severity,
            mitigated,
            limit
        )
        return alerts
    except Exception as e:
        logger.error(f"Error getting threat alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8900) 