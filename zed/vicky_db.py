import os
import time
import asyncio
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, text

# Load database configuration from environment variable
# e.g., DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/vicky"
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", 
    "sqlite+aiosqlite:///vicky_logs.db"
)

# Async database engine configuration
# Disable connection pooling warnings on SQLite
connect_args: Dict[str, Any] = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args=connect_args
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

class SpatialLog(Base):
    __tablename__ = "vicky_spatial_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    timestamp = Column(Float, index=True, nullable=False)
    compute_node = Column(String, nullable=False)
    mode_flag = Column(String, nullable=False)
    
    # User Spatial Pose
    pose_x = Column(Float, default=0.0, nullable=False)
    pose_y = Column(Float, default=0.0, nullable=False)
    pose_z = Column(Float, default=0.0, nullable=False)
    roll = Column(Float, default=0.0, nullable=False)
    pitch = Column(Float, default=0.0, nullable=False)
    yaw = Column(Float, default=0.0, nullable=False)
    
    # Spatial Depth Zones
    left_clearance_mm = Column(Float, default=0.0, nullable=False)
    center_clearance_mm = Column(Float, default=0.0, nullable=False)
    right_clearance_mm = Column(Float, default=0.0, nullable=False)
    escape_vector = Column(String, default="STOP", nullable=False)
    
    # Semantic Objects (JSON list of dicts)
    semantic_objects_in_frustum = Column(JSON, default=list, nullable=False)
    
    # Performance Metrics
    inference_latency_ms = Column(Float, default=0.0, nullable=False)
    network_rtt_ms = Column(Float, default=0.0, nullable=False)
    total_srt_ms = Column(Float, default=0.0, nullable=False)
    hallucination_flag = Column(Boolean, default=False, nullable=False)
    
    # Smartphone IMU Metrics
    phone_accel_x = Column(Float, default=0.0, nullable=True)
    phone_accel_y = Column(Float, default=0.0, nullable=True)
    phone_accel_z = Column(Float, default=0.0, nullable=True)
    phone_roll = Column(Float, default=0.0, nullable=True)
    phone_pitch = Column(Float, default=0.0, nullable=True)
    phone_yaw = Column(Float, default=0.0, nullable=True)

class OccupancyMap(Base):
    __tablename__ = "vicky_occupancy_maps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    timestamp = Column(Float, index=True, nullable=False)
    pose_x = Column(Float, default=0.0, nullable=False)
    pose_z = Column(Float, default=0.0, nullable=False)
    yaw = Column(Float, default=0.0, nullable=False)
    grid_data = Column(JSON, nullable=False)

# Asynchronous log queue worker to enable non-blocking logging
class AsyncLogCollector:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self.running: bool = False

    async def start(self) -> None:
        """Initialize database tables and start database writer worker task."""
        max_retries = 10
        retry_delay = 3
        for attempt in range(1, max_retries + 1):
            try:
                async with engine.begin() as conn:
                    # Create tables if they do not exist
                    await conn.run_sync(Base.metadata.create_all)
                    
                    # Run dynamic SQLite migrations to add columns if they are missing
                    def migrate_db(connection):
                        # Query table info to check existing columns
                        result = connection.execute(text("PRAGMA table_info(vicky_spatial_logs)"))
                        columns = [row[1] for row in result.fetchall()]
                        
                        new_cols = {
                            "phone_accel_x": "FLOAT DEFAULT 0.0",
                            "phone_accel_y": "FLOAT DEFAULT 0.0",
                            "phone_accel_z": "FLOAT DEFAULT 0.0",
                            "phone_roll": "FLOAT DEFAULT 0.0",
                            "phone_pitch": "FLOAT DEFAULT 0.0",
                            "phone_yaw": "FLOAT DEFAULT 0.0"
                        }
                        
                        for col_name, col_type in new_cols.items():
                            if col_name not in columns:
                                print(f"[DB LOGGER] Running migration: Adding column {col_name} to vicky_spatial_logs")
                                connection.execute(text(f"ALTER TABLE vicky_spatial_logs ADD COLUMN {col_name} {col_type}"))
                    
                    await conn.run_sync(migrate_db)
                break
            except Exception as e:
                print(f"[DB LOGGER WARNING] Database connection attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    raise e
                await asyncio.sleep(retry_delay)
        
        self.running = True
        self.worker_task = asyncio.create_task(self._db_writer_loop())
        print(f"[DB LOGGER] Async log writer started using: {DATABASE_URL}")

    async def stop(self) -> None:
        """Stop worker task and flush queue."""
        self.running = False
        if self.worker_task:
            # Insert a sentinel to stop loop
            await self.queue.put(None)
            await self.worker_task
        print("[DB LOGGER] Async log writer stopped.")

    async def log_frame_data(self, frame_payload: Dict[str, Any]) -> None:
        """Queue frame data for non-blocking asynchronous writing."""
        await self.queue.put(frame_payload)

    async def _db_writer_loop(self) -> None:
        while self.running or not self.queue.empty():
            try:
                payload = await self.queue.get()
                if payload is None:
                    self.queue.task_done()
                    break
                
                await self._write_to_db(payload)
                self.queue.task_done()
            except Exception as e:
                print(f"[DB LOGGER ERROR] Failed to process database logging queue item: {e}")

    async def _write_to_db(self, payload: Dict[str, Any]) -> None:
        metadata = payload.get("packet_metadata", {})
        pose = payload.get("user_spatial_pose", {})
        pos = pose.get("position_meters", {})
        rot = pose.get("rotation_degrees", {})
        zones = payload.get("spatial_depth_zones", {})
        perf = payload.get("performance_metrics", {})
        objects = payload.get("semantic_objects_in_frustum", [])
        
        # Extract smartphone imu if present
        phone_imu = payload.get("smartphone_imu")
        if phone_imu:
            phone_accel = phone_imu.get("linear_accel", {})
            phone_rot = phone_imu.get("rotation_rpy", {})
            p_ax = float(phone_accel.get("x", 0.0))
            p_ay = float(phone_accel.get("y", 0.0))
            p_az = float(phone_accel.get("z", 0.0))
            p_roll = float(phone_rot.get("roll", 0.0))
            p_pitch = float(phone_rot.get("pitch", 0.0))
            p_yaw = float(phone_rot.get("yaw", 0.0))
        else:
            p_ax, p_ay, p_az = 0.0, 0.0, 0.0
            p_roll, p_pitch, p_yaw = 0.0, 0.0, 0.0

        db_log = SpatialLog(
            session_id=str(metadata.get("session_id", "default_session")),
            timestamp=float(metadata.get("timestamp", time.time())),
            compute_node=str(metadata.get("compute_node", "unknown_node")),
            mode_flag=str(metadata.get("mode_flag", "A")),
            
            pose_x=float(pos.get("x", 0.0)),
            pose_y=float(pos.get("y", 0.0)),
            pose_z=float(pos.get("z", 0.0)),
            roll=float(rot.get("roll", 0.0)),
            pitch=float(rot.get("pitch", 0.0)),
            yaw=float(rot.get("yaw", 0.0)),
            
            left_clearance_mm=float(zones.get("left_clearance_mm", 0.0)),
            center_clearance_mm=float(zones.get("center_clearance_mm", 0.0)),
            right_clearance_mm=float(zones.get("right_clearance_mm", 0.0)),
            escape_vector=str(zones.get("escape_vector", "STOP")),
            
            semantic_objects_in_frustum=objects,
            
            inference_latency_ms=float(perf.get("inference_latency_ms", 0.0)),
            network_rtt_ms=float(perf.get("network_rtt_ms", 0.0)),
            total_srt_ms=float(perf.get("total_srt_ms", 0.0)),
            hallucination_flag=bool(perf.get("hallucination_flag", False)),
            
            phone_accel_x=p_ax,
            phone_accel_y=p_ay,
            phone_accel_z=p_az,
            phone_roll=p_roll,
            phone_pitch=p_pitch,
            phone_yaw=p_yaw
        )

        async with AsyncSessionLocal() as session:
            async with session.begin():
                session.add(db_log)
            await session.commit()

# Singleton instance for simple app-wide usage
db_logger = AsyncLogCollector()
