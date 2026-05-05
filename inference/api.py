"""
Enhanced FastAPI with Hybrid Siamese + Prototypical Endpoints
Production-grade REST API for face verification

Author: LAP Project
Version: 2.0
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import cv2
import numpy as np
from pathlib import Path
import json
import threading
import logging
from datetime import datetime
import io

from models.siamese import SiameseNetwork
from models.prototypical import PrototypicalNetwork
from inference.config import MODEL_REGISTRY
from inference.validation import validate_image


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="LAP Hybrid Face Verification API",
    description="Few-shot face verification with Siamese + Prototypical Networks",
    version="2.0"
)

# CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static/css", StaticFiles(directory="ui/css"), name="css")
app.mount("/static/js", StaticFiles(directory="ui/js"), name="js")


# ============================================================================
# DATA MODELS
# ============================================================================

class EnrollmentStore:
    """Thread-safe storage for enrolled face prototypes."""
    
    def __init__(self, filepath="data/enrollments.json"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.data = self._load()
    
    def _load(self):
        """Load enrollments from JSON."""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save(self):
        """Save enrollments to JSON."""
        with open(self.filepath, 'w') as f:
            # Convert tensors to lists for JSON serialization
            data_to_save = {}
            for user_id, user_data in self.data.items():
                data_to_save[user_id] = {
                    'prototype': user_data.get('prototype'),
                    'num_images': user_data.get('num_images'),
                    'enrolled_at': user_data.get('enrolled_at'),
                }
            json.dump(data_to_save, f, indent=2)
    
    def enroll(self, user_id: str, prototype: np.ndarray, num_images: int):
        """Store prototype for a user."""
        with self.lock:
            self.data[user_id] = {
                'prototype': prototype.tolist() if isinstance(prototype, np.ndarray) else prototype,
                'num_images': num_images,
                'enrolled_at': datetime.now().isoformat(),
            }
            self._save()
    
    def get(self, user_id: str) -> Optional[Dict]:
        """Get enrollment data for a user."""
        with self.lock:
            data = self.data.get(user_id)
            if data:
                data['prototype'] = torch.tensor(data['prototype'], dtype=torch.float32)
            return data
    
    def delete(self, user_id: str):
        """Delete enrollment."""
        with self.lock:
            if user_id in self.data:
                del self.data[user_id]
                self._save()
    
    def list_users(self) -> List[str]:
        """List all enrolled users."""
        with self.lock:
            return list(self.data.keys())


class EngineManager:
    """Lazy-load and manage model instances."""
    
    def __init__(self):
        self.engines = {}
        self.locks = {}
    
    def get_engine(self, modality: str, model_type: str, device='cuda'):
        """Get or create engine for a model."""
        key = f"{modality}_{model_type}"
        
        if key not in self.locks:
            self.locks[key] = threading.Lock()
        
        with self.locks[key]:
            if key not in self.engines:
                logger.info(f"Loading {modality} {model_type} model...")
                
                try:
                    config_path, checkpoint, threshold = MODEL_REGISTRY[
                        (modality, model_type)
                    ].values()
                    
                    if model_type == 'siamese':
                        model = SiameseNetwork(
                            backbone='resnet50',
                            embedding_dim=512,
                            pretrained=True,
                            in_channels=3,
                        )
                    else:  # prototypical
                        model = PrototypicalNetwork(
                            backbone='resnet50',
                            embedding_dim=512,
                            pretrained=True,
                            in_channels=3,
                        )
                    
                    # Load checkpoint
                    checkpoint_data = torch.load(checkpoint, map_location=device)
                    if isinstance(checkpoint_data, dict):
                        model.load_state_dict(checkpoint_data.get('state_dict', checkpoint_data))
                    else:
                        model.load_state_dict(checkpoint_data)
                    
                    model = model.to(device)
                    model.eval()
                    
                    self.engines[key] = {
                        'model': model,
                        'device': device,
                        'threshold': threshold,
                    }
                    
                    logger.info(f"Model {key} loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load model {key}: {e}")
                    raise
            
            return self.engines[key]


# Global instances
enrollment_store = EnrollmentStore("data/enrollments.json")
engine_manager = EngineManager()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_image(file_bytes) -> np.ndarray:
    """Load and preprocess image from bytes."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Validate image
    is_valid, message = validate_image(img)
    if not is_valid:
        raise ValueError(f"Image validation failed: {message}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 112x112
    img = cv2.resize(img, (112, 112))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor (C, H, W)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0",
    }


@app.post("/hybrid/enroll")
async def hybrid_enroll(
    user_id: str,
    files: List[UploadFile] = File(...),
):
    """
    Enroll a user with 3-5 face images.
    
    Creates prototype from multiple enrollment images.
    
    Args:
        user_id: Unique user identifier
        files: List of 3-5 face images
    
    Returns:
        Enrollment status and prototype info
    """
    if len(files) < 3 or len(files) > 5:
        raise HTTPException(
            status_code=400,
            detail="Please provide 3-5 enrollment images"
        )
    
    try:
        # Get models
        siamese_engine = engine_manager.get_engine('face', 'siamese')
        proto_engine = engine_manager.get_engine('face', 'prototypical')
        
        device = siamese_engine['device']
        siamese_model = siamese_engine['model']
        proto_model = proto_engine['model']
        
        # Process images
        embeddings = []
        for file in files:
            file_bytes = await file.read()
            img = load_image(file_bytes)
            img = img.to(device)
            
            with torch.no_grad():
                emb = proto_model.encode(img).squeeze(0)
                embeddings.append(emb.cpu())
        
        embeddings = torch.stack(embeddings)  # (N, embedding_dim)
        
        # Create prototype (mean)
        prototype = embeddings.mean(dim=0)
        
        # Store enrollment
        enrollment_store.enroll(
            user_id=user_id,
            prototype=prototype.numpy(),
            num_images=len(files),
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "num_enrolled_images": len(files),
            "prototype_dim": int(prototype.shape[0]),
            "enrolled_at": datetime.now().isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Enrollment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hybrid/verify")
async def hybrid_verify(
    user_id: str,
    file: UploadFile = File(...),
):
    """
    Verify query image against enrolled prototype.
    
    Uses both Siamese and Prototypical scoring with weighted fusion.
    
    Args:
        user_id: User to verify against
        file: Query face image
    
    Returns:
        Verification result with confidence scores
    """
    try:
        # Get enrollment
        enrollment = enrollment_store.get(user_id)
        if not enrollment:
            raise HTTPException(status_code=404, detail=f"User {user_id} not enrolled")
        
        prototype = enrollment['prototype'].to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get models
        siamese_engine = engine_manager.get_engine('face', 'siamese')
        proto_engine = engine_manager.get_engine('face', 'prototypical')
        
        device = siamese_engine['device']
        siamese_model = siamese_engine['model']
        proto_model = proto_engine['model']
        
        # Load and process query image
        file_bytes = await file.read()
        query_img = load_image(file_bytes)
        query_img = query_img.to(device)
        
        with torch.no_grad():
            # Prototypical scoring
            proto_result = proto_model.verify(query_img.squeeze(0), prototype)
            proto_score = proto_result['confidence']
            
            # Siamese scoring (against prototype as reference)
            siamese_output = siamese_model(query_img, prototype.unsqueeze(0).to(device))
            siamese_score = torch.sigmoid(siamese_output['cosine_sim']).item()
        
        # Fusion
        final_score = (
            0.5 * proto_score +
            0.5 * siamese_score
        )
        
        # Decision
        threshold = 0.65
        verdict = "MATCH" if final_score > threshold else "NO MATCH"
        
        return {
            "user_id": user_id,
            "verdict": verdict,
            "confidence": float(final_score),
            "proto_score": float(proto_score),
            "siamese_score": float(siamese_score),
            "threshold": threshold,
            "verified_at": datetime.now().isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hybrid/rank")
async def hybrid_rank(
    file: UploadFile = File(...),
    top_n: int = 5,
):
    """
    Rank query image against all enrolled users.
    
    Args:
        file: Query face image
        top_n: Number of top matches to return
    
    Returns:
        Ranked list of matches
    """
    try:
        users = enrollment_store.list_users()
        if not users:
            raise HTTPException(status_code=404, detail="No users enrolled")
        
        proto_engine = engine_manager.get_engine('face', 'prototypical')
        device = proto_engine['device']
        proto_model = proto_engine['model']
        
        # Load query
        file_bytes = await file.read()
        query_img = load_image(file_bytes)
        query_img = query_img.to(device)
        
        scores = []
        with torch.no_grad():
            for user_id in users:
                enrollment = enrollment_store.get(user_id)
                prototype = enrollment['prototype'].to(device)
                
                result = proto_model.verify(query_img.squeeze(0), prototype)
                scores.append({
                    'user_id': user_id,
                    'confidence': result['confidence'],
                    'rank': 0,
                })
        
        # Sort by confidence descending
        scores = sorted(scores, key=lambda x: x['confidence'], reverse=True)
        
        # Add rank
        for i, score in enumerate(scores[:top_n]):
            score['rank'] = i + 1
        
        return {
            "query_timestamp": datetime.now().isoformat(),
            "total_enrolled_users": len(users),
            "top_matches": scores[:top_n],
        }
    
    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/hybrid/delete/{user_id}")
async def delete_enrollment(user_id: str):
    """Delete user enrollment."""
    try:
        enrollment_store.delete(user_id)
        return {"status": "deleted", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hybrid/list")
async def list_enrollments():
    """List all enrolled users."""
    try:
        users = enrollment_store.list_users()
        return {
            "total_users": len(users),
            "users": users,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
