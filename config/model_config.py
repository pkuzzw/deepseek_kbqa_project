# config/model_config.py
import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # 基础配置
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 模型运行模式配置
    run_mode: str = "api"  # 可选 ["local", "api", "hybrid"]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Qwen2.5-7B 本地模型配置
    local_model: dict = field(default_factory=lambda: {
        "model_path": "/path/to/Qwen2.5-7B-Instruct",
        "precision": "fp16",          # 可选 ["fp32", "fp16", "bf16"]
        "quantization": "none",       # 可选 ["none", "4bit", "8bit"]
        "cache_dir": "model_cache",
        "max_seq_length": 32768
    })
    
    # API配置
    api_config: dict = field(default_factory=lambda: {
        "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": "sk-your-key-here",
        "timeout": 30.0,
        "max_retries": 3
    })
    
    # 检索系统配置
    retrievers: dict = field(default_factory=lambda: {
        "active_retrievers": ["bm25", "dpr"],  # 修改启用检索器
        "dpr": {
            "question_model": "facebook/dpr-question_encoder-single-nq-base",
            "context_model": "facebook/dpr-ctx_encoder-single-nq-base",
            "index_path": "dpr_faiss_index.bin"
        }
    })
    
    # 日志配置
    logging: dict = field(default_factory=lambda: {
        "level": "INFO",
        "file": "logs/system.log",
        "max_size": 100  # MB
    })

    def __post_init__(self):
        # 自动路径处理
        self.local_model["model_path"] = os.path.expanduser(self.local_model["model_path"])
        self.local_model["cache_dir"] = os.path.join(self.project_root, self.local_model["cache_dir"])
        
        # 量化配置验证
        if self.local_model["quantization"] not in ["none", "4bit", "8bit"]:
            raise ValueError(f"Invalid quantization: {self.local_model['quantization']}")
        
        # 创建必要目录
        os.makedirs(self.local_model["cache_dir"], exist_ok=True)
        os.makedirs(os.path.dirname(self.logging["file"]), exist_ok=True)

# 配置实例化
model_config = ModelConfig()