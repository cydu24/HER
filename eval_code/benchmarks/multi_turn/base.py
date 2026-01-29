"""
多轮对话任务基类

支持功能：
1. 两步评估：推理 + 评估 可分开执行
2. 缓存/增量保存：支持中断后继续
3. 结果目录管理：统一的输入输出目录结构
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """单条推理结果"""
    sample_id: str  # 样本唯一标识
    model_name: str
    input_data: Dict[str, Any]  # 原始输入
    dialogue: List[Dict[str, Any]]  # 生成的对话
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元信息
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "completed"  # completed, failed, skipped
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """单条评估结果"""
    sample_id: str
    inference_result: InferenceResult
    scores: Dict[str, float]  # 各维度得分
    details: Dict[str, Any] = field(default_factory=dict)  # 详细评估信息
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkConfig:
    """Benchmark 配置"""
    name: str
    data_path: str
    output_dir: str
    # 推理参数
    max_turns: int = 10
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 4096
    # 并发参数
    workers: int = 4
    rpm: int = 60
    delay: float = 0.5
    # 缓存参数
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    # 评估参数
    judge_model: Optional[str] = None
    judge_config: Dict[str, Any] = field(default_factory=dict)


class CacheManager:
    """缓存管理器 - 支持增量保存和断点续传"""
    
    def __init__(self, cache_dir: str, benchmark_name: str, model_name: str):
        self.cache_dir = Path(cache_dir)
        self.benchmark_name = benchmark_name
        self.model_name = model_name
        
        # 创建缓存目录
        self.inference_cache_dir = self.cache_dir / "inference"
        self.eval_cache_dir = self.cache_dir / "evaluation"
        self.inference_cache_dir.mkdir(parents=True, exist_ok=True)
        self.eval_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存索引文件
        self.inference_index_file = self.inference_cache_dir / "index.json"
        self.eval_index_file = self.eval_cache_dir / "index.json"
        
        # 加载索引
        self._inference_index = self._load_index(self.inference_index_file)
        self._eval_index = self._load_index(self.eval_index_file)
    
    def _load_index(self, index_file: Path) -> Dict[str, str]:
        """加载缓存索引"""
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_index(self, index_file: Path, index: Dict[str, str]):
        """保存缓存索引"""
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    
    def _get_sample_hash(self, sample_id: str, input_data: Dict) -> str:
        """生成样本的唯一哈希"""
        content = json.dumps({
            "sample_id": sample_id,
            "input": input_data,
            "model": self.model_name,
            "benchmark": self.benchmark_name
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def has_inference_cache(self, sample_id: str) -> bool:
        """检查是否有推理缓存"""
        return sample_id in self._inference_index
    
    def get_inference_cache(self, sample_id: str) -> Optional[InferenceResult]:
        """获取推理缓存"""
        if sample_id not in self._inference_index:
            return None
        
        cache_file = self.inference_cache_dir / self._inference_index[sample_id]
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return InferenceResult(**data)
        except Exception as e:
            logger.warning(f"Failed to load inference cache for {sample_id}: {e}")
            return None
    
    def save_inference_cache(self, result: InferenceResult):
        """保存推理结果到缓存"""
        cache_filename = f"{result.sample_id}.json"
        cache_file = self.inference_cache_dir / cache_filename
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        
        # 更新索引
        self._inference_index[result.sample_id] = cache_filename
        self._save_index(self.inference_index_file, self._inference_index)
    
    def has_eval_cache(self, sample_id: str) -> bool:
        """检查是否有评估缓存"""
        return sample_id in self._eval_index
    
    def get_eval_cache(self, sample_id: str) -> Optional[EvaluationResult]:
        """获取评估缓存"""
        if sample_id not in self._eval_index:
            return None
        
        cache_file = self.eval_cache_dir / self._eval_index[sample_id]
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 重建嵌套对象
                inference_data = data.pop('inference_result')
                inference_result = InferenceResult(**inference_data)
                return EvaluationResult(inference_result=inference_result, **data)
        except Exception as e:
            logger.warning(f"Failed to load eval cache for {sample_id}: {e}")
            return None
    
    def save_eval_cache(self, result: EvaluationResult):
        """保存评估结果到缓存"""
        cache_filename = f"{result.sample_id}.json"
        cache_file = self.eval_cache_dir / cache_filename
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        
        # 更新索引
        self._eval_index[result.sample_id] = cache_filename
        self._save_index(self.eval_index_file, self._eval_index)
    
    def get_all_inference_results(self) -> List[InferenceResult]:
        """获取所有推理结果"""
        results = []
        for sample_id in self._inference_index:
            result = self.get_inference_cache(sample_id)
            if result:
                results.append(result)
        return results
    
    def get_all_eval_results(self) -> List[EvaluationResult]:
        """获取所有评估结果"""
        results = []
        for sample_id in self._eval_index:
            result = self.get_eval_cache(sample_id)
            if result:
                results.append(result)
        return results
    
    def get_pending_inference_samples(self, all_sample_ids: List[str]) -> List[str]:
        """获取待推理的样本ID"""
        return [sid for sid in all_sample_ids if not self.has_inference_cache(sid)]
    
    def get_pending_eval_samples(self) -> List[str]:
        """获取待评估的样本ID（已推理但未评估）"""
        return [sid for sid in self._inference_index if not self.has_eval_cache(sid)]
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            "total_inference": len(self._inference_index),
            "total_eval": len(self._eval_index),
            "pending_eval": len(self.get_pending_eval_samples())
        }


class MultiTurnBenchmark(ABC):
    """多轮对话基准测试基类"""
    
    name: str = "multi_turn_base"
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化缓存管理器
        cache_dir = config.cache_dir or str(self.output_dir / "cache")
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            benchmark_name=self.name,
            model_name="default"  # 会在 run 时更新
        )
        
        # 加载数据
        self.data = self._load_data()
        
        logger.info(f"Initialized {self.name} benchmark")
        logger.info(f"  - Data path: {config.data_path}")
        logger.info(f"  - Output dir: {config.output_dir}")
        logger.info(f"  - Total samples: {len(self.data)}")
    
    @abstractmethod
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载评测数据"""
        pass
    
    @abstractmethod
    def get_sample_id(self, sample: Dict[str, Any], index: int) -> str:
        """获取样本的唯一标识"""
        pass
    
    @abstractmethod
    async def run_inference_single(
        self,
        sample: Dict[str, Any],
        model: Any,
        **kwargs
    ) -> InferenceResult:
        """对单个样本进行推理"""
        pass
    
    @abstractmethod
    async def evaluate_single(
        self,
        inference_result: InferenceResult,
        judge_model: Optional[Any] = None,
        **kwargs
    ) -> EvaluationResult:
        """评估单条推理结果"""
        pass
    
    def update_model_name(self, model_name: str):
        """更新模型名称（用于缓存隔离）"""
        cache_dir = self.config.cache_dir or str(self.output_dir / "cache")
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            benchmark_name=self.name,
            model_name=model_name
        )
    
    async def run_inference(
        self,
        model: Any,
        model_name: str,
        limit: Optional[int] = None,
        skip_cache: bool = False,
        **kwargs
    ) -> List[InferenceResult]:
        """
        执行推理阶段
        
        Args:
            model: 推理模型
            model_name: 模型名称
            limit: 限制样本数量
            skip_cache: 是否跳过缓存
        
        Returns:
            推理结果列表
        """
        # 更新缓存的模型名称
        self.update_model_name(model_name)
        
        # 获取所有样本ID
        samples = self.data[:limit] if limit else self.data
        all_sample_ids = [self.get_sample_id(s, i) for i, s in enumerate(samples)]
        
        # 获取待处理的样本
        if skip_cache:
            pending_ids = all_sample_ids
        else:
            pending_ids = self.cache_manager.get_pending_inference_samples(all_sample_ids)
        
        cached_count = len(all_sample_ids) - len(pending_ids)
        logger.info(f"Inference: {len(pending_ids)} pending, {cached_count} cached")
        
        results = []
        
        # 加载已缓存的结果
        for sample_id in all_sample_ids:
            if sample_id not in pending_ids:
                cached = self.cache_manager.get_inference_cache(sample_id)
                if cached:
                    results.append(cached)
        
        # 处理待推理的样本
        for i, sample in enumerate(samples):
            sample_id = self.get_sample_id(sample, i)
            if sample_id not in pending_ids:
                continue
            
            logger.info(f"[{i+1}/{len(samples)}] Processing {sample_id}")
            
            try:
                result = await self.run_inference_single(sample, model, **kwargs)
                result.sample_id = sample_id
                result.model_name = model_name
                
                # 保存到缓存
                if self.config.enable_cache:
                    self.cache_manager.save_inference_cache(result)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {sample_id}: {e}")
                error_result = InferenceResult(
                    sample_id=sample_id,
                    model_name=model_name,
                    input_data=sample,
                    dialogue=[],
                    status="failed",
                    error=str(e)
                )
                if self.config.enable_cache:
                    self.cache_manager.save_inference_cache(error_result)
                results.append(error_result)
        
        # 保存推理结果汇总
        self._save_inference_summary(results, model_name)
        
        return results
    
    async def run_evaluation(
        self,
        inference_results: Optional[List[InferenceResult]] = None,
        inference_dir: Optional[str] = None,
        judge_model: Optional[Any] = None,
        skip_cache: bool = False,
        **kwargs
    ) -> Tuple[List[EvaluationResult], Dict[str, Any]]:
        """
        执行评估阶段
        
        Args:
            inference_results: 推理结果列表（直接传入）
            inference_dir: 推理结果目录（从文件加载）
            judge_model: 评估模型（用于 LLM-as-Judge）
            skip_cache: 是否跳过缓存
        
        Returns:
            (评估结果列表, 汇总统计)
        """
        # 加载推理结果
        if inference_results is None:
            if inference_dir:
                inference_results = self._load_inference_from_dir(inference_dir)
            else:
                inference_results = self.cache_manager.get_all_inference_results()
        
        if not inference_results:
            raise ValueError("No inference results to evaluate")
        
        logger.info(f"Evaluating {len(inference_results)} samples")
        
        eval_results = []
        
        for i, inf_result in enumerate(inference_results):
            if inf_result.status == "failed":
                logger.warning(f"Skipping failed sample: {inf_result.sample_id}")
                continue
            
            # 检查缓存
            if not skip_cache and self.cache_manager.has_eval_cache(inf_result.sample_id):
                cached = self.cache_manager.get_eval_cache(inf_result.sample_id)
                if cached:
                    eval_results.append(cached)
                    continue
            
            logger.info(f"[{i+1}/{len(inference_results)}] Evaluating {inf_result.sample_id}")
            
            try:
                result = await self.evaluate_single(inf_result, judge_model, **kwargs)
                
                # 保存到缓存
                if self.config.enable_cache:
                    self.cache_manager.save_eval_cache(result)
                
                eval_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating {inf_result.sample_id}: {e}")
        
        # 计算汇总统计
        summary = self._compute_summary(eval_results)
        
        # 保存评估结果
        self._save_evaluation_results(eval_results, summary)
        
        return eval_results, summary
    
    async def run(
        self,
        model: Any,
        model_name: str,
        judge_model: Optional[Any] = None,
        mode: str = "full",  # "full", "inference", "evaluate"
        inference_dir: Optional[str] = None,
        limit: Optional[int] = None,
        skip_cache: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行完整流程
        
        Args:
            model: 推理模型
            model_name: 模型名称
            judge_model: 评估模型
            mode: 运行模式 ("full", "inference", "evaluate")
            inference_dir: 评估模式下的推理结果目录
            limit: 限制样本数量
            skip_cache: 是否跳过缓存
        
        Returns:
            运行结果
        """
        result = {
            "benchmark": self.name,
            "model": model_name,
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        }
        
        if mode in ["full", "inference"]:
            inference_results = await self.run_inference(
                model, model_name, limit, skip_cache, **kwargs
            )
            result["inference"] = {
                "total": len(inference_results),
                "completed": len([r for r in inference_results if r.status == "completed"]),
                "failed": len([r for r in inference_results if r.status == "failed"])
            }
        else:
            inference_results = None
        
        if mode in ["full", "evaluate"]:
            eval_results, summary = await self.run_evaluation(
                inference_results=inference_results,
                inference_dir=inference_dir,
                judge_model=judge_model,
                skip_cache=skip_cache,
                **kwargs
            )
            result["evaluation"] = summary
        
        return result
    
    def _save_inference_summary(self, results: List[InferenceResult], model_name: str):
        """保存推理结果汇总"""
        summary_file = self.output_dir / f"inference_summary_{model_name}.json"
        
        summary = {
            "benchmark": self.name,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "total": len(results),
            "completed": len([r for r in results if r.status == "completed"]),
            "failed": len([r for r in results if r.status == "failed"]),
            "samples": [
                {
                    "sample_id": r.sample_id,
                    "status": r.status,
                    "dialogue_turns": len(r.dialogue),
                    "error": r.error
                }
                for r in results
            ]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved inference summary to {summary_file}")
    
    def _load_inference_from_dir(self, inference_dir: str) -> List[InferenceResult]:
        """从目录加载推理结果"""
        results = []
        inference_path = Path(inference_dir)
        
        if not inference_path.exists():
            raise FileNotFoundError(f"Inference directory not found: {inference_dir}")
        
        # 加载索引文件
        index_file = inference_path / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            for sample_id, filename in index.items():
                cache_file = inference_path / filename
                if cache_file.exists():
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        results.append(InferenceResult(**data))
        else:
            # 直接遍历目录
            for json_file in inference_path.glob("*.json"):
                if json_file.name == "index.json":
                    continue
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "sample_id" in data:
                            results.append(InferenceResult(**data))
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded {len(results)} inference results from {inference_dir}")
        return results
    
    def _compute_summary(self, eval_results: List[EvaluationResult]) -> Dict[str, Any]:
        """计算评估结果汇总"""
        if not eval_results:
            return {"error": "No evaluation results"}
        
        # 收集所有维度的得分
        all_scores = {}
        for result in eval_results:
            for key, value in result.scores.items():
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(value)
        
        # 计算平均分
        summary = {
            "total_samples": len(eval_results),
            "average_scores": {
                key: sum(values) / len(values)
                for key, values in all_scores.items()
            },
            "score_details": {
                key: {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
                for key, values in all_scores.items()
            }
        }
        
        return summary
    
    def _save_evaluation_results(
        self,
        eval_results: List[EvaluationResult],
        summary: Dict[str, Any]
    ):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存汇总
        summary_file = self.output_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 保存详细结果
        details_file = self.output_dir / f"evaluation_details_{timestamp}.json"
        details = [asdict(r) for r in eval_results]
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved evaluation results to {self.output_dir}")
        logger.info(f"  - Summary: {summary_file}")
        logger.info(f"  - Details: {details_file}")

