#!/usr/bin/env python3
"""
CoSER GCA Evaluation Script (Multi-Agent Version)
CoSER GCA 评测运行脚本 (Multi-Agent 版本)

Uses the complete multi-agent implementation from benchmarks/multi_turn/coser:
使用 benchmarks/multi_turn/coser 的完整多 Agent 实现：
- Each character has independent dialogue history / 每个角色有独立的对话历史
- For self: keep system_thinking + role_thinking / 给自己：保留 system_thinking + role_thinking
- For others: remove system_thinking, keep role_thinking / 给别人：移除 system_thinking，保留 role_thinking
- For others (full clean): remove all inner thoughts / 给别人（完全清理）：移除所有内心想法

Usage / 使用方式:
    # Full evaluation / 完整评测
    python run_coser.py --actor j-xxx --max-rounds 20 --num-samples 1
    
    # NSP random mode (skip NSP model, randomly select next speaker)
    # NSP 随机模式（不调用 NSP 模型，直接随机选择下一个角色）
    python run_coser.py --actor j-xxx --nsp-mode random
    
    # Simulation only / 只运行推理
    python run_coser.py --actor j-xxx --simulation-only
    
    # Enable verbose logging / 启用详细日志
    FULL_LOG=1 python run_coser.py --actor j-xxx --verbose
"""

import os
import sys
import json
import yaml
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import ModelFactory
from benchmarks.multi_turn.coser.benchmark import (
    CoSERBenchmark,
    CoSERConfig,
    ENVIRONMENT,
    NSP
)

# 完整日志模式
FULL_LOG = os.environ.get('FULL_LOG', '0') == '1'


def get_default_judge_config(models_config_path: str = "configs/models.yaml") -> Dict:
    """从配置文件获取默认的 Judge 配置"""
    try:
        with open(models_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        judge_config = config.get('judge', {})
        coser_config = judge_config.get('coser', {})
        
        return {
            'default': judge_config.get('default', 'qwen3-235B'),
            'judge': coser_config.get('judge', judge_config.get('default', 'qwen3-235B')),
            'nsp': coser_config.get('nsp', judge_config.get('default', 'qwen3-235B')),
            'env': coser_config.get('env', judge_config.get('default', 'qwen3-235B')),
        }
    except Exception as e:
        print(f"⚠️ 无法读取 Judge 配置，使用默认值: {e}")
        return {
            'default': 'qwen3-235B',
            'judge': 'qwen3-235B',
            'nsp': 'qwen3-235B',
            'env': 'qwen3-235B',
        }


class CoSERRunner:
    """CoSER Evaluation Runner (Multi-Agent Version) / CoSER 评测运行器 (Multi-Agent 版本)"""
    
    def __init__(
        self,
        actor_model_name: str,
        judge_model_name: str = "qwen3-235B",
        env_model_name: str = None,
        nsp_model_name: str = None,
        nsp_mode: str = "model",
        models_config: str = "configs/models.yaml",
        data_path: str = None,
        roleplay_format: str = "her",
        max_rounds: int = 10,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        output_dir: str = "results/coser",
        verbose: bool = False,
        display_name: str = None,
        eval_remove_role_thinking: bool = False,  # 消融实验：评估时是否移除 role_thinking
        enable_cache: bool = True,  # 是否启用缓存
        skip_cache: bool = False  # 是否跳过已有缓存
    ):
        self.actor_model_name = actor_model_name
        self.judge_model_name = judge_model_name
        self.env_model_name = env_model_name or judge_model_name
        self.nsp_model_name = nsp_model_name or judge_model_name
        self.nsp_mode = nsp_mode
        self.models_config = models_config
        self.roleplay_format = roleplay_format
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.output_dir = output_dir
        self.verbose = verbose
        # 清理 display_name 中的 / 字符（避免路径问题）
        self.display_name = (display_name or actor_model_name).replace('/', '_')
        self.eval_remove_role_thinking = eval_remove_role_thinking
        self.enable_cache = enable_cache
        self.skip_cache = skip_cache
        
        # Cache 目录
        self.cache_dir = Path(output_dir) / "cache" / "inference"
        if enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"   📁 Cache目录: {self.cache_dir}")
        
        # 确定数据路径
        if data_path:
            self.data_path = data_path
        else:
            # 使用相对路径，基于脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(script_dir, "data/coser")
            if roleplay_format in ["her", "her_nosys", "her_with_systhink", "her_without_systhink", "qwen", "api"]:
                self.data_path = os.path.join(base_path, "test_set_her.json")
            else:
                self.data_path = os.path.join(base_path, "test_set.json")
        
        # 加载模型
        print(f"\n📦 加载模型...")
        self.actor_model = ModelFactory.get_model(models_config, actor_model_name)
        self.judge_model = ModelFactory.get_model(models_config, judge_model_name)
        
        if self.env_model_name != judge_model_name:
            self.env_model = ModelFactory.get_model(models_config, self.env_model_name)
        else:
            self.env_model = self.judge_model
            
        if self.nsp_model_name != judge_model_name:
            self.nsp_model = ModelFactory.get_model(models_config, self.nsp_model_name)
        else:
            self.nsp_model = self.judge_model
        
        print(f"   ✅ Actor: {actor_model_name}")
        print(f"   ✅ Judge: {judge_model_name}")
        print(f"   ✅ Env: {self.env_model_name}")
        print(f"   ✅ NSP: {self.nsp_model_name} (mode: {nsp_mode})")
        
        # 创建 CoSER 配置
        self.config = CoSERConfig(
            name="coser",
            data_path=self.data_path,
            output_dir=output_dir,
            max_tokens=max_tokens,
            temperature=temperature,
            actor_model=actor_model_name,
            env_model=self.env_model_name,
            nsp_model=self.nsp_model_name,
            judge_model=judge_model_name,
            max_rounds=max_rounds,
            nsp_mode=nsp_mode,
            model_type=roleplay_format,
            eval_remove_role_thinking=eval_remove_role_thinking
        )
        
        # 创建 Benchmark
        self.benchmark = CoSERBenchmark(self.config)
        print(f"   ✅ 数据: {self.data_path} ({len(self.benchmark.data)} 条)")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_cache_path(self, sample_id: str) -> Path:
        """获取缓存文件路径"""
        # 清理 sample_id 中的非法字符
        safe_id = sample_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        return self.cache_dir / f"{safe_id}.json"
    
    def _load_cache(self, sample_id: str) -> Optional[Dict]:
        """从缓存加载结果"""
        if not self.enable_cache or self.skip_cache:
            return None
        cache_path = self._get_cache_path(sample_id)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"   ⚠️ 缓存加载失败 {sample_id}: {e}")
        return None
    
    def _save_cache(self, sample_id: str, result: Dict):
        """保存结果到缓存"""
        if not self.enable_cache:
            return
        cache_path = self._get_cache_path(sample_id)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
        except Exception as e:
            print(f"   ⚠️ 缓存保存失败 {sample_id}: {e}")
    
    async def run_simulation(self, num_samples: int = None, sample_offset: int = 0, workers: int = 1) -> List[Dict]:
        """运行推理阶段（支持并发和缓存）"""
        print(f"\n🎭 开始 Simulation (推理)...")
        print(f"   最大轮数: {self.max_rounds}")
        print(f"   NSP 模式: {self.nsp_mode}")
        print(f"   并发数: {workers}")
        print(f"   缓存: {'启用' if self.enable_cache else '禁用'}")
        if sample_offset > 0:
            print(f"   样本偏移: {sample_offset}")
        
        # 支持 sample_offset 分批
        all_data = self.benchmark.data
        if sample_offset > 0:
            all_data = all_data[sample_offset:]
        data = all_data[:num_samples] if num_samples else all_data
        
        print(f"   实际样本范围: {sample_offset} - {sample_offset + len(data) - 1} (共 {len(data)} 条)")
        
        # 检查缓存，跳过已完成的样本
        cached_count = 0
        pending_indices = []
        results = [None] * len(data)
        
        for idx, sample in enumerate(data):
            sample_id = self.benchmark.get_sample_id(sample, idx + sample_offset)
            cached = self._load_cache(sample_id)
            if cached and cached.get('status') == 'completed':
                results[idx] = cached
                cached_count += 1
            else:
                pending_indices.append(idx)
        
        if cached_count > 0:
            print(f"   📦 从缓存恢复: {cached_count} 条")
        print(f"   ⏳ 待处理: {len(pending_indices)} 条")
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(workers)
        completed = [cached_count]  # 使用列表来在闭包中修改
        
        async def process_sample(idx: int, sample: Dict):
            async with semaphore:
                sample_id = self.benchmark.get_sample_id(sample, idx + sample_offset)
                
                try:
                    result = await self.benchmark.run_inference_single(
                        sample=sample,
                        model=self.actor_model,
                        env_model=self.env_model,
                        nsp_model=self.nsp_model
                    )
                    result_dict = {
                        "sample_id": sample_id,
                        "sample": sample,
                        "dialogue": result.dialogue,
                        "metadata": result.metadata,
                        "status": "completed"
                    }
                    results[idx] = result_dict
                    
                    # 保存缓存
                    self._save_cache(sample_id, result_dict)
                    
                    completed[0] += 1
                    print(f"   ✅ [{completed[0]}/{len(data)}] {sample_id}: {len(result.dialogue)} 轮对话")
                except Exception as e:
                    print(f"   ❌ [{idx+1}/{len(data)}] {sample_id} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    results[idx] = {
                        "sample_id": sample_id,
                        "sample": sample,
                        "error": str(e),
                        "status": "failed"
                    }
                    completed[0] += 1
        
        # 只处理未缓存的样本
        tasks = [process_sample(idx, data[idx]) for idx in pending_indices]
        await asyncio.gather(*tasks)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_path = f"{self.output_dir}/simulation_{self.display_name}_{timestamp}.json"
        
        output = {
            "config": {
                "actor_model": self.actor_model_name,
                "roleplay_format": self.roleplay_format,
                "max_rounds": self.max_rounds,
                "nsp_mode": self.nsp_mode,
                "timestamp": timestamp
            },
            "results": results
        }
        
        with open(sim_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Simulation 完成! 保存到: {sim_path}")
        return results
    
    async def run_evaluation(self, simulation_results: List[Dict] = None, simulation_path: str = None) -> Dict:
        """运行评估阶段"""
        print(f"\n📊 开始 Evaluation (评估)...")
        
        # 加载推理结果
        if simulation_results is None:
            if simulation_path is None:
                raise ValueError("必须提供 simulation_results 或 simulation_path")
            with open(simulation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                simulation_results = data.get('results', data)
        
        # 评估每个样本
        evaluations = []
        all_scores = []
        
        for idx, result in enumerate(simulation_results):
            if result.get('status') == 'failed':
                continue
            
            sample_id = result.get('sample_id', f'sample_{idx}')
            print(f"   评估: {sample_id}")
            
            try:
                # 调用 benchmark 的评估方法
                from benchmarks.multi_turn.coser.benchmark import InferenceResult
                
                # 构建 InferenceResult 对象
                inference_result = InferenceResult(
                    sample_id=sample_id,
                    model_name=self.actor_model_name,
                    input_data=result.get('sample', {}),
                    dialogue=result.get('dialogue', []),
                    metadata=result.get('metadata', {}),
                    status='completed'
                )
                
                # 调用评估
                eval_result = await self.benchmark.evaluate_single(
                    inference_result=inference_result,
                    judge_model=self.judge_model
                )
                
                evaluations.append({
                    "sample_id": sample_id,
                    "scores": eval_result.scores,
                    "details": eval_result.details
                })
                all_scores.append(eval_result.scores)
                
                print(f"   ✅ {sample_id}: avg={eval_result.scores.get('avg', 0):.1f}")
                
            except Exception as e:
                import traceback
                print(f"   ❌ 评估失败: {e}")
                traceback.print_exc()
                evaluations.append({
                    "sample_id": sample_id,
                    "scores": {},
                    "error": str(e)
                })
        
        # 计算平均分数
        final_scores = {}
        if all_scores:
            for key in all_scores[0].keys():
                scores = [s.get(key, 0) for s in all_scores if key in s]
                if scores:
                    final_scores[key] = sum(scores) / len(scores)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_path = f"{self.output_dir}/evaluation_{self.display_name}_{timestamp}.json"
        
        output = {
            "config": {
                "actor_model": self.actor_model_name,
                "judge_model": self.judge_model_name,
                "timestamp": timestamp
            },
            "scores": final_scores,
            "evaluations": evaluations
        }
        
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        # 打印汇总分数
        if final_scores:
            print(f"\n📊 评估汇总:")
            for dim, score in final_scores.items():
                print(f"   {dim}: {score:.2f}")
        
        print(f"\n✅ Evaluation 完成! 保存到: {eval_path}")
        return final_scores
    
    async def run(
        self,
        num_samples: int = None,
        sample_offset: int = 0,
        workers: int = 1,
        simulation_only: bool = False,
        evaluation_only: bool = False,
        simulation_file: str = None
    ):
        """运行完整评测流程"""
        if evaluation_only:
            if simulation_file:
                return await self.run_evaluation(simulation_path=simulation_file)
            else:
                raise ValueError("evaluation-only 模式需要指定 --simulation-file")
        
        # 运行推理
        results = await self.run_simulation(num_samples, sample_offset=sample_offset, workers=workers)
        
        if simulation_only:
            return results
        
        # 运行评估
        return await self.run_evaluation(results)


def main():
    parser = argparse.ArgumentParser(description='CoSER GCA 评测 (Multi-Agent 版本)')
    
    # 模型配置
    parser.add_argument('--actor', type=str, required=True, help='待评测的角色扮演模型')
    parser.add_argument('--display-name', type=str, default=None, help='模型显示名称')
    parser.add_argument('--judge', type=str, default=None, help='评估用的 Judge 模型')
    parser.add_argument('--env', type=str, default=None, help='环境描述模型')
    parser.add_argument('--nsp', type=str, default=None, help='下一说话人预测模型')
    parser.add_argument('--nsp-mode', type=str, default='model', choices=['model', 'random'],
                        help='NSP模式: model=使用模型预测, random=随机选择角色')
    
    # 数据配置
    parser.add_argument('--data', type=str, default=None, help='测试数据路径')
    parser.add_argument('--format', type=str, default='her',
                        choices=['her', 'her_nosys', 'her_with_systhink', 'her_without_systhink', 
                                 'qwen', 'coser', 'api', 'llama3'],
                        help='角色扮演格式')
    parser.add_argument('--max-rounds', type=int, default=10, help='最大对话轮数')
    parser.add_argument('--max-tokens', type=int, default=4096, help='单次生成最大 tokens')
    parser.add_argument('--num-samples', type=int, default=None, help='测试样本数')
    parser.add_argument('--sample-offset', type=int, default=0, help='跳过前N条样本 (用于分批)')
    
    # 运行模式
    parser.add_argument('--simulation-only', action='store_true', help='只运行推理')
    parser.add_argument('--evaluation-only', action='store_true', help='只运行评估')
    parser.add_argument('--simulation-file', type=str, default=None, help='指定 simulation 文件路径')
    
    # 消融实验
    parser.add_argument('--eval-remove-role-thinking', action='store_true', 
                        help='消融实验：评估时移除 role_thinking (默认保留)')
    
    # 输出配置
    parser.add_argument('--output', type=str, default='results/coser', help='输出目录')
    parser.add_argument('--log-dir', type=str, default=None, help='日志目录 (默认同 output)')
    parser.add_argument('--models-config', type=str, default='configs/models.yaml', help='模型配置文件')
    parser.add_argument('--verbose', '-v', action='store_true', help='输出详细日志')
    parser.add_argument('--full-log', action='store_true', help='输出完整日志')
    parser.add_argument('--workers', type=int, default=100, help='推理并发数 (默认100)')
    
    # Cache 配置
    parser.add_argument('--enable-cache', action='store_true', default=True, help='启用缓存 (默认启用)')
    parser.add_argument('--no-cache', action='store_true', help='禁用缓存')
    parser.add_argument('--skip-cache', action='store_true', help='跳过已有缓存重新运行')
    
    args = parser.parse_args()
    
    # 获取默认 Judge 配置
    default_judge = get_default_judge_config(args.models_config)
    
    judge_model = args.judge or default_judge['judge']
    env_model = args.env or default_judge['env']
    nsp_model = args.nsp or default_judge['nsp']
    
    print(f"\n🔧 配置:")
    print(f"   Actor: {args.actor}")
    print(f"   Judge: {judge_model}")
    print(f"   NSP:   {nsp_model} (mode: {args.nsp_mode})")
    print(f"   Env:   {env_model}")
    print(f"   Format: {args.format}")
    print(f"   Max Rounds: {args.max_rounds}")
    
    # 设置 FULL_LOG 环境变量
    if args.verbose or args.full_log:
        os.environ['FULL_LOG'] = '1'
    
    # Cache 配置
    enable_cache = args.enable_cache and not args.no_cache
    
    runner = CoSERRunner(
        actor_model_name=args.actor,
        judge_model_name=judge_model,
        env_model_name=env_model,
        nsp_model_name=nsp_model,
        nsp_mode=args.nsp_mode,
        models_config=args.models_config,
        data_path=args.data,
        roleplay_format=args.format,
        max_rounds=args.max_rounds,
        max_tokens=args.max_tokens,
        output_dir=args.output,
        verbose=args.verbose,
        display_name=args.display_name,
        eval_remove_role_thinking=args.eval_remove_role_thinking,
        enable_cache=enable_cache,
        skip_cache=args.skip_cache
    )
    
    # 运行
    asyncio.run(runner.run(
        num_samples=args.num_samples,
        sample_offset=args.sample_offset,
        workers=args.workers,
        simulation_only=args.simulation_only,
        evaluation_only=args.evaluation_only,
        simulation_file=args.simulation_file
    ))


if __name__ == "__main__":
    main()

