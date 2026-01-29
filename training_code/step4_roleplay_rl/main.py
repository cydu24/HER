import json
import re
from typing import Dict, Any, List, Optional

def extract_better_response_from_roleplay(response_content: str) -> Optional[str]:
    """从roleplay RL数据的response字段中提取better_response"""
    if not response_content or response_content.strip() == "":
        return None
        
    try:
        # 首先尝试正则表达式（更可靠）
        pattern = r'"better_response":\s*"(cand_[12])"'
        match = re.search(pattern, response_content)
        if match:
            return match.group(1)
        
        # 如果正则没找到，尝试解析JSON
        json_content = response_content.strip()
        
        # 移除```json和```包装
        if json_content.startswith('```json'):
            json_content = json_content[7:]  # 移除```json
        if json_content.endswith('```'):
            json_content = json_content[:-3]  # 移除```
        json_content = json_content.strip()
        
        # 如果内容太短，跳过
        if len(json_content) < 20:
            return None
        
        # 尝试修复常见的JSON错误
        json_content = fix_common_json_errors(json_content)
        
        # 解析JSON
        data = json.loads(json_content)
        
        # 从result数组中提取better_response
        if 'result' in data and isinstance(data['result'], list) and len(data['result']) > 0:
            result_item = data['result'][0]
            if 'better_response' in result_item:
                return result_item['better_response']
        
        return None
        
    except Exception as e:
        # 只在调试模式下打印错误
        # print(f"解析roleplay response出错: {e}")
        return None

def fix_common_json_errors(json_str: str) -> str:
    """修复常见的JSON格式错误"""
    # 修复未闭合的字符串（在行尾添加引号）
    lines = json_str.split('\n')
    fixed_lines = []
    
    for line in lines:
        # 如果行中有未配对的引号，尝试修复
        quote_count = line.count('"')
        if quote_count % 2 == 1:  # 奇数个引号，可能未闭合
            # 检查是否是字符串值的行
            if ':' in line and not line.strip().endswith('"') and not line.strip().endswith(','):
                line = line.rstrip() + '"'
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def extract_candidates_from_raw_record(raw_record: Dict[str, Any]) -> Dict[str, str]:
    """从raw_record中提取candidate_1和candidate_2"""
    candidates = {}
    if 'candidate_1' in raw_record:
        candidates['cand_1'] = raw_record['candidate_1']
    if 'candidate_2' in raw_record:
        candidates['cand_2'] = raw_record['candidate_2']
    return candidates

# 已移除标签修复和字符串拼接函数
# 现在使用标准的多轮对话格式

def build_prompt_from_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    从messages构建prompt，排除最后一个assistant消息
    保持标准的多轮对话格式，不拼接成字符串
    """
    if not messages:
        return []
    
    # 排除最后一个assistant消息
    if messages[-1]['role'] == 'assistant':
        messages = messages[:-1]
    
    # 返回标准的多轮对话格式
    return messages

def convert_to_rl_format(raw_record: Dict[str, Any], better_response: str) -> Dict[str, Any]:
    """将数据转换为RL训练格式"""
    # 构建prompt
    messages = raw_record.get('messages', [])
    prompt = build_prompt_from_messages(messages)
    
    # 提取candidates
    candidates = extract_candidates_from_raw_record(raw_record)
    
    # 根据better_response选择答案
    answer = candidates.get(better_response, "")
    
    # 构建最终格式
    result = {
        "data_source": "roleplay_rl",
        "prompt": prompt,
        "reward_model": {
            "answer": answer,
            "solution": json.dumps({"input": json.dumps(prompt), "ref_answer": answer,
            "golden_type": "3"}),
            "style": "rule"
        },
        "extra_info": {
            "raw_record": raw_record,
            "better_response": better_response,
            "candidates": candidates,
            "golden_type": "3",
            "index": 0  # 添加必需的index字段
        }
    }
    
    return result

def process_roleplay_rl_data(input_file: str, output_file: str):
    """处理roleplay RL数据的主函数"""
    processed_count = 0
    error_count = 0
    empty_response_count = 0
    no_better_response_count = 0
    json_error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                line = line.strip()
                if not line:
                    continue
                
                # 解析输入行
                data = json.loads(line)
                
                # 检查数据结构，实际数据可能在不同字段中
                response_text = None
                raw_record = None
                
                # 尝试不同的数据结构
                if 'response' in data:
                    response_text = data['response']
                elif 'data' in data and len(data['data']) > 1:
                    # 可能response在data数组的第二个元素的text字段中
                    if 'text' in data['data'][1]:
                        response_text = data['data'][1]['text']
                
                # 如果 data[1]['text'] 为空，尝试从 model_request_output 提取
                if not response_text or response_text.strip() == "":
                    if 'model_request_output' in data:
                        mro = data['model_request_output']
                        output = mro.get('output', [])
                        for item in output:
                            if item.get('type') == 'message':
                                content = item.get('content', [])
                                for c in content:
                                    if isinstance(c, dict) and c.get('type') in ['text', 'output_text']:
                                        response_text = c.get('text', '')
                                        break
                                if response_text:
                                    break
                
                if 'raw_record' in data:
                    raw_record = data['raw_record']
                elif 'raw_record' in data.get('data', [{}])[0] if 'data' in data else {}:
                    raw_record = data['data'][0]['raw_record']
                
                # 如果找不到raw_record，跳过
                if not raw_record:
                    error_count += 1
                    continue
                
                # 检查response是否为空
                if not response_text or response_text.strip() == "":
                    empty_response_count += 1
                    continue
                
                # 提取better_response
                better_response = extract_better_response_from_roleplay(response_text)
                if not better_response:
                    no_better_response_count += 1
                    continue
                
                # 转换格式
                rl_data = convert_to_rl_format(raw_record, better_response)
                
                # 写入输出文件
                outfile.write(json.dumps(rl_data, ensure_ascii=False) + '\n')
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"已处理 {processed_count} 条数据")
                
                # 显示前几条成功处理的数据
                if processed_count <= 3:
                    print(f"第{line_num}行成功处理，better_response: {better_response}")
                
            except json.JSONDecodeError as e:
                json_error_count += 1
                continue
            except Exception as e:
                error_count += 1
                continue
    
    total_errors = error_count + empty_response_count + no_better_response_count + json_error_count
    print(f"\n处理完成！")
    print(f"✅ 成功处理: {processed_count} 条")
    print(f"❌ 总错误: {total_errors} 条")
    print(f"  - JSON解析错误: {json_error_count} 条")
    print(f"  - 空response: {empty_response_count} 条") 
    print(f"  - 无better_response: {no_better_response_count} 条")
    print(f"  - 其他错误: {error_count} 条")
    print(f"📊 成功率: {processed_count/(processed_count + total_errors)*100:.1f}%")

# 测试函数
def test_with_sample_data():
    """使用示例数据进行测试"""
    print("=== 测试多轮对话格式 ===")
    
    sample_response = '''```json
{
  "result": [
    {
      "cand_1": "Harv: <role_action>I soften my gaze and nod slowly, offering a silent acknowledgment of his vulnerability. I remain still, my hands folded calmly on the table, giving him the space he needs.</role_action> <role_thinking>That was it. The dam broke. Now he just needs the final piece. The last, most important word.</role_thinking",
      "cand_2": "Harv: <role_action>I raise a hand slowly, a small, placating gesture to hold the room in check, my voice soft but firm with authority.</role_action> <role_thinking>That voice. It's the sound of pure need. I must protect this moment from interruption.</role_thinking> Let's give Kevin a moment, gentlemen.",
      "better_response": "cand_2"
    }
  ]
}
```'''
    
    sample_raw_record = {
        "trace_id": "3bc9c732-7757-4342-ba69-9d31483ed6bd/0050/Harv_turn_2",
        "messages": [
            {
                "role": "system", 
                "content": "You are Harv, a skilled group therapy facilitator. You maintain calm authority while creating safe spaces for emotional breakthroughs. The group is in session, and Kevin is having a vulnerable moment."
            },
            {
                "role": "user", 
                "content": "Kevin: *His voice breaks, tears streaming* I... I can't... *He looks around the circle desperately* I need... I need..."
            },
            {
                "role": "assistant", 
                "content": "Harv: *I lean forward slightly, my voice gentle but steady* Take your time, Kevin. We're all here with you. What is it you need right now?"
            },
            {
                "role": "user",
                "content": "Kevin: *Sobbing openly now, his words barely audible* Help... I need help... *The room falls silent, everyone holding their breath*"
            },
            {
                "role": "assistant",
                "content": "Harv: *I nod slowly, acknowledging his courage* That took incredible strength to say, Kevin. You've just taken the most important step."
            }
        ],
        "dataset_type": "rl",
        "candidate_1": "Harv: <role_action>I soften my gaze and nod slowly, offering a silent acknowledgment of his vulnerability. I remain still, my hands folded calmly on the table, giving him the space he needs.</role_action> <role_thinking>That was it. The dam broke. Now he just needs the final piece. The last, most important word.</role_thinking",
        "candidate_2": "Harv: <role_action>I raise a hand slowly, a small, placating gesture to hold the room in check, my voice soft but firm with authority.</role_action> <role_thinking>That voice. It's the sound of pure need. I must protect this moment from interruption.</role_thinking> Let's give Kevin a moment, gentlemen."
    }
    
    # 测试提取better_response
    better_response = extract_better_response_from_roleplay(sample_response)
    print(f"提取的better_response: {better_response}")
    
    # 详细展示prompt构建过程
    print("\n=== 原始messages ===")
    for i, msg in enumerate(sample_raw_record['messages']):
        print(f"{i}: {msg['role']} -> {msg['content']}")
    
    # 测试prompt构建
    prompt = build_prompt_from_messages(sample_raw_record['messages'])
    print(f"\n=== 构建的prompt (排除最后一个assistant) ===")
    for i, msg in enumerate(prompt):
        print(f"{i}: {msg['role']} -> {msg['content']}")
    
    # 测试转换格式
    if better_response:
        rl_data = convert_to_rl_format(sample_raw_record, better_response)
        print(f"\n=== 最终选择的答案 (better_response={better_response}) ===")
        print(f"答案内容: {rl_data['reward_model']['answer']}")
        
        print("\n=== 完整的RL格式 ===")
        print(json.dumps(rl_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # 运行测试
    print("运行测试...")
    test_with_sample_data()
    
    print("\n" + "="*50)
    print("开始处理实际数据...")
    
    # 处理实际文件
    input_file = "/path/to/data/example"
    output_file = "/path/to/data/example"
    
    process_roleplay_rl_data(input_file, output_file)

    '''
    
    /path/to/data/example
├── roleplay_rl.jsonl      ← 用于 roleplay RL
├── reward_rl.jsonl        ← 用于 reward RL
└── rl_split_stats.json    ← 统计信息
    '''