"""
Reward Model Evaluation Script / 奖励模型评估脚本

Process model inference results and extract evaluation scores.
处理模型推理结果并提取评估分数。
"""

import os
import json
import glob
from tqdm import tqdm
from typing import List, Dict, Any
import numpy as np
import re
import random

def process_file(file_path: str, model_name: str) -> List[Dict[str, Any]]:
    """Process single jsonl file / 处理单个jsonl文件"""
    import json  # 移到函数开头，避免作用域问题
    from tqdm import tqdm
    import os
    
    processed_data = []
    token_info = []
    failed_items = []
    skipped_items = []  # 新增：保存跳过的数据
    
    # 获取文件名用于进度条描述
    file_name = os.path.basename(file_path)
    
    # 首先计算总行数
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 使用tqdm显示进度
        pbar = tqdm(enumerate(f, 1), total=total_lines, desc=f"处理 {file_name}", leave=False)
        for line_num, line in pbar:
                # 更新进度条统计信息
                pbar.set_postfix({
                    '成功': len(processed_data), 
                    '跳过': len(skipped_items),
                    '失败': len(failed_items)
                })
          
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                data = json.loads(line)
                # 提取response和think
                think = ""
                response = ""

                if model_name == 'claude_3_7_sonnet_20250219' or model_name == 'claude_opus_4_20250514' or model_name == 'claude_sonnet_4_20250514':
                    model_req = json.loads(data['vulcan_output']['model_req'])
                    for content in model_req['content']:
                        if content['type'] == 'thinking':
                            data['think'] = content['thinking'].strip()
                        if content['type'] == 'text':
                            data['response'] = content['text'].strip()
                elif model_name == "model":
                    try:
                        # 检查status是否为completed（支持多种status位置）
                        status = None
                        if 'model_request_output' in data['vulcan_output']:
                            status = data['vulcan_output']['model_request_output'].get('status', '')
                        elif 'status' in data['vulcan_output']:
                            status = data['vulcan_output'].get('status', '')
                        
                        # 跳过非completed状态（包括incomplete、failed等）
                        if status and status.lower() not in ['completed', 'complete']:
                            skipped_items.append({
                                'line_num': line_num,
                                'reason': f'status不是completed: {status}',
                                'data': data
                            })
                            continue
                        
                        # 检查output的长度
                        output_list = data['vulcan_output'].get('output', [])
                        if not output_list or len(output_list) < 2:
                            skipped_items.append({
                                'line_num': line_num,
                                'reason': f'output只有{len(output_list)}个元素，需要至少2个',
                                'data': data
                            })
                            continue
                            
                        # 获取reasoning数据
                        reasoning_data = output_list[0].get("summary", [])
                        
                        # 检查summary是否为空
                        if not reasoning_data:
                            # 静默跳过，只记录到skipped_items
                            # print(f"\n⚠️ summary为空，跳过...")
                            skipped_items.append({
                                'line_num': line_num,
                                'reason': 'summary为空',
                                'data': data
                            })
                            continue
                        
                        # 定义有序的连接词结构
                        ordinal_connectors = {
                            0: [
                                 "First of all,", "To begin with,", "Initially,", 
                                "Let me start by noting that", "To start,", "At the outset,"
                            ],
                            1: [
                                 "Next,", "Then,", "Following this,", 
                                "Moving on to the second point,", "Secondly,", "After that,"
                            ],
                            2: [
                                 "Furthermore,", "Additionally,", "Thirdly,",
                                "Considering another aspect,", "Also worth noting,", "In addition to this,"
                            ],
                            3: [
                                 "Moreover,", "Building on this,", "Fourthly,",
                                "What's more,", "Equally important,", "Along these lines,"
                            ],
                            4: [
                                "Subsequently,", "In addition,", "Fifthly,",
                                "Further to this point,", "Another key consideration,", "Beyond that,"
                            ],
                            "middle": [
                                "Continuing with the analysis,", "Moving forward,", "Also,", 
                                "Let me also consider that", "It's also worth examining how", "Delving deeper,",
                                "On a related note,", "In a similar manner,", "In the same vein,", 
                                "Expanding on this,", "To elaborate further,", "Looking more closely,"
                            ],
                            "penultimate": [
                                "Before concluding,", "As a final consideration,", "Lastly,", 
                                "To wrap up the analysis,", "One final point to consider,",
                                "In the final analysis,", "As a closing thought,", "To summarize the key findings,"
                            ],
                        }
                        
                        # 开始构建思考文本
                        think_parts = []
                        
                        # 多样化的开场白
                        openings = [
                            "Alright, let me analyze this task now.",
                            "Let me carefully examine this evaluation task.",
                            "I need to thoroughly analyze these candidates.",
                            "Let me break down this comparison systematically.",
                            "Time to evaluate these responses comprehensively.",
                            "Let me approach this analysis methodically.",
                            "I'll now conduct a detailed assessment of these candidates.",
                            "Let me work through this evaluation step by step."
                        ]
                        think_parts.append(random.choice(openings) + "\n\n")
                        
                        # 处理每个summary项
                        total_items = len(reasoning_data)
                        for i, item in enumerate(reasoning_data):
                            # 提取文本内容（跳过标题行）
                            text_content = "\n\n".join(item['text'].split("\n\n")[1:])
                            
                            # 选择合适的连接词
                            if i == 0:
                                # 第一项
                                connector = random.choice(ordinal_connectors[0])
                                think_parts.append(f"{connector} {text_content}")
                            elif i == total_items - 1 and total_items > 2:
                                # 最后一项（如果超过2项）
                                connector = random.choice(ordinal_connectors["penultimate"])
                                think_parts.append(f"\n\n{connector} {text_content}")
                            elif i <= 4 and i in ordinal_connectors:
                                # 第2-5项使用对应的序数连接词
                                connector = random.choice(ordinal_connectors[i])
                                think_parts.append(f"\n\n{connector} {text_content}")
                            else:
                                # 中间项使用通用连接词
                                connector = random.choice(ordinal_connectors["middle"])
                                think_parts.append(f"\n\n{connector} {text_content}")
                        
                        # 添加结束语
                        final_phrases = [
                            "\n\nFinally, after analyzing all aspects, I've reached my conclusion.",
                            "\n\nFinally, having considered all dimensions, my evaluation is complete.",
                            "\n\nFinally, based on this comprehensive analysis, I can now determine the outcome.",
                            "\n\nFinally, with all factors weighed, I've arrived at my decision.",
                            "\n\nIn conclusion, after careful consideration of all the evidence, I have my answer.",
                            "\n\nUltimately, taking everything into account, the better response is clear.",
                            "\n\nTo conclude, having examined all the relevant principles, I can make my judgment.",
                            "\n\nAll things considered, I've arrived at a definitive evaluation.",
                            "\n\nHaving weighed all the factors, my assessment is complete.",
                            "\n\nBased on this thorough analysis, I can now make an informed decision."
                        ]
                        think_parts.append(random.choice(final_phrases))
                        
                        # 多样化的输出转换语句
                        output_transitions = [
                            "\n\nOkay, now let me generate the output.",
                            "\n\nNow I'll structure this evaluation in the required JSON format.",
                            "\n\nLet me now format this analysis into the proper output structure.",
                            "\n\nTime to compile this assessment into the final response.",
                            "\n\nI'll now generate the structured evaluation output.",
                            "\n\nLet me produce the formal evaluation result.",
                            "\n\nNow to create the detailed JSON output for this evaluation.",
                            "\n\nWith the analysis complete, I'll generate the formatted response."
                        ]
                        think_parts.append(random.choice(output_transitions))
                        
                        # 拼接所有部分
                        data['think'] = "".join(think_parts)
                        
                        # 尝试获取response（可能在output[1]或output[2]中）
                        response_found = False
                        
                        # 先尝试output[2]（3个元素的情况）
                        if len(output_list) > 2 and 'content' in output_list[2]:
                            try:
                                data['response'] = output_list[2]['content'][0]['text']
                                response_found = True
                            except (KeyError, IndexError):
                                pass
                        
                        # 如果没找到，尝试output[1]（2个元素的情况）
                        if not response_found and len(output_list) > 1 and 'content' in output_list[1]:
                            try:
                                data['response'] = output_list[1]['content'][0]['text']
                                response_found = True
                            except (KeyError, IndexError):
                                pass
                        
                        # 如果都没找到，跳过
                        if not response_found:
                            skipped_items.append({
                                'line_num': line_num,
                                'reason': '在output中没有找到content或text',
                                'data': data
                            })
                            continue
                        
                    except (IndexError, KeyError) as e:
                        print("\n" + "="*80)
                        print(f"❌ 错误类型: {type(e).__name__}")
                        print(f"❌ 错误信息: {str(e)}")
                        print("="*80)
                        print("\n📊 输出结构分析:")
                        print(f"output列表长度: {len(data['vulcan_output']['output'])}")
                        
                        # 打印每个output元素的结构
                        for idx, output_item in enumerate(data['vulcan_output']['output']):
                            print(f"\n[Output {idx}] 的键:")
                            print(f"  - Keys: {list(output_item.keys())}")
                            
                            # 如果有content键，打印其结构
                            if 'content' in output_item:
                                if isinstance(output_item['content'], list):
                                    print(f"  - content是列表，长度: {len(output_item['content'])}")
                                    if len(output_item['content']) > 0:
                                        print(f"  - content[0]的类型: {type(output_item['content'][0])}")
                                        if isinstance(output_item['content'][0], dict):
                                            print(f"  - content[0]的键: {list(output_item['content'][0].keys())}")
                                else:
                                    print(f"  - content类型: {type(output_item['content'])}")
                        
                        # 打印完整的output结构（限制长度）
                        print("\n📝 完整output结构（前500字符）:")
                        output_str = json.dumps(data['vulcan_output']['output'], indent=2, ensure_ascii=False)
                        print(output_str[:500] + "..." if len(output_str) > 500 else output_str)
                        
                        print("\n" + "="*80)
                        
                        # 跳过这条数据
                        print("⚠️ 跳过这条数据，继续处理...")
                        skipped_items.append({
                            'line_num': line_num,
                            'reason': f'处理错误: {type(e).__name__}: {str(e)}',
                            'data': data
                        })
                        continue
                processed_data.append(data)
                # 安全删除vulcan_output字段
                if 'vulcan_output' in data:
                    del data['vulcan_output']
               
                
                
    
    return processed_data, token_info, failed_items, skipped_items
'''

Processing gemini_2_5_pro_preview_05_06
model_req: dict_keys(['candidates', 'usageMetadata', 'modelVersion', 'createTime', 'responseId'])
Processing gemini_2_5_pro_preview_05_06: 100%|██████████████████████████████████████████████████| 1/1 [00:00<00:00, 1467.05it/s]
Processing gemini_2_5_pro_preview_06_05
model_req: dict_keys(['candidates', 'usageMetadata', 'modelVersion', 'createTime', 'responseId'])
{'candidates': [{'content': {'role': 'model', 'parts': [{'text' {'text': 
'''  

# Ray处理函数已移除，现在使用串行处理


def extract_field(text: str, start_pattern: str, end_pattern: str) -> str:
    """提取字段内容，使用下一个字段作为结束标记"""
    try:
        if not text or not isinstance(text, str):
            return ""
            
        start = re.search(start_pattern, text, re.DOTALL)
        if not start:
            return ""
        start_pos = start.end()
        
        # 安全检查：确保start_pos不超出文本长度
        if start_pos >= len(text):
            return ""
        
        end = re.search(end_pattern, text[start_pos:], re.DOTALL)
        if not end:
            return text[start_pos:].strip()
        
        # 安全检查：确保end.start()不超出剩余文本长度
        if end.start() > len(text[start_pos:]):
            return text[start_pos:].strip()
            
        content = text[start_pos:start_pos + end.start()].strip()
        return content
    except Exception as e:
        print(f"Error extracting field: {e}")
        return ""

def parse_characters(characters_text: str) -> list:
    """解析角色数组，支持多种格式"""
    try:
        if not characters_text.strip():
            return []
        
        # 方法1: 尝试直接解析JSON
        try:
            # 尝试修复常见的JSON格式问题
            cleaned_text = characters_text.strip()
            if not cleaned_text.startswith('['):
                cleaned_text = '[' + cleaned_text
            if not cleaned_text.endswith(']'):
                cleaned_text = cleaned_text + ']'
            
            characters = json.loads(cleaned_text)
            if isinstance(characters, list):
                return characters
        except json.JSONDecodeError:
            pass
        
        # 方法2: 使用正则表达式逐个匹配角色
        character_patterns = [
            # 标准格式 - 支持多行内容
            r'\{\s*"name"\s*:\s*"([^"]*)",\s*"motivation"\s*:\s*"(.*?)",\s*"background"\s*:\s*"(.*?)"\s*\}',
            # 容错格式 - 支持单引号
            r"\{\s*'name'\s*:\s*'([^']*)',\s*'motivation'\s*:\s*'(.*?)',\s*'background'\s*:\s*'(.*?)'\s*\}",
            # 简化格式
            r'\{\s*"name"\s*:\s*"([^"]*)",\s*"motivation"\s*:\s*"([^"]*)",\s*"background"\s*:\s*"([^"]*)"\s*\}',
        ]
        
        characters_list = []
        for pattern in character_patterns:
            matches = re.findall(pattern, characters_text, re.DOTALL)
            if matches:
                for match in matches:
                    # 安全检查：确保match有足够的元素
                    if len(match) >= 3:
                        character = {
                            'name': match[0].strip(),
                            'motivation': match[1].strip().replace('\\"', '"'),
                            'background': match[2].strip().replace('\\"', '"')
                        }
                        characters_list.append(character)
                    else:
                        # 如果元素不足，记录错误并跳过
                        print(f"Warning: Incomplete match found in pattern {pattern}: {match}")
                        continue
                break
        
        return characters_list
        
    except Exception as e:
        print(f"Error parsing characters: {e}")
        return []

def extract_json_response(response_text: str) -> List[Dict[str, Any]]:
    """从response中提取JSON格式的数据"""
    try:
        if not response_text or not isinstance(response_text, str):
            return []
        
        # 查找JSON代码块
        json_pattern = r'```json\s*(\[.*?\])\s*```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        extracted_items = []
        for match in matches:
            try:
                json_data = json.loads(match)
                if isinstance(json_data, list):
                    extracted_items.extend(json_data)
                elif isinstance(json_data, dict):
                    extracted_items.append(json_data)
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                continue
        
        return extracted_items
        
    except Exception as e:
        print(f"提取JSON响应时出错: {e}")
        return []

def process_directory(input_dir: str, model_name: str):
    """处理目录中的所有jsonl文件并将结果写入输出文件"""
    
    # 获取所有jsonl文件
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    jsonl_files = [file for file in jsonl_files if not file.endswith('failed.jsonl')]
    print(f"找到 {len(jsonl_files)} 个jsonl文件（跳过failed文件）")
    
    all_processed_data = []
    all_token_info = []
    all_failed_items = []
    all_skipped_items = []  # 新增：收集所有跳过的数据
    
    # 串行处理所有文件，使用主进度条
    for file_path in tqdm(jsonl_files, desc="处理文件", unit="file"):
        result, token_info, failed_items, skipped_items = process_file(file_path, model_name)
        all_processed_data.extend(result)
        all_token_info.extend(token_info)
        all_failed_items.extend(failed_items)
        all_skipped_items.extend(skipped_items)  # 收集跳过的数据
    output_path = f'{input_dir}_processed/{model_name}_processed.jsonl'
    print(f'all_processed_data: {len(all_processed_data)}, now write to {output_path}')
    
    # 写入输出文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存处理失败的数据
    if all_failed_items:
        failed_output_path = output_path.replace('.jsonl', '_parse_failed.jsonl')
        with open(failed_output_path, 'w', encoding='utf-8') as f:
            for item in all_failed_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"保存了 {len(all_failed_items)} 条解析失败的数据到 {failed_output_path}")
    
    # 保存跳过的数据
    if all_skipped_items:
        skipped_output_path = output_path.replace('.jsonl', '_skipped.jsonl')
        with open(skipped_output_path, 'w', encoding='utf-8') as f:
            for item in all_skipped_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"保存了 {len(all_skipped_items)} 条跳过的数据到 {skipped_output_path}")
    
    # 打印详细统计
    print("\n" + "="*60)
    print(f"📊 处理统计：")
    print(f"  ✅ 成功处理: {len(all_processed_data)} 条")
    print(f"  ⚠️ 跳过数据: {len(all_skipped_items)} 条")
    print(f"  ❌ 失败数据: {len(all_failed_items)} 条")
    print(f"  📝 总计: {len(all_processed_data) + len(all_skipped_items) + len(all_failed_items)} 条")
    print("="*60 + "\n")
    
    print(f"处理完成，共处理 {len(all_processed_data)} 条数据，已写入 {output_path}")

    # 提取judge_result
    print(f"all_processed_data: {len(all_processed_data)}, now extract judge_result")

    # 改进的抽取模式配置
    def get_field_patterns(model_name: str):
        """根据模型名称返回对应的字段抽取模式"""
        # 旧格式模式
        old_patterns = [
            ('scenario', '"scenario"\\s*:\\s*"', '",\\s*"topic":', 'string'),
            ('topic', '"topic"\\s*:\\s*"', '",\\s*"key_characters":', 'string'),
            ('key_characters', '"key_characters"\\s*:\\s*\\[', '\\]\\s*}\\s*\\]\\s*}', 'character_array')
        ]
        
        # 新JSON格式模式 - 支持提取多个字段
        json_patterns = [
            ('id', '"id"\\s*:\\s*(\\d+)', ',', 'integer'),
            ('type', '"type"\\s*:\\s*"([^"]*)"', ',', 'string'),
            ('from', '"from"\\s*:\\s*"([^"]*)"', ',', 'string'),
            ('sys_thinking_value', '"sys_thinking_value"\\s*:\\s*"(.*?)"(?=\\s*,\\s*"role_thinking_value")', ',', 'string'),
            ('role_thinking_value', '"role_thinking_value"\\s*:\\s*"(.*?)"(?=\\s*})', '}', 'string')
        ]
        
        # 不同模型可以有不同的模式
        model_patterns = {
            'claude_3_7_sonnet_20250219': json_patterns,  # 使用新的JSON模式
            'claude_opus_4_20250514': json_patterns,
            'claude_sonnet_4_20250514': json_patterns,
            'gemini_2_5_pro': json_patterns,
            'gemini_2_5_pro_preview_05_06': json_patterns,
            'gemini_2_5_pro_preview_06_05': json_patterns,
            'default': json_patterns
        }
        
        return model_patterns.get(model_name, json_patterns)
    
    field_patterns = get_field_patterns(model_name)
    extract_data = []
    success_count = 0
    error_count = 0
    
    for idx, item in enumerate(all_processed_data):
        try:
            item['enhanced_result'] = []
            response_text = item.get('response', '')
            
            # 安全检查：确保response_text是字符串
            if not isinstance(response_text, str):
                response_text = str(response_text) if response_text else ''
            
            if idx % 100 == 0:  # 每100条打印一次进度
                print(f"Processing item {idx+1}/{len(all_processed_data)}, response length: {len(response_text)}")
            
            # 首先尝试提取JSON格式的数据
            json_items = extract_json_response(response_text)
            
            if json_items:
                # 如果成功提取到JSON数据，直接使用
                item['enhanced_result'] = json_items
                extract_data.append(item)
                success_count += 1
                
                if idx % 100 == 0:
                    print(f"成功提取JSON数据，包含 {len(json_items)} 个项目")
            else:
                # 如果没有JSON数据，使用原有的字段提取方法作为备用
                result_dict = {
                    'scenario': "",
                    'topic': "",
                    'key_characters': []
                }
                
                for field_name, start_pattern, end_pattern, field_type in field_patterns:
                    try:
                        field_value = extract_field(response_text, start_pattern, end_pattern)
                        
                        # 根据字段类型进行后处理
                        if field_type == 'character_array' and field_name == 'key_characters':
                            field_value = parse_characters(field_value)
                        elif field_type == 'string':
                            # 清理字符串中的转义字符
                            if isinstance(field_value, str):
                                field_value = field_value.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                        elif field_type == 'integer':
                            # 处理整数类型
                            try:
                                field_value = int(field_value) if field_value else 0
                            except ValueError:
                                field_value = 0
                        
                        result_dict[field_name] = field_value
                    except Exception as field_error:
                        print(f"Warning: Error processing field {field_name} in item {idx}: {field_error}")
                        # 设置默认值
                        if field_type == 'character_array':
                            result_dict[field_name] = []
                        elif field_type == 'integer':
                            result_dict[field_name] = 0
                        else:
                            result_dict[field_name] = ""
                        continue
                
                item['enhanced_result'].append(result_dict)
                extract_data.append(item)
                success_count += 1
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            error_count += 1
            # 即使出错也保留原始数据
            item['enhanced_result'] = [{'error': str(e)}]
            extract_data.append(item)
    
    print(f"抽取完成: 成功 {success_count} 条, 错误 {error_count} 条")
    
    # 统计各字段的抽取情况
    field_stats = {'id': 0, 'type': 0, 'from': 0, 'sys_thinking_value': 0, 'role_thinking_value': 0, 
                   'scenario': 0, 'topic': 0, 'key_characters': 0}
    character_count = 0
    json_items_count = 0
    
    for item in extract_data:
        if item.get('enhanced_result'):
            enhanced_results = item['enhanced_result']
            
            # 如果是JSON格式的数据（多个项目）
            if isinstance(enhanced_results, list) and len(enhanced_results) > 1:
                json_items_count += len(enhanced_results)
                for result in enhanced_results:
                    # 统计新的JSON字段
                    for field in ['id', 'type', 'from', 'sys_thinking_value', 'role_thinking_value']:
                        if result.get(field):
                            field_stats[field] += 1
            
            # 如果是单个结果（旧格式或JSON中只有一个项目）
            elif isinstance(enhanced_results, list) and len(enhanced_results) >= 1:
                result = enhanced_results[0]
                
                # 统计JSON字段
                for field in ['id', 'type', 'from', 'sys_thinking_value', 'role_thinking_value']:
                    if result.get(field):
                        field_stats[field] += 1
                
                # 统计旧格式字段
                if result.get('scenario'):
                    field_stats['scenario'] += 1
                if result.get('topic'):
                    field_stats['topic'] += 1
                if result.get('key_characters'):
                    field_stats['key_characters'] += 1
                    # 安全检查：确保key_characters是列表
                    if isinstance(result['key_characters'], list):
                        character_count += len(result['key_characters'])
                    else:
                        print(f"Warning: key_characters is not a list in item: {type(result['key_characters'])}")
    
    print("字段抽取统计:")
    print(f"  JSON项目总数: {json_items_count}")
    for field, count in field_stats.items():
        if count > 0:  # 只显示有数据的字段
            print(f"  {field}: {count}/{len(extract_data)} ({count/len(extract_data)*100:.1f}%)")
    if character_count > 0:
        print(f"  总角色数量: {character_count}")
    
    # 保存结果
    enhanced_output_path = output_path.replace('.jsonl', '_enhanced_result.jsonl')
    with open(enhanced_output_path, 'w', encoding='utf-8') as f:
        for item in extract_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"增强结果已保存到: {enhanced_output_path}")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="处理vulcan输出格式的数据")
    parser.add_argument("--input_dir", type=str, required=True, help="输入目录，包含jsonl文件")
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    args = parser.parse_args()
    print(f"args: {args}")
    process_directory(args.input_dir, args.model_name)




python_cmd = """
python  /path/to/data/example
    --input_dir /path/to/data/example
    --model_name claude_3_7_sonnet_20250219
    """
python_cmd = """
python /path/to/data/example
    --input_dir /path/to/data/example
    --model_name model
    """

python_cmd = """
python /path/to/data/example
    --input_dir /path/to/data/example
    --model_name model
    """
