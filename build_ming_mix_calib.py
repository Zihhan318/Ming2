#!/usr/bin/env python3
import json
from pathlib import Path

from ming_quantization_common import REPO_ROOT


OUTPUT_PATH = REPO_ROOT / "data" / "calib" / "mix_calib.jsonl"
TARGET_SIZE = 256


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        normalized = " ".join(item.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(item.strip())
    return ordered


def build_entries() -> list[str]:
    prompts: list[str] = []

    seed_prompts = [
        "你好，请用一句话介绍你自己。",
        "用一句话解释什么是光合作用。",
        "北京和上海，哪个是中国的首都？",
        "请写一首四句、每句不超过10个字的春天短诗。",
        "请把“人工智能正在改变世界”改写成更正式的书面语。",
        "请用两句话解释为什么天空是蓝色的。",
        "把下面这句话翻译成英文：今天天气很好，我们去公园散步吧。",
        "把下面这句话翻译成中文：Knowledge is power, but wisdom is knowing how to use it.",
        "请给“坚持”写三个近义词。",
        "请列出番茄炒蛋的三个主要食材。",
        "如果一本书原价80元，打八折后多少钱？",
        "一个长方形长8厘米、宽5厘米，面积是多少？",
        "请用一句话总结《西游记》的主题。",
        "请解释“授人以鱼不如授人以渔”的含义。",
        "请判断这句话是否有语病，并简要说明：通过大家的努力，使项目顺利完成。",
        "请把下面内容整理成一条会议通知：明天下午三点在三楼会议室开产品评审会，参会人员为产品部和研发部全体成员。",
        "请写一个 Python 函数，返回列表中的最大值。",
        "请给出一个 JSON 示例，包含 name、age、city 三个字段。",
        "请模拟客服，用礼貌语气回复：我的快递为什么还没到？",
        "请用三点概括早睡的好处。",
    ]
    prompts.extend(seed_prompts)

    knowledge_topics = [
        "量子计算", "供应链管理", "水循环", "区块链", "碳中和", "机器学习",
        "数据库索引", "缓存穿透", "容器编排", "网络延迟", "光纤通信", "疫苗原理",
        "民法中的合同效力", "公文写作", "财务报表", "汇率波动", "搜索引擎排序",
        "边缘计算", "数据治理", "用户增长", "城市轨道交通", "工业机器人", "新材料",
        "分布式事务",
    ]
    for topic in knowledge_topics:
        prompts.append(f"请用一句话解释什么是{topic}。")
        prompts.append(f"请用三点概括{topic}的主要特点。")

    rewrite_topics = [
        "公司正在推进流程优化", "项目进入关键阶段", "团队需要加强协作", "用户反馈值得重视",
        "研发成本正在上升", "这个方案存在风险", "数据质量影响决策", "产品体验需要提升",
        "我们要控制预算", "客户希望尽快上线", "线上故障已经恢复", "需求范围需要收敛",
    ]
    for sentence in rewrite_topics:
        prompts.append(f"请把“{sentence}”改写成更正式的书面语。")
        prompts.append(f"请把“{sentence}”改写成适合周报里的表达。")

    translation_pairs = [
        "知识需要通过实践转化为能力。",
        "系统的稳定性比功能数量更重要。",
        "今天的会议重点讨论性能瓶颈和上线计划。",
        "我们需要先定位根因，再决定修复策略。",
        "数据越及时，决策通常越有效。",
        "沟通成本会随着团队规模增加而上升。",
        "自动化测试可以降低回归风险。",
        "这个接口需要兼容旧版本客户端。",
    ]
    for text in translation_pairs:
        prompts.append(f"把下面这句话翻译成英文：{text}")
    english_pairs = [
        "A reliable process is often more valuable than a heroic fix.",
        "Good calibration data should cover both common and rare paths.",
        "The final decoder layer should stay on the primary device.",
        "A model can pass smoke tests and still fail long-form generation.",
        "MoE routing coverage matters during post-training quantization.",
        "Prompt diversity improves calibration robustness.",
        "Quantization should be measured by both quality and stability.",
        "Profiling is only useful when the workload is representative.",
    ]
    for text in english_pairs:
        prompts.append(f"把下面这句话翻译成中文：{text}")

    math_cases = [
        (128, 35), (96, 18), (245, 57), (360, 48), (512, 73), (144, 19),
        (88, 15), (1000, 125), (780, 96), (56, 13),
    ]
    for total, cost in math_cases:
        prompts.append(f"一个项目预算为{total}万元，已经花费{cost}万元，还剩多少万元？")
        prompts.append(f"如果商品原价是{total}元，立减{cost}元后再打九折，最终价格是多少？")

    code_tasks = [
        "请写一个 Python 函数，对列表进行去重并保持原顺序。",
        "请写一个 Python 函数，统计字符串中每个字符出现的次数。",
        "请写一个 Python 函数，判断一个字符串是不是回文。",
        "请写一个 Python 函数，把秒数转换成时分秒。",
        "请写一个 Python 函数，合并两个按升序排列的列表。",
        "请写一个 Python 函数，从字典列表中按键去重。",
        "请写一个 Python 函数，按指定大小切分列表。",
        "请写一个 Python 函数，返回两个集合的交集和差集。",
        "请写一个 Python 函数，计算斐波那契数列的前 n 项。",
        "请写一个 Python 函数，找出列表中的第二大值。",
        "请给出一个 SQL 查询，统计每天新增用户数。",
        "请给出一个 SQL 查询，找出每个部门薪资最高的员工。",
    ]
    prompts.extend(code_tasks)

    long_form_tasks = [
        "请写一段 150 字左右的项目周报总结，主题是性能优化阶段性完成，但仍有稳定性风险。",
        "请写一段 150 字左右的复盘总结，主题是一次线上故障的定位与修复。",
        "请写一段 150 字左右的产品说明，主题是新版本支持多设备协同。",
        "请写一段 150 字左右的公告，主题是系统将在周末凌晨维护升级。",
        "请写一段 150 字左右的邮件回复，主题是向客户解释交付延迟原因。",
        "请写一段 150 字左右的建议，主题是如何提高跨团队协作效率。",
        "请写一段 150 字左右的说明，主题是为什么要做校准集扩充。",
        "请写一段 150 字左右的总结，主题是量化项目当前的主要风险。",
    ]
    prompts.extend(long_form_tasks)

    image_subjects = [
        "雨夜中的赛博朋克街景", "竹林里的水墨山居", "清晨海边的灯塔", "玻璃温室里的热带植物",
        "雪山脚下的木屋", "蒸汽朋克风格的工作室", "复古书店的阅读角", "城市天台上的花园",
        "月光下的沙漠营地", "机械感十足的未来实验室", "传统茶馆的室内一角", "黄昏时分的港口"
    ]
    image_styles = [
        "写实摄影风格", "电影分镜风格", "水彩插画风格", "国风水墨风格",
        "高细节数字绘画", "柔和暖光风格"
    ]
    for subject in image_subjects:
        for style in image_styles[:4]:
            prompts.append(f"请为图像生成模型写一个提示词，主题是“{subject}”，要求采用{style}，突出构图、光线和氛围。")

    edit_cases = [
        "把生日蛋糕上的草莓换成蓝莓，保留奶油质感和俯拍构图。",
        "把城市街景改成雨天夜景，保留招牌和路面反光。",
        "把人物服装改成深绿色风衣，保留姿态和背景。",
        "把室内暖光改成冷色霓虹光，保留家具布局。",
        "把画面中的晴天改成日落时刻，保留建筑轮廓。",
        "把桌面上的马克杯换成透明玻璃杯，保留整体色调。",
        "把山谷中的河流改得更宽一些，保留远山和云层。",
        "把插画风格改成更偏写实的油画质感，保留主体构图。",
    ]
    for case in edit_cases:
        prompts.append(f"请将下面的图像编辑需求整理成一条清晰的中文指令：{case}")
        prompts.append(f"请把下面的图像编辑需求改写成适合模型理解的英文提示词：{case}")

    english_reasoning = [
        "Summarize the main risk of quantizing a mixture-of-experts model in two sentences.",
        "Explain why calibration coverage matters for W8A8 post-training quantization.",
        "List three signs that a quantized model is not production ready.",
        "Rewrite this sentence in a more formal tone: the current build still feels fragile.",
        "Provide a short meeting note about profiling, calibration, and regression testing.",
        "Explain the difference between smoke testing and full regression in one paragraph.",
        "Write a polite customer-support reply for a delayed shipment inquiry.",
        "Give three recommendations for improving data quality in model evaluation.",
    ]
    prompts.extend(english_reasoning)

    classification_cases = [
        "请判断下面这句话更像是需求、方案、风险还是结论：需要先补齐校准集，再做全量 W8A8 尝试。",
        "请判断下面这句话更像是需求、方案、风险还是结论：当前 attention 层仍然存在数值稳定性风险。",
        "请判断下面这句话更像是需求、方案、风险还是结论：先对文本链路回归，再看 imagegen 条件嵌入漂移。",
        "请判断下面这句话更像是需求、方案、风险还是结论：建议新增 expert coverage 报告作为量化前检查项。",
    ]
    prompts.extend(classification_cases)

    prompts = dedupe_keep_order(prompts)
    if len(prompts) < TARGET_SIZE:
        extra_topics = [
            "DevOps", "日志分析", "模型压缩", "缓存一致性", "性能测试", "异常监控",
            "文档治理", "版本回滚", "数据回灌", "发布审批", "接口幂等性", "故障演练",
        ]
        for topic in extra_topics:
            prompts.append(f"请写一段 100 字左右的说明，主题是{topic}在工程实践中的作用。")
            prompts.append(f"请用三点概括{topic}的常见风险。")
            prompts.append(f"请给出一个关于{topic}的简短示例。")
            if len(prompts) >= TARGET_SIZE:
                break

    return prompts[:TARGET_SIZE]


def main():
    entries = build_entries()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for prompt in entries:
            handle.write(json.dumps({"inputs_pretokenized": prompt}, ensure_ascii=False) + "\n")
    print(f"wrote {len(entries)} prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
