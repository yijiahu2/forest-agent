import os
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


# -----------------------------
# 1) 定义 Graph State
# -----------------------------
class ForestState(TypedDict, total=False):
    compartment_id: str
    stand_attributes: Dict[str, Any]
    segmentation_problem: str
    diagnosis: str
    experiment_plan: str
    messages: List[Dict[str, str]]


# -----------------------------
# 2) 初始化豆包 LLM
# -----------------------------
llm = ChatOpenAI(
    model=os.getenv("ARK_MODEL"),
    api_key=os.getenv("ARK_API_KEY"),
    base_url=os.getenv("ARK_BASE_URL"),
    temperature=0.2,
)


# -----------------------------
# 3) 节点1：诊断节点
#    根据小班属性和当前问题，先生成“问题诊断”
# -----------------------------
def diagnose_node(state: ForestState) -> ForestState:
    stand = state["stand_attributes"]
    problem = state["segmentation_problem"]

    prompt = f"""
你是林业遥感与单木分割专家。

现在有一个小班，其属性如下：
- 小班ID: {state['compartment_id']}
- 林龄: {stand.get('age')}
- 平均胸径: {stand.get('dbh_mean')}
- 平均树高: {stand.get('height_mean')}
- 平均冠幅: {stand.get('crown_mean')}
- 林木数量: {stand.get('tree_count')}
- 小班面积: {stand.get('area')}
- 郁闭度: {stand.get('canopy_closure')}
- 林分密度: {stand.get('density')}
- 是否有DEM: {stand.get('has_dem')}

当前分割现象描述：
{problem}

请你先只做“问题诊断”，不要给具体代码。
要求输出：
1. 当前主要失败类型（欠分裂/过分裂/漏检/尺度失衡/边界弱）
2. 造成该问题的最可能原因
3. 是否建议引入DEM约束
4. 是否建议后续轻量微调或伪标签 refinement
"""

    resp = llm.invoke(prompt)
    return {"diagnosis": resp.content}


# -----------------------------
# 4) 节点2：实验规划节点
#    基于诊断结果，生成下一轮实验方案
# -----------------------------
def planning_node(state: ForestState) -> ForestState:
    diagnosis = state["diagnosis"]
    stand = state["stand_attributes"]

    prompt = f"""
你是单木分割科研实验规划助手。

当前小班属性：
{stand}

上一节点给出的诊断如下：
{diagnosis}

请为基于 ZS-TreeSeg 的下一轮实验生成一个“可执行实验方案”。

输出必须分为5部分：
1. 参数调整建议
   - diam_list
   - tile / overlap
   - iou_merge_thr
   - 是否按小班类型设置不同阈值
2. 是否加入DEM派生约束
   - 局部高差
   - 坡度/坡向
   - 冠顶峰值辅助
3. 是否生成伪标签并做轻量微调
4. 需要做的消融实验列表（3~5项）
5. 该小班最推荐的优化路线（只给一个主路线）

要求：
- 面向复杂林分，不要泛泛而谈
- 结论要贴合小班统计特征
- 输出简洁、条理清晰
"""

    resp = llm.invoke(prompt)
    return {"experiment_plan": resp.content}


# -----------------------------
# 5) 构建 LangGraph
# -----------------------------
graph = StateGraph(ForestState)

graph.add_node("diagnose", diagnose_node)
graph.add_node("plan", planning_node)

graph.set_entry_point("diagnose")
graph.add_edge("diagnose", "plan")
graph.add_edge("plan", END)

app = graph.compile()


# -----------------------------
# 6) 运行示例
# -----------------------------
if __name__ == "__main__":
    init_state: ForestState = {
        "compartment_id": "XB_001",
        "stand_attributes": {
            "age": 28,
            "dbh_mean": 18.5,
            "height_mean": 16.2,
            "crown_mean": 3.8,
            "tree_count": 1450,
            "area": 0.82,
            "canopy_closure": 0.83,
            "density": 1768,
            "has_dem": True,
        },
        "segmentation_problem": (
            "ZS-TreeSeg 在该小班中对规整区域分割较好，但在高郁闭、混交、"
            "树冠交叠明显区域容易出现多个单木粘连成一个实例；"
            "同时小冠幅目标漏检较多，边界弱，尺度差异明显。"
        ),
    }

    result = app.invoke(init_state)

    print("\n========== 诊断结果 ==========\n")
    print(result["diagnosis"])

    print("\n========== 实验规划 ==========\n")
    print(result["experiment_plan"])