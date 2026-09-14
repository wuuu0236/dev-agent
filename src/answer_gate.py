"""
检索质量门控（阈值拒答）——决定检索结果够不够格拿来回答

为什么需要：
  此前不管检索到什么，top_k 都会被塞进 prompt。库里没有答案时，模型面对一堆
  不相关的「参考文档」仍会努力编一个——这是幻觉最直接的来源。
  **敢说"不知道"，比多答对一题更能决定产品的可信度。**

为什么建在精排分数上：
  精排（交叉编码）的分数有绝对意义：相关 0.6+、不相关 <0.01。
  而 RRF 分数是 1/(60+rank)，只表示相对顺序，没有阈值可言。
  `config.py` 的 rerank 一节早就论证过这条，本模块把它落地。

阈值怎么定的（kb 83cd2d0c 实测精排 top1，2026-09-14）：
  · 可答（库里有答案）：0.620（「混合检索是怎么做的」）；历史正样本最低 0.578
  · 不可答（库里没有 / 不该查库）：全部 ≤0.125 ——「你好」0.125、
    「你能做什么」0.122、「1+1 等于几」0.091、「帮我写首诗」0.010、「今天天气」0.001
  默认取 0.3：不可答的一律拦下，可答的留约 2 倍余量。

顺带解决的一件事：
  原本打算单独做「意图路由」（判断该不该查库）。实测发现**精排分数已经做到了**——
  闲聊 / 常识 / 越界问题的分数全部落在 0.125 以下，与可答问题相差 5 倍。
  所以不必再加一次 LLM 分类调用，一个阈值同时覆盖「幻觉」与「不该查库」。

一个必须说清的边界：
  阈值的语义是「**检索结果能否支撑回答**」，不是「问题该不该问」。
  「文档上传大小限制是多少」实测 0.039——库里确实没写，拒答是对的；
  但同样的低分也可能出现在「库里有、只是用户换了个说法」的情况。
  所以阈值宁可保守（漏拒好过误拒），且必须可配、可关。
"""
from src.config import RETRIEVAL_MIN_SCORE


def top_score(contexts: list[dict]) -> float:
    """取检索结果里的最高相关性分数。

    优先读精排分数（rerank_score）——开了精排后结果就是按它排序的；
    精排关闭 / 降级时回退到粗排的 rrf_score，行为与改动前一致。
    """
    best = 0.0
    for c in contexts or []:
        if not isinstance(c, dict):
            continue
        s = c.get("rerank_score")
        if s is None:
            s = c.get("rrf_score")
        try:
            s = float(s)
        except (TypeError, ValueError):
            continue
        if s > best:
            best = s
    return best


def is_grounded(contexts: list[dict], threshold: float | None = None) -> bool:
    """检索结果是否足以支撑回答。

    threshold 显式传入时覆盖配置（评测 / 测试做 A/B 用），None 则读
    `config.RETRIEVAL_MIN_SCORE`；阈值 <= 0 表示关闭门控（只要有检索结果就回答，
    与改动前行为一致）。
    """
    if not contexts:
        return False
    t = RETRIEVAL_MIN_SCORE if threshold is None else threshold
    if t <= 0:
        return True
    return top_score(contexts) >= t
