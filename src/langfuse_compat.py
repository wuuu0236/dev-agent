"""langfuse + DeepSeek 兼容补丁。

langfuse.openai 会把全局 openai 客户端包装起来做观测，但它解析响应时
假设 usage 字段一定存在：DeepSeek 等 OpenAI 兼容 API 部分响应不返回 usage，
langfuse 在 `usage.__dict__` 处崩溃，并把异常 re-raise 给上层——
导致真实的 LLM 响应丢失（不只是少记一条 trace，是整个调用失败）。

这里在崩溃点做防御：usage 为 None 时安全返回，langfuse 只是少记 usage，
不吞掉真实响应。import 本模块即生效，需在 langfuse.openai 之后导入。
"""
import langfuse.openai as _lo


def _safe_default_response(open_ai_resource, openai_response):
    """同 _get_langfuse_data_from_default_response，但容忍 usage=None。"""
    model = openai_response.get("model", None)
    completion = None

    if open_ai_resource.type == "completion":
        choices = openai_response.get("choices", [])
        if choices:
            choice = choices[-1]
            completion = choice.text if _lo._is_openai_v1() else choice.get("text", None)
    elif open_ai_resource.type == "chat":
        choices = openai_response.get("choices", [])
        if choices:
            choice = choices[-1]
            completion = (
                _lo._extract_chat_response(choice.message.__dict__)
                if _lo._is_openai_v1()
                else choice.get("message", None)
            )

    usage = openai_response.get("usage", None)
    if _lo._is_openai_v1() and usage is not None:
        usage = usage.__dict__
    return model, completion, usage


_lo._get_langfuse_data_from_default_response = _safe_default_response
