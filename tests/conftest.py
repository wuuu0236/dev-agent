"""pytest 公共夹具。

目前只有一个：`stub_langfuse` —— 在缺 langfuse 的环境里注入一个最小替身
（只提供 @observe 直通），让**主链路行为**的回归测试在任何环境都跑得起来。

为什么不做成 autouse：
  `test_smoke_imports.py` 存在的意义正是验证「langfuse 不可用时能否优雅降级」，
  给它塞替身等于把这个测试变成空转。所以只给明确需要它、且测的不是降级行为的
  文件用（显式声明依赖，不隐式生效）。
"""
import importlib
import importlib.machinery
import sys
import types

import pytest


def _make_stub() -> None:
    def observe(*a, **kw):
        if a and callable(a[0]):
            return a[0]
        return lambda fn: fn

    pkg = types.ModuleType("langfuse")
    dec = types.ModuleType("langfuse.decorators")
    dec.observe = observe
    pkg.decorators = dec
    # 补 __spec__：任何第三方代码用 find_spec 探测时，不会因为它是"裸模块"而报错
    pkg.__spec__ = importlib.machinery.ModuleSpec("langfuse", None)
    dec.__spec__ = importlib.machinery.ModuleSpec("langfuse.decorators", None)
    sys.modules["langfuse"] = pkg
    sys.modules["langfuse.decorators"] = dec


@pytest.fixture
def stub_langfuse():
    """保证 `from langfuse.decorators import observe` 能成功。

    ⚠️ 判"在不在"必须先看 `sys.modules`，**不能**用 `importlib.util.find_spec`：
    替身进过 sys.modules 之后，find_spec 会因为 `__spec__` 的查找路径而抛
    ValueError（而不是返回 None），于是第二个用到它的测试就崩了。
    ⚠️ 装了真 langfuse 就用真的 —— 替身只补缺，不覆盖。
    """
    if "langfuse" not in sys.modules:
        try:
            # 用 `importlib.import_module` 而不是 `import langfuse`：这里要的只是
            # 「装了没有」这个事实，名字本身不会被用到。写成 import 语句的话静态检查
            # 会把 `# noqa` 之外的一切都报成未使用导入——而 pyflakes 根本不认 noqa
            # （那是 flake8 的功能），于是每次全仓扫描都留一条假告警。
            importlib.import_module("langfuse")
        except ImportError:
            _make_stub()
