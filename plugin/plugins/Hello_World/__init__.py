# 导入SDK核心工具
from plugin.sdk.plugin import (
    NekoPluginBase, neko_plugin, plugin_entry, lifecycle,
    Ok, Err,
)
from typing import Any

# 标记插件类
@neko_plugin
class HelloWorldPlugin(NekoPluginBase):
    """我的第一个GitHub在线编写插件"""

    # 插件初始化
    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.logger = ctx.logger
        self.counter = 0

    # 启动生命周期钩子
    @lifecycle(id="startup")
    def on_startup(self, **_):
        self.logger.info("GitHub写的插件启动成功！")
        return Ok({"status": "ready"})

    # 停止生命周期钩子
    @lifecycle(id="shutdown")
    def on_shutdown(self, **_):
        self.logger.info("插件已停止！")
        return Ok({"status": "stopped"})

    # 对外暴露的问候功能
    @plugin_entry(
        id="greet",
        name="问候功能",
        description="给输入的名字返回问候语",
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要问候的名字"
                }
            },
            "required": ["name"]
        }
    )
    def greet(self, name: str, **_):
        self.counter += 1
        self.logger.info(f"功能被调用！第{self.counter}次，调用者：{name}")
        return Ok(f"你好呀{name}！这是插件第{self.counter}次收到你的问候~"
