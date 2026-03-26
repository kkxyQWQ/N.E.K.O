from plugin.sdk.plugin import (
    NekoPluginBase, neko_plugin, plugin_entry, lifecycle,
    Ok, Err,
)
from typing import Any

@neko_plugin
class HelloWorldPlugin(NekoPluginBase):
    """Hello World 插件示例。"""

    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.logger = ctx.logger
        self.counter = 0

    @lifecycle(id="startup")
    def on_startup(self, **_):
        self.logger.info("HelloWorldPlugin started!")
        return Ok({"status": "ready"})

    @lifecycle(id="shutdown")
    def on_shutdown(self, **_):
        self.logger.info("HelloWorldPlugin stopped!")
        return Ok({"status": "stopped"})

    @plugin_entry(
        id="greet",
        name="Greet",
        description="Return a greeting message",
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name to greet",
                    "default": "World"
                }
            }
        }
    )
    async def greet(self, name: str = "World", **_):
        self.counter += 1
        message = f"Hello, {name}! (call #{self.counter})"
        self.logger.info(f"Greeting: {message}")
        return Ok({"message": message, "count": self.counter})
