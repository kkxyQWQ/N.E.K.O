from typing import Any
from plugin.sdk.plugin import (
    NekoPluginBase, neko_plugin, plugin_entry, lifecycle,
    Ok, Err, SdkError,
)


@neko_plugin
class SquareCalculatorPlugin(NekoPluginBase):
    """计算平方的插件示例。"""

    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.logger = ctx.logger
        self.calculate_count = 0

    @lifecycle(id="startup")
    async def on_startup(self, **_):
        self.logger.info("SquareCalculatorPlugin started!")
        return Ok({"status": "ready"})

    @lifecycle(id="shutdown")
    async def on_shutdown(self, **_):
        self.logger.info("SquareCalculatorPlugin stopped!")
        return Ok({"status": "stopped"})

    @plugin_entry(
        id="square",
        name="Square",
        description="Calculate the square of a number",
        input_schema={
            "type": "object",
            "properties": {
                "number": {
                    "type": "number",
                    "description": "The number to square"
                }
            },
            "required": ["number"]
        }
    )
    async def square(self, number: float, **_):
        """输入一个数字，返回它的平方。"""
        self.calculate_count += 1

        if not isinstance(number, (int, float)):
            return Err(SdkError("Input must be a number"))

        result = number ** 2
        self.logger.info(f"Square({number}) = {result}")

        return Ok({
            "input": number,
            "result": result,
            "count": self.calculate_count
        })
