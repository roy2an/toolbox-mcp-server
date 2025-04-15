import asyncio
import cv2

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio

from .tools.compare_image_with_box import compare_image_with_box
from .tools.compare_image_with_ssim import compare_image_with_ssim

# Store notes as a simple key-value dict to demonstrate state management
notes: dict[str, str] = {}

server = Server("toolbox-mcp-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="toolbox_compare_image_with_box",
            description="Compares two images and returns the difference with boxes around the differences",
            inputSchema={
                "type": "object",
                "properties": {
                    "image1": {
                        "type": "string",
                        "description": "Path to the first image"
                    },
                    "image2": {
                        "type": "string",
                        "description": "Path to the second image"
                    },
                },
                "required": ["image1", "image2"],
            },
        ),
        types.Tool(
            name="toolbox_compare_image_with_ssim",
            description="Compares two images and returns the difference with SSIM score",
            inputSchema={
                "type": "object",
                "properties": {
                    "image1": {
                        "type": "string",
                        "description": "Path to the first image"
                    },
                    "image2": {
                        "type": "string",
                        "description": "Path to the second image"
                    },
                },
                "required": ["image1", "image2"],
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    try:
        match name.replace("toolbox_", ""):
            case "compare_image_with_box":
                return compare_image_with_box(
                    arguments["image1"],
                    arguments["image2"],
                )
            case "compare_image_with_ssim":
                return compare_image_with_ssim(
                    arguments["image1"],
                    arguments["image2"],
                )
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="toolbox-mcp-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )