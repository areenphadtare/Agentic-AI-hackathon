"""Small MCP client for testing ArgusGraph's Streamable HTTP tools."""
import asyncio
import json
import os
from fastmcp import Client

MCP_URL = os.getenv("ARGUSGRAPH_MCP_URL", "http://127.0.0.1:8000/mcp")


async def main():
    async with Client(MCP_URL) as client:

        result = await client.call_tool("get_graph_overview", {})

        print("\nTemporal context:")
        data = (
            result.structured_content
            if result.structured_content is not None
            else str(result)
        )
        print(json.dumps(data, indent=2, default=str))



if __name__ == "__main__":
    asyncio.run(main())
