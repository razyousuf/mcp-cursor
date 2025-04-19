import asyncio
import os
import json
import time
from typing import Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class ChatConfig:
    def __init__(self, config_path: str = "chat_config.json"):
        self.config_path = config_path
        self.default_config = {
            "model": "qwen-qwq-32b",
            "max_steps": 15,
            "history_file": "chat_history",
            "config_file": "browser_mcp.json"
        }
        self.config = self.load_config()

    def load_config(self) -> dict:
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.save_config(self.default_config)
            return self.default_config

    def save_config(self, config: dict) -> None:
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def update_config(self, key: str, value: any) -> None:
        self.config[key] = value
        self.save_config(self.config)

class ChatSession:
    def __init__(self, agent: MCPAgent):
        self.agent = agent
        self.start_time = datetime.now()
        self.history_path = Path("chat_histories")
        self.history_path.mkdir(exist_ok=True)

    def export_history(self) -> str:
        """Export conversation history to a JSON file."""
        if not self.agent.conversation_history:
            return "No history to export."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.history_path / f"chat_history_{timestamp}.json"
        
        history_data = {
            "timestamp": timestamp,
            "history": self.agent.conversation_history
        }
        
        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=4)
        
        return f"History exported to {filename}"

    def import_history(self, filename: str) -> str:
        """Import conversation history from a JSON file."""
        try:
            filepath = self.history_path / filename
            with open(filepath, 'r') as f:
                history_data = json.load(f)
            self.agent.conversation_history = history_data["history"]
            return "History imported successfully."
        except Exception as e:
            return f"Error importing history: {e}"

async def run_memory_chat():
    """Run an enhanced chat using MCPAgent's built-in conversation memory."""
    
    # Load environment variables and configuration
    load_dotenv()
    config = ChatConfig()
    
    # Validate API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        console.print("[red]Error: GROQ_API_KEY not found in environment variables.[/red]")
        return
    os.environ["GROQ_API_KEY"] = api_key

    console.print("[blue]Initializing chat...[/blue]")

    try:
        # Create MCP client and LLM
        client = MCPClient.from_config_file(config.config["config_file"])
        llm = ChatGroq(model=config.config["model"])

        # Create MCP agent with memory enabled
        agent = MCPAgent(
            llm=llm,
            client=client,
            max_steps=config.config["max_steps"],
            memory_enabled=True
        )

        # Initialize chat session
        session = ChatSession(agent)

        console.print("\n[green]=== Interactive MCP Chat ===[/green]")
        console.print("Commands:")
        console.print("  /exit, /quit - End conversation")
        console.print("  /clear - Clear conversation history")
        console.print("  /export - Export conversation history")
        console.print("  /import <filename> - Import conversation history")
        console.print("  /config <key> <value> - Update configuration")
        console.print("  /help - Show this help message")
        console.print("[green]=========================[/green]\n")

        # Main chat loop
        while True:
            user_input = console.input("[cyan]You:[/cyan] ")

            # Command handling
            if user_input.startswith('/'):
                cmd_parts = user_input[1:].split()
                cmd = cmd_parts[0].lower()

                if cmd in ["exit", "quit"]:
                    console.print("[yellow]Ending conversation...[/yellow]")
                    break
                
                elif cmd == "clear":
                    agent.clear_conversation_history()
                    console.print("[green]Conversation history cleared.[/green]")
                    continue
                
                elif cmd == "export":
                    result = session.export_history()
                    console.print(f"[green]{result}[/green]")
                    continue
                
                elif cmd == "import" and len(cmd_parts) > 1:
                    result = session.import_history(cmd_parts[1])
                    console.print(f"[green]{result}[/green]")
                    continue
                
                elif cmd == "config" and len(cmd_parts) > 2:
                    key, value = cmd_parts[1], cmd_parts[2]
                    try:
                        # Convert string value to appropriate type
                        if value.isdigit():
                            value = int(value)
                        elif value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        config.update_config(key, value)
                        console.print(f"[green]Configuration updated: {key} = {value}[/green]")
                    except Exception as e:
                        console.print(f"[red]Error updating config: {e}[/red]")
                    continue
                
                elif cmd == "help":
                    console.print("\n[yellow]Available Commands:[/yellow]")
                    console.print("  /exit, /quit - End conversation")
                    console.print("  /clear - Clear conversation history")
                    console.print("  /export - Export conversation history")
                    console.print("  /import <filename> - Import conversation history")
                    console.print("  /config <key> <value> - Update configuration")
                    console.print("  /help - Show this help message\n")
                    continue
                
                else:
                    console.print("[red]Unknown command. Type /help for available commands.[/red]")
                    continue

            # Get response from agent with progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Thinking...", total=None)
                try:
                    response = await agent.run(user_input)
                    console.print("\n[green]Assistant:[/green] ", end="")
                    console.print(response)
                except Exception as e:
                    console.print(f"\n[red]Error: {str(e)}[/red]")

    except Exception as e:
        console.print(f"\n[red]Fatal error: {str(e)}[/red]")
    
    finally:
        # Graceful shutdown
        if 'client' in locals() and client.sessions:
            console.print("\n[yellow]Closing active sessions...[/yellow]")
            await client.close_all_sessions()
            console.print("[green]Sessions closed successfully.[/green]")

if __name__ == "__main__":
    try:
        asyncio.run(run_memory_chat())
    except KeyboardInterrupt:
        console.print("\n[yellow]Received interrupt signal. Shutting down...[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {str(e)}[/red]")
