import os
import json
import time
import datetime
from typing import Dict, List, Any, Optional, Union
import asyncio
import aiofiles
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import io
import discord

class AgentMonitor:
    """Monitors and tracks agent performance and usage metrics"""
    
    def __init__(self, storage_dir='bot_data/metrics'):
        """Initialize the agent monitor"""
        self.storage_dir = storage_dir
        self.metrics_file = f"{storage_dir}/agent_metrics.json"
        self.command_usage_file = f"{storage_dir}/command_usage.json"
        self.user_stats_file = f"{storage_dir}/user_stats.json"
        
        # Ensure the storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize metrics files if they don't exist
        self._initialize_metrics_files()
    
    def _initialize_metrics_files(self):
        """Initialize the metrics files if they don't exist"""
        # Agent metrics file
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w') as f:
                json.dump([], f)
        
        # Command usage file
        if not os.path.exists(self.command_usage_file):
            with open(self.command_usage_file, 'w') as f:
                json.dump({}, f)
        
        # User stats file
        if not os.path.exists(self.user_stats_file):
            with open(self.user_stats_file, 'w') as f:
                json.dump({}, f)
    
    async def log_interaction(self, command_name: str, user_id: str, server_id: str, 
                              execution_time: float, success: bool, 
                              token_usage: Optional[Dict[str, int]] = None,
                              error: Optional[str] = None):
        """Log an agent interaction"""
        # Build the metric entry
        metric = {
            "timestamp": datetime.datetime.now().isoformat(),
            "command": command_name,
            "user_id": user_id,
            "server_id": server_id,
            "execution_time_seconds": execution_time,
            "success": success
        }
        
        if token_usage:
            metric["token_usage"] = token_usage
        
        if error:
            metric["error"] = error
        
        # Append to the metrics file
        async with aiofiles.open(self.metrics_file, 'r') as f:
            content = await f.read()
            metrics = json.loads(content) if content else []
        
        metrics.append(metric)
        
        async with aiofiles.open(self.metrics_file, 'w') as f:
            await f.write(json.dumps(metrics, indent=2))
        
        # Update command usage statistics
        await self._update_command_usage(command_name)
        
        # Update user statistics
        await self._update_user_stats(user_id, command_name, success, execution_time, token_usage)
    
    async def _update_command_usage(self, command_name: str):
        """Update command usage statistics"""
        async with aiofiles.open(self.command_usage_file, 'r') as f:
            content = await f.read()
            command_usage = json.loads(content) if content else {}
        
        # Initialize command if it doesn't exist
        if command_name not in command_usage:
            command_usage[command_name] = {
                "total_calls": 0,
                "calls_by_day": {}
            }
        
        # Update total calls
        command_usage[command_name]["total_calls"] += 1
        
        # Update calls for today
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        calls_by_day = command_usage[command_name]["calls_by_day"]
        
        if today not in calls_by_day:
            calls_by_day[today] = 0
        
        calls_by_day[today] += 1
        
        # Save updated command usage
        async with aiofiles.open(self.command_usage_file, 'w') as f:
            await f.write(json.dumps(command_usage, indent=2))
    
    async def _update_user_stats(self, user_id: str, command_name: str, success: bool, 
                                 execution_time: float, token_usage: Optional[Dict[str, int]] = None):
        """Update user statistics"""
        async with aiofiles.open(self.user_stats_file, 'r') as f:
            content = await f.read()
            user_stats = json.loads(content) if content else {}
        
        # Initialize user if they don't exist
        if user_id not in user_stats:
            user_stats[user_id] = {
                "total_interactions": 0,
                "commands_used": {},
                "total_tokens": {
                    "prompt": 0,
                    "completion": 0,
                    "total": 0
                },
                "success_rate": 0,
                "average_execution_time": 0,
                "last_interaction": None
            }
        
        user = user_stats[user_id]
        
        # Update total interactions
        user["total_interactions"] += 1
        
        # Update commands used
        if command_name not in user["commands_used"]:
            user["commands_used"][command_name] = 0
        
        user["commands_used"][command_name] += 1
        
        # Update token usage if provided
        if token_usage:
            user["total_tokens"]["prompt"] += token_usage.get("prompt_tokens", 0)
            user["total_tokens"]["completion"] += token_usage.get("completion_tokens", 0)
            user["total_tokens"]["total"] += token_usage.get("total_tokens", 0)
        
        # Update success rate
        successes = user.get("successful_interactions", 0)
        if success:
            successes += 1
        user["successful_interactions"] = successes
        user["success_rate"] = successes / user["total_interactions"]
        
        # Update average execution time
        current_avg = user.get("average_execution_time", 0)
        user["average_execution_time"] = (current_avg * (user["total_interactions"] - 1) + execution_time) / user["total_interactions"]
        
        # Update last interaction time
        user["last_interaction"] = datetime.datetime.now().isoformat()
        
        # Save updated user stats
        async with aiofiles.open(self.user_stats_file, 'w') as f:
            await f.write(json.dumps(user_stats, indent=2))
    
    async def get_command_usage_stats(self) -> Dict[str, Any]:
        """Get command usage statistics"""
        async with aiofiles.open(self.command_usage_file, 'r') as f:
            content = await f.read()
            return json.loads(content) if content else {}
    
    async def get_user_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user statistics, for a specific user or all users"""
        async with aiofiles.open(self.user_stats_file, 'r') as f:
            content = await f.read()
            user_stats = json.loads(content) if content else {}
        
        if user_id:
            return user_stats.get(user_id, {})
        else:
            return user_stats
    
    async def get_agent_metrics(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get agent metrics for the specified number of days"""
        async with aiofiles.open(self.metrics_file, 'r') as f:
            content = await f.read()
            metrics = json.loads(content) if content else []
        
        # Filter metrics for the specified time period
        cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
        return [m for m in metrics if m["timestamp"] >= cutoff_date]
    
    async def generate_usage_chart(self, days: int = 7) -> io.BytesIO:
        """Generate a chart of command usage over time"""
        command_usage = await self.get_command_usage_stats()
        
        # Get all dates in the specified period
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        date_range = [(start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d") 
                      for i in range(days)]
        
        # Create a dictionary to hold the data for each command
        commands_data = {}
        
        for command, data in command_usage.items():
            daily_counts = []
            for date in date_range:
                daily_counts.append(data.get("calls_by_day", {}).get(date, 0))
            
            if sum(daily_counts) > 0:  # Only include commands that were used
                commands_data[command] = daily_counts
        
        if not commands_data:
            # Create a simple message instead of a chart if no data
            fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No command usage data available", 
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
        else:
            # Create the chart
            fig = plt.figure(figsize=(10, 6))
            
            for command, counts in commands_data.items():
                plt.plot(date_range, counts, marker='o', label=command)
            
            plt.xlabel('Date')
            plt.ylabel('Number of calls')
            plt.title(f'Command Usage Over the Past {days} Days')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
        
        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return buf
    
    async def generate_performance_report(self, days: int = 7) -> str:
        """Generate a performance report for the specified number of days"""
        metrics = await self.get_agent_metrics(days)
        command_usage = await self.get_command_usage_stats()
        
        if not metrics:
            return "No agent performance data available for the specified time period."
        
        # Calculate overall statistics
        total_calls = len(metrics)
        successful_calls = sum(1 for m in metrics if m.get("success", False))
        success_rate = successful_calls / total_calls if total_calls > 0 else 0
        
        avg_execution_time = sum(m.get("execution_time_seconds", 0) for m in metrics) / total_calls if total_calls > 0 else 0
        
        # Get token usage if available
        total_prompt_tokens = sum(m.get("token_usage", {}).get("prompt_tokens", 0) for m in metrics)
        total_completion_tokens = sum(m.get("token_usage", {}).get("completion_tokens", 0) for m in metrics)
        total_tokens = sum(m.get("token_usage", {}).get("total_tokens", 0) for m in metrics)
        
        # Generate the report
        report = f"# Agent Performance Report (Past {days} days)\n\n"
        
        report += f"## Overall Statistics\n"
        report += f"- Total calls: {total_calls}\n"
        report += f"- Success rate: {success_rate:.2%}\n"
        report += f"- Average execution time: {avg_execution_time:.2f} seconds\n"
        
        if total_tokens > 0:
            report += f"- Total tokens used: {total_tokens:,}\n"
            report += f"  - Prompt tokens: {total_prompt_tokens:,}\n"
            report += f"  - Completion tokens: {total_completion_tokens:,}\n"
        
        # Command usage breakdown
        report += f"\n## Command Usage\n"
        for command, data in command_usage.items():
            # Only include commands used in the time period
            command_metrics = [m for m in metrics if m.get("command") == command]
            if command_metrics:
                command_success_rate = sum(1 for m in command_metrics if m.get("success", False)) / len(command_metrics)
                report += f"- {command}: {len(command_metrics)} calls, {command_success_rate:.2%} success rate\n"
        
        # Error analysis if there are failures
        failed_metrics = [m for m in metrics if not m.get("success", True)]
        if failed_metrics:
            report += f"\n## Error Analysis\n"
            error_counts = {}
            for m in failed_metrics:
                error = m.get("error", "Unknown error")
                if error not in error_counts:
                    error_counts[error] = 0
                error_counts[error] += 1
            
            # List the top 5 most common errors
            top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for error, count in top_errors:
                report += f"- {error}: {count} occurrences\n"
        
        return report

class PerformanceDecorator:
    """Decorator to track performance of agent commands"""
    
    def __init__(self, monitor: AgentMonitor):
        self.monitor = monitor
    
    def __call__(self, func):
        """Decorator for tracking performance of a command"""
        async def wrapper(*args, **kwargs):
            command_name = func.__name__
            
            # Extract the interaction object to get user and server IDs
            interaction = None
            for arg in args:
                if isinstance(arg, discord.Interaction):
                    interaction = arg
                    break
            
            if not interaction:
                # If no interaction found, just call the function normally
                return await func(*args, **kwargs)
            
            user_id = str(interaction.user.id)
            server_id = str(interaction.guild.id) if interaction.guild else "DM"
            
            start_time = time.time()
            success = True
            error = None
            token_usage = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                execution_time = time.time() - start_time
                
                # Log the interaction asynchronously
                asyncio.create_task(self.monitor.log_interaction(
                    command_name=command_name,
                    user_id=user_id,
                    server_id=server_id,
                    execution_time=execution_time,
                    success=success,
                    token_usage=token_usage,
                    error=error
                ))
        
        return wrapper 

class UserActivityMonitor:
    """Monitors user command usage and activity"""
    
    def __init__(self, storage_dir='bot_data/user_activity'):
        """Initialize the user activity monitor"""
        self.storage_dir = storage_dir
        self.activity_file = f"{storage_dir}/command_activity.json"
        
        # Ensure the storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize activity file if it doesn't exist
        if not os.path.exists(self.activity_file):
            with open(self.activity_file, 'w') as f:
                json.dump({}, f)
    
    async def log_command_usage(self, user_id: str, command_name: str, guild_id: str, success: bool = True):
        """Log command usage by a user"""
        try:
            # Read current activity data
            async with aiofiles.open(self.activity_file, 'r') as f:
                content = await f.read()
                activity_data = json.loads(content) if content else {}
            
            # Initialize user if they don't exist
            if user_id not in activity_data:
                activity_data[user_id] = {
                    "commands": {},
                    "guilds": {},
                    "total_commands": 0,
                    "successful_commands": 0,
                    "first_interaction": datetime.datetime.now().isoformat(),
                    "latest_interaction": None
                }
            
            user_data = activity_data[user_id]
            
            # Update command stats
            if command_name not in user_data["commands"]:
                user_data["commands"][command_name] = 0
            user_data["commands"][command_name] += 1
            
            # Update guild stats
            if guild_id not in user_data["guilds"]:
                user_data["guilds"][guild_id] = 0
            user_data["guilds"][guild_id] += 1
            
            # Update overall stats
            user_data["total_commands"] += 1
            if success:
                user_data["successful_commands"] += 1
            user_data["latest_interaction"] = datetime.datetime.now().isoformat()
            
            # Save updated activity data
            async with aiofiles.open(self.activity_file, 'w') as f:
                await f.write(json.dumps(activity_data, indent=2))
                
        except Exception as e:
            print(f"Error logging command usage: {e}")
    
    async def get_user_activity(self, user_id: str = None) -> Dict[str, Any]:
        """Get activity data for a specific user or all users"""
        try:
            async with aiofiles.open(self.activity_file, 'r') as f:
                content = await f.read()
                activity_data = json.loads(content) if content else {}
            
            if user_id:
                return activity_data.get(user_id, {})
            else:
                return activity_data
        except Exception as e:
            print(f"Error retrieving user activity: {e}")
            return {} 