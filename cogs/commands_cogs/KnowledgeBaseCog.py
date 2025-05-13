import discord
from discord.ext import commands
from discord import app_commands
import traceback
import asyncio
import io

from bot_utilities.rag_utils import RAGSystem
from bot_utilities.config_loader import config

class KnowledgeBaseCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.server_knowledge_bases = {}
        
    def get_server_rag(self, guild_id: str) -> RAGSystem:
        """Get or create a RAG system for the server"""
        if guild_id not in self.server_knowledge_bases:
            self.server_knowledge_bases[guild_id] = RAGSystem(guild_id)
        return self.server_knowledge_bases[guild_id]
    
    @app_commands.command(name="kb-add", description="Add knowledge to the server's knowledge base")
    @app_commands.describe(content="The information to add to the knowledge base")
    async def kb_add_command(self, interaction: discord.Interaction, content: str):
        """
        Add information to the server's knowledge base
        
        Parameters:
        content (str): The information to add
        """
        # Defer the response since adding to KB might take time
        await interaction.response.defer(thinking=True)
        
        try:
            # Get server ID
            guild_id = str(interaction.guild.id)
            
            # Get RAG system for this server
            rag_system = self.get_server_rag(guild_id)
            
            # Add the content
            metadata = [{
                "source": "discord",
                "added_by": str(interaction.user.id),
                "added_by_name": str(interaction.user.name),
                "added_on": discord.utils.utcnow().isoformat()
            }]
            
            await rag_system.add_documents([content], metadata)
            
            await interaction.followup.send("✅ Knowledge added to the server's knowledge base!")
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in kb-add command: {error_traceback}")
            await interaction.followup.send(f"💥 Error: I encountered a problem while adding to the knowledge base. Please try again later.\n```{str(e)[:1500]}```")
    
    @app_commands.command(name="kb-upload", description="Upload a text file to the knowledge base")
    @app_commands.describe(file="Text file to upload (max 1MB)")
    async def kb_upload_command(self, interaction: discord.Interaction, file: discord.Attachment):
        """
        Upload a text file to the knowledge base
        
        Parameters:
        file (discord.Attachment): Text file to upload
        """
        # Check file size (max 1MB to prevent abuse)
        if file.size > 1_000_000:
            await interaction.response.send_message("⚠️ File too large. Maximum size is 1MB.")
            return
            
        # Check file type
        if not file.filename.endswith(('.txt', '.md', '.csv')):
            await interaction.response.send_message("⚠️ Only .txt, .md, and .csv files are supported.")
            return
            
        # Defer the response since processing might take time
        await interaction.response.defer(thinking=True)
        
        try:
            # Get the file content
            file_bytes = await file.read()
            file_content = file_bytes.decode('utf-8')
            
            # Get server ID
            guild_id = str(interaction.guild.id)
            
            # Get RAG system for this server
            rag_system = self.get_server_rag(guild_id)
            
            # Add the content
            metadata = [{
                "source": f"file:{file.filename}",
                "added_by": str(interaction.user.id),
                "added_by_name": str(interaction.user.name),
                "added_on": discord.utils.utcnow().isoformat()
            }]
            
            # For longer texts, chunk into paragraphs
            if len(file_content) > 2000:
                # Split by double newline (paragraphs)
                chunks = file_content.split('\n\n')
                # Filter out empty chunks and short chunks
                chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
                
                # Add chunks with metadata
                chunk_metadata = []
                for i, _ in enumerate(chunks):
                    meta = metadata[0].copy()
                    meta["chunk"] = i + 1
                    meta["total_chunks"] = len(chunks)
                    chunk_metadata.append(meta)
                
                count = await rag_system.add_documents(chunks, chunk_metadata)
                await interaction.followup.send(f"✅ File uploaded and processed into {count} knowledge chunks!")
            else:
                # Add as a single document
                await rag_system.add_documents([file_content], metadata)
                await interaction.followup.send("✅ File uploaded to the knowledge base!")
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in kb-upload command: {error_traceback}")
            await interaction.followup.send(f"💥 Error: I encountered a problem while processing your file. Please try again later.\n```{str(e)[:1500]}```")
    
    @app_commands.command(name="kb-query", description="Query the server's knowledge base")
    @app_commands.describe(query="The query to search for in the knowledge base")
    async def kb_query_command(self, interaction: discord.Interaction, query: str):
        """
        Query the server's knowledge base
        
        Parameters:
        query (str): The search query
        """
        # Defer the response since querying might take time
        await interaction.response.defer(thinking=True)
        
        try:
            # Get server ID
            guild_id = str(interaction.guild.id)
            
            # Get RAG system for this server
            rag_system = self.get_server_rag(guild_id)
            
            # Query the knowledge base
            results = await rag_system.query(query, k=3)
            
            if not results:
                await interaction.followup.send("ℹ️ No relevant information found in the knowledge base.")
                return
            
            # Format results
            response = f"🔍 Results for query: `{query}`\n\n"
            
            for i, doc in enumerate(results):
                response += f"**Result {i+1}**:\n"
                response += f"{doc.page_content[:500]}...\n"
                if "source" in doc.metadata:
                    response += f"*Source: {doc.metadata['source']}*\n"
                if "added_by_name" in doc.metadata:
                    response += f"*Added by: {doc.metadata['added_by_name']}*\n"
                response += "\n"
            
            # Split into chunks if needed
            if len(response) > 2000:
                chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                await interaction.followup.send(chunks[0])
                
                for chunk in chunks[1:]:
                    await interaction.channel.send(chunk)
            else:
                await interaction.followup.send(response)
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in kb-query command: {error_traceback}")
            await interaction.followup.send(f"💥 Error: I encountered a problem while querying the knowledge base. Please try again later.\n```{str(e)[:1500]}```")

async def setup(bot):
    await bot.add_cog(KnowledgeBaseCog(bot)) 