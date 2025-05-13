import discord
from discord.ext import commands
from discord import app_commands
import traceback
import asyncio

from bot_utilities.reflective_rag import SelfReflectiveRAG

class ReflectiveRAGCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.server_rag_systems = {}
    
    def get_server_rag(self, guild_id: str) -> SelfReflectiveRAG:
        """Get or create a reflective RAG system for the server"""
        if guild_id not in self.server_rag_systems:
            self.server_rag_systems[guild_id] = SelfReflectiveRAG(guild_id)
        return self.server_rag_systems[guild_id]
    
    @app_commands.command(name="reflective-search", description="Search the knowledge base with quality assessment")
    @app_commands.describe(query="The search query")
    async def reflective_search_command(self, interaction: discord.Interaction, query: str):
        """
        Search the knowledge base with quality assessment using self-reflective RAG
        
        Parameters:
        query (str): The search query
        """
        # Defer the response since this might take time
        await interaction.response.defer(thinking=True)
        
        try:
            # Get server ID
            guild_id = str(interaction.guild.id)
            
            # Get reflective RAG system for this server
            rag_system = self.get_server_rag(guild_id)
            
            # Show typing indicator
            async with interaction.channel.typing():
                # Start a spinner indicator
                message = await interaction.followup.send(f"🔄 Searching with reflection: `{query}`")
                
                # Get results with reflection
                documents, scores = await rag_system.query_with_reflection(query)
                
                if not documents:
                    await message.edit(content="ℹ️ No relevant information found in the knowledge base.")
                    return
                
                # Format response
                response = f"🔍 Reflective search results for: `{query}`\n\n"
                
                for i, (doc, score) in enumerate(zip(documents, scores)):
                    response += f"**Result {i+1} - Relevance: {score.score:.2f}**\n"
                    response += f"{doc.page_content[:500]}...\n\n"
                    
                    # Add reasoning for relevance score
                    response += f"*Why this is relevant: {score.reasoning}*\n\n"
                    
                    # Add source information if available
                    if "source" in doc.metadata:
                        response += f"*Source: {doc.metadata['source']}*\n"
                    
                    if "added_by_name" in doc.metadata:
                        response += f"*Added by: {doc.metadata['added_by_name']}*\n"
                    
                    response += "\n"
                
                # Split into chunks if needed
                if len(response) > 2000:
                    chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                    await message.edit(content=chunks[0])
                    
                    for chunk in chunks[1:]:
                        await interaction.channel.send(chunk)
                else:
                    await message.edit(content=response)
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in reflective-search command: {error_traceback}")
            await interaction.followup.send(f"💥 Error: I encountered a problem while performing reflective search. Please try again later.\n```{str(e)[:1500]}```")
    
    @app_commands.command(name="smart-ask", description="Ask a question using self-reflective knowledge retrieval")
    @app_commands.describe(question="The question to ask")
    async def smart_ask_command(self, interaction: discord.Interaction, question: str):
        """
        Ask a question using self-reflective knowledge retrieval
        
        Parameters:
        question (str): The question to ask
        """
        # Defer the response since this might take time
        await interaction.response.defer(thinking=True)
        
        try:
            # Get server ID
            guild_id = str(interaction.guild.id)
            
            # Get reflective RAG system for this server
            rag_system = self.get_server_rag(guild_id)
            
            # Show typing indicator
            async with interaction.channel.typing():
                # Start a spinner indicator
                message = await interaction.followup.send(f"🔄 Processing question: `{question}`")
                
                # Get reflective context
                context = await rag_system.format_reflective_results(question)
                
                # If no relevant context was found
                if not context:
                    await message.edit(content=f"I don't have enough information in my knowledge base to answer this question: `{question}`\n\nConsider adding relevant information with the /kb-add command first.")
                    return
                
                # Use the LLM to answer the question based on the retrieved context
                prompt = f"""You are a helpful assistant answering questions based on retrieved information.
                
                Retrieved Information:
                {context}
                
                Question: {question}
                
                Using only the retrieved information above, provide a comprehensive answer to the question.
                If the retrieved information doesn't contain enough details to answer the question fully, acknowledge what you know and what information is missing.
                Always cite your sources when possible.
                """
                
                # Get answer from the LLM
                answer = await rag_system.llm.ainvoke(prompt)
                
                # Format the response
                response = f"**Question:** {question}\n\n{answer.content}"
                
                # Split into chunks if needed
                if len(response) > 2000:
                    chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                    await message.edit(content=chunks[0])
                    
                    for chunk in chunks[1:]:
                        await interaction.channel.send(chunk)
                else:
                    await message.edit(content=response)
                
        except Exception as e:
            # Handle errors
            error_traceback = traceback.format_exc()
            print(f"Error in smart-ask command: {error_traceback}")
            await interaction.followup.send(f"💥 Error: I encountered a problem while answering your question. Please try again later.\n```{str(e)[:1500]}```")

async def setup(bot):
    await bot.add_cog(ReflectiveRAGCog(bot)) 