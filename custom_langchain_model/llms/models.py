from typing import List, Optional, Sequence, Any, Dict, Union
import asyncio
import uuid
import logging
import httpx
import json
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.callbacks import CallbackManagerForLLMRun
from custom_langchain_model.core.config import settings
from custom_langchain_model.core.security import get_access_token


logger = logging.getLogger(__name__)


class AzureOpenAICompatibleChat(BaseChatModel):
    conversation_id: Optional[int] = None
    engine: str
    endpoint: str = None

    
    def model_post_init(self, __context):
        self.endpoint = settings.API_BASE_URL.format(model=self.engine)


    def _prepare_messages(self, messages: List[ChatMessage]) -> dict:
        """
        Convert LangChain messages to Azure OpenAI Chat API format.
        This includes handling tool calls in AI messages and route
        tool results back to the API for further AI interpretation.
        """
        payload_messages = []
        for msg in messages:
            message_dict = {"content": msg.content or ""}
            
            # determine role and other fields
            if isinstance(msg, HumanMessage):
                message_dict["role"] = "user"                
            elif isinstance(msg, AIMessage):
                message_dict["role"] = "assistant"
                if getattr(msg, "tool_calls", None):
                    # Convert tool_calls to Azure OpenAI format
                    # Each call must include an id and function info
                    message_dict["tool_calls"] = []
                    for call in msg.tool_calls:
                        message_dict["tool_calls"].append({
                            "id": call["id"],
                            "type": "function", 
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["args"]),
                            }
                        })            
            elif isinstance(msg, SystemMessage):
                message_dict["role"] = "system"                
            elif isinstance(msg, ToolMessage):
                message_dict["role"] = "tool"
                message_dict["name"] = msg.name
                message_dict["tool_call_id"] = msg.tool_call_id
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")            
            payload_messages.append(message_dict)
        return payload_messages


    async def _chat_completion_request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        context: Optional[dict] = None,
        stream: bool = False,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> Dict[str, Any]:
        message_history = context.message_history or []

        messages = message_history + messages

        payload = {
            "messages": self._prepare_messages(messages),
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {get_access_token()}",
        }
        
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        """
        kwargs can contain:
          tools: list of tool specs (from bind_tools)
          tool_choice: optional forcing spec
          context: for conversation id injection
          stream: bool
        """
        config: RunnableConfig = kwargs.get("config") or {}
        configurable = config.get("configurable", {})

        tools = kwargs.get("tools") or configurable.get("tools")
        tool_choice = kwargs.get("tool_choice") or configurable.get("tool_choice")
        
        context = kwargs.get("context")
        stream = kwargs.get("stream", False)     
        
        data = await self._chat_completion_request(
            messages,
            stop=stop,
            context=context,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
        )

        message_dict = data["choices"][0]["message"]

        # Parse tool calls if present
        tool_calls_raw = message_dict.get("tool_calls") or []
        parsed_tool_calls = []
        
        # For OpenAI-compatible response
        for tc in tool_calls_raw:
            if tc.get("type") == "function":
                fn = tc.get("function", {})
                name = fn.get("name")
                raw_args = fn.get("arguments", "") or "{}"
                try:
                    args_obj = json.loads(raw_args)
                except Exception:
                    args_obj = {"_raw": raw_args}
                parsed_tool_calls.append({
                    "name": name,
                    "args": args_obj,
                    "id": tc.get("id", str(uuid.uuid4()))
                })
        
        content = message_dict.get("content") or ""

        ai_msg = AIMessage( content=content, tool_calls=parsed_tool_calls ) if parsed_tool_calls else AIMessage( content=content )

        generation = ChatGeneration(
            message=ai_msg, 
            generation_info={"raw": data}
        )
               
        return ChatResult(generations=[generation])        
    

    def _generate(self, messages, stop=None,  **kwargs) -> ChatResult:
        raise NotImplementedError("AzureCompatibleChat only supports async calls")        
    

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: Optional[str | Dict[str, Any]] = None
    ) -> Runnable:
        """
        Return a new runnable with tools pre-bound.
        Usage:
            model_with_tools = model.bind_tools([search_tool, calc_tool])
            model_with_tools.invoke(messages)
        """
        tool_specs = [self._convert_tool(tool) for tool in tools]
        extra: Dict[str, Any] = {"tools": tool_specs}
        if tool_choice is not None:
            # Accept "auto", "none", or {"type": "function", "function": {"name": "..."}}
            extra["tool_choice"] = tool_choice
        return self.bind(**extra)

    
    def _convert_tool(self, tool: BaseTool) -> dict:
        """
        Convert a LangChain tool to OpenAI/Azure function spec.
        """
        if getattr(tool, "args_schema", None):
            schema = tool.args_schema.model_json_schema()
        else:
            # Fallback: try tool.to_json_schema() or minimal empty schema
            schema = getattr(tool, "to_json_schema", lambda: {
                "type": "object",
                "properties": {},
                "required": []
            })()
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            },
        }


    @property
    def _llm_type(self) -> str:
        return "Custom Auzure OpenAI Chat API"

    @property
    def model(self) -> str:
        return "AzureOpenAI_" + self.engine
