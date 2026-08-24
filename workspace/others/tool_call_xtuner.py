from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP
from xtuner.v1.data_proto.messages import ChatMessages


if __name__ == '__main__':
    # chatml_messages = {
    #     "messages": [
    #         {"role": "system", "content": "这是系统消息"},
    #         {"role": "user", "content": "Hey, what's the temperature in Paris right now?"},
    #         {"role": "assistant", "content": "你是对的",
    #                     "tool_calls": [
    #                             {
    #                                 "id": "call_123",
    #                                 "type": "function",
    #                                 "function": {
    #                                     "name": "get_weather",
    #                                     "arguments": "{\"location\": \"Boston\"}"
    #                                 }
    #                             },
    #                             {
    #                                 "id": "call_456",
    #                                 "type": "function",
    #                                 "function": {
    #                                     "name": "get_weather",
    #                                     "arguments": "{\"location\": \"beijing \"}"
    #                                 }
    #                             }
    #                         ],
    #                         "reasoning_content": ""},
    #         {"role": "tool", "content": "22"},
    #         {"role": "assistant", "content": "你问的特别好"}
    #         ],
    #     "tools": [{
    #                 "type": "function",
    #                 "function": {
    #                     "name": "get_current_temperature",
    #                     "description": "Gets the temperature at a given location.",
    #                     "parameters": {
    #                         "type": "object",
    #                         "properties": {
    #                             "location": {
    #                                 "type": "string",
    #                                 "description": "The location to get the temperature for"
    #                             }
    #                         },
    #                         "required": [
    #                             "location"
    #                         ]
    #                     }
    #                 }
    #             },
    #             {"type": "function", "function": {"name": "get_current_wind_speed", "description": "Get the current wind speed in km/h at a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the wind speed for, in the format \"City, Country\""}}, "required": ["location"]}}}
    #               ],
    # }
    
    chatml_messages = {"messages": [{"role": "system", "content": "You are a function calling AI model. \nYou may call one or more functions to assist with the user query. \nDon't make assumptions about what values to plug into functions.\n\nUse the following pydantic model json schema for each tool call you will make: \n{\"title\": \"FunctionCalls\", \"type\": \"array\", \"properties\": {\"arguments\": {\"title\": \"Arguments\", \"type\": \"object\"}, \"name\": {\"title\": \"Name\", \"type\": \"string\"}}, \"required\": [\"arguments\", \"name\"]}\n\nAt each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task."}, {"role": "user", "content": "Where can I find live giveaways for beta access and games?"}, {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "live_giveaways_by_type", "arguments": "{\"type\": \"beta\"}"}}, {"id": "call_2", "type": "function", "function": {"name": "live_giveaways_by_type", "arguments": "{\"type\": \"game\"}"}}], "reasoning_content": "", "content": "<think></think>\n\n"}], "tools": [{"name": "live_giveaways_by_type", "description": "Retrieve live giveaways from the GamerPower API based on the specified type.", "parameters": {"type": {"description": "The type of giveaways to retrieve (e.g., game, loot, beta).", "type": "str", "default": "game"}}}]}
    chat_template = CHAT_TEMPLATE_MAP['qwen3-vl']
    _messages = ChatMessages(**chatml_messages)
    prompt = _messages.get_prompt(chat_template)
    print(prompt)
    
    # import json

    # jsonl_path='/mnt/shared-storage-user/huanghaian/too_call_demo.jsonl'

    # data=[]
    # with open(jsonl_path, 'rb') as f:
    #     for line in f:
    #         data.append(json.loads(line))

    # for d in data:
    #     _messages = ChatMessages(messages=d['messages'],tools=d.get('tools', None))
    #     prompt = _messages.get_prompt(chat_template)
    #     print(prompt)

