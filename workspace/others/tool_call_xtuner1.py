from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP
from xtuner.v1.data_proto.messages import ChatMessages

if __name__ == '__main__':
    tools = [
        {'type': 'function',
         'function': {
             'name': 'find_user_id_by_name_zip',
             'description': 'Find user id by first name, last name, and zip code. If the user is not found, the function will return an error message. By default, find user id by email, and only call this function if the user is not found by email or cannot remember email.',
             'parameters': {'type': 'object',
                            'properties': {'first_name': {'type': 'string',
                                                          'description': "The first name of the customer, such as 'John'."},
                                           'last_name': {'type': 'string',
                                                         'description': "The last name of the customer, such as 'Doe'."},
                                           'zip': {'type': 'string',
                                                   'description': "The zip code of the customer, such as '12345'."}},
                            'required': ['first_name', 'last_name', 'zip']}}},
        {'type': 'function',
         'function': {
             'name': 'get_order_details',
             'description': 'Get the status and details of an order.',
             'parameters': {'type': 'object',
                            'properties': {'order_id': {'type': 'string',
                                                        'description': "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."}},
                            'required': ['order_id']}}}
    ]
    messages = [
        {'role': 'user',
         'content': "Hi there! I'm looking to return a couple of items from a recent order."},
        {'role': 'assistant',
         'content': 'Would you please provide your email address associated with the account, or share your first name, last name, and zip code?',
         'thinking': 'Okay, the user wants to return some items from a recent order. Let me start by authenticating their identity...'},
        {'role': 'user',
         'content': 'Sure, my name is Omar Anderson and my zip code is 19031.'},
        {'role': 'assistant',
         'content': '',
         'thinking': "Thinking2.1: Since the user didn't provide an email, I should use the find_user_id_by_name_zip function. Let me...",
         'tool_calls': [
             {'function': {'arguments': '{"first_name": "Omar", "last_name": "Anderson", "zip": "19031"}',
                           'name': 'find_user_id_by_name_zip'},
              'id': 'chatcmpl-tool-a9f439084bfc4af29fee2e5105050a38',
              'type': 'function'},
         ]},
        {'content': 'omar_anderson_3203',
         'name': 'find_user_id_by_name_zip',
         'role': 'tool'},
        {'role': 'assistant',
         'content': '',
         'thinking': "Thinking2.2: I have the user ID now. Next, I need to get the order details to proceed with the return. I'll use the get_order_details function...",
         'tool_calls': [
             {'function': {'arguments': '{"order_id": "#W1234567"}',
                           'name': 'get_order_details'},
              'id': 'chatcmpl-tool-1f4e5d6c789a4b2e8c9d0f6a7b8c9d0e',
              'type': 'function'},
         ]},
        {
            'content': '{"order_id": "#W1234567", "status": "Delivered", "items": [{"item_id": "A123", "name": "Wireless Mouse", "returnable": true}, {"item_id": "B456", "name": "Keyboard", "returnable": false}]}',
            'name': 'get_order_details',
            'role': 'tool'},
        {'role': 'assistant',
         'content': 'Thank you, Omar. I have located your order #W1234567, which includes a Wireless Mouse that is returnable and a Keyboard that is not returnable. How would you like to proceed with the return of the Wireless Mouse?',
         'thinking': 'Thinking2.3: I have the order details now. I see that the Wireless Mouse is returnable while the Keyboard is not. I will inform the user about this and ask how they would like to proceed with the return.'}

    ]

    chat_template = CHAT_TEMPLATE_MAP['qwen3-vl']
    _messages = ChatMessages(messages=messages)
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

