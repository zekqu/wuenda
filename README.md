#WuTeachingAI


#  教你快速上手AI应用——吴恩达AI系列教程(1)


人工智能风靡全球,它的应用已经渗透到我们生活的方方面面,从自动驾驶到智能家居,再到医疗辅助和量化交易等等。他们逐渐改变了我们的生活方式,然而,对于许多人来说,AI仍然是一个神秘且无法理解的领域。

为了帮助更多的人理解并掌握AI技术,更享受AI带给人们便捷的服务,吴恩达博士开设了一系列的AI教程。

在这篇博客中,我们将介绍吴恩达AI系列教程的第一部分,教你如何快速上手AI应用——我们将学习如何利用AI创造一个披萨店的客服人员,通过和它的对话我们可以购买需要的披萨。无论你是AI领域的初学者,还是有一定基础想要进一步提升的开发者。我们都能通过引导你让你在AI世界中发现自己的道路。




## 介绍吴恩达博士


![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/7.jpg)

吴恩达（英语：Andrew Ng，1976年4月18日—）是斯坦福大学计算机科学系和电气工程系的客座教授，曾任斯坦福人工智能实验室主任。

2011年，吴恩达在谷歌创建了谷歌大脑项目

2014年5月16日，吴恩达加入百度，负责“百度大脑

2017年12月，吴恩达宣布成立人工智能公司Landing.ai，担任公司的首席执行官。

5月初，DeepLearning.ai 创始人吴恩达联合 OpenAI 推出入门大模型学习的经典课程《ChatGPT Prompt Engineering for Developers》，迅速成为了大模型学习的现象级课程，获得极高的热度。后续，吴恩达教授又联合 LangChain、Huggingface 等机构联合推出了多门深入学习课程，助力学习者全面、深入地学习如何使用大模型并基于大模型开发完整、强大的应用程序。



## **功能演示**


让我们先来看看我们做出来的的AI披萨店客服是怎样回答问题的:

![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/123.gif)

 我们可以看到当我们点需要的pizza的时候,它会问你详细的尺寸并且告诉你相应的钱是多少。


##  如何应用



首先我们要设置一个 Openai Python包

```python
import os
import openai
from dotenv import load_dotenv , find_dotenv 
_ = load_dotenv(find_dotenv())                                                          
```


同时我们要输入自己的OPENAI_API_KEY,可以去Openai官方获取,然后你只需要把你的API_KEY填进这里就行了。
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/10.png)
<br>
<br>
像 ChatGPT 这样的聊天模型实际上是组装成以一系列消息作为输入，并返回一个模型生成的消息作为输出的。虽然聊天格式的设计旨在使这种多轮对话变得容易，但我们通过之前的学习可以知道，它对于没有任何对话的单轮任务也同样有用。


接下来，我们将定义两个辅助函数。第一个是单轮的，我们将prompt放入看起来像是某种用户消息的东西中。另一个则传入一个消息列表。这些消息可以来自不同的角色，我们会描述一下这些角色。


第一条消息是一个系统消息，它提供了一个总体的指示，然后在这个消息之后，我们有用户和助手之间的交替。如果你曾经使用过 ChatGPT 网页界面，那么你的消息是用户消息，而 ChatGPT 的消息是助手消息。系统消息则有助于设置助手的行为和角色，并作为对话的高级指示。

你可以想象它在助手的耳边低语，引导它的回应，而用户不会注意到系统消息。

因此，作为用户，如果你曾经使用过 ChatGPT，你可能不知道 ChatGPT 的系统消息是什么，这是有意为之的。系统消息的好处是为开发者提供了一种方法，在不让请求本身成为对话的一部分的情况下，引导助手并指导其回应。


- **定义第一个辅助函数**
<br>
<br>
这两个函数是用于聊天的自动补全。

```python
def get_completion(prompt , model ='gpt-3.5-turbo'):
    messages=[{'role': 'user' , 'content':prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message['content']

def get_completion_from_messages(messages,model='gpt-3.5-turbo',temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature 
    )
    print(response.choices[0].message)
    return response.choices[0].message['content']                                                           
```


- **定义另一个辅助函数**


它将从下面构建的用户界面中收集提示,
然后将其追加到一个名为上下文的列表中,并每次使用上下文调用模型。这样他就会不断的增长。

                                               
```python

def collect_messages(_):
    prompt = inp.value_input
    inp.value = ''
    context.append({'role':'user', 'content':f"{prompt}"})
    response = get_completion_from_messages(context) 
    context.append({'role':'assistant', 'content':f"{response}"})
    panels.append(
        pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
    panels.append(
        pn.Row('Assistant:', pn.pane.Markdown(response, width=600, style={'background-color': '#F6F6F6'})))

    return pn.Column(*panels)                                                            
```

两个辅助函数定义显示如下:
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/5.jpg)
<br>
<br>
定义了两个函数之后,我们通过**context**来描述prompt,通过prompt让客服知道自己的工作是什么,以及披萨店商品的价格和基本规则。


同时我们可以设置并运行这种UI以显示我们的AI客服。

- **设置prompt语句与UI界面**

                                               
```python

import panel as pn  # GUI
pn.extension()

panels = [] # collect display 

context = [ {'role':'system', 'content':"""
You are OrderBot, an automated service to collect orders for a pizza restaurant. \
You first greet the customer, then collects the order, \
and then asks if it's a pickup or delivery. \
You wait to collect the entire order, then summarize it and check for a final \
time if the customer wants to add anything else. \
If it's a delivery, you ask for an address. \
Finally you collect the payment.\
Make sure to clarify all options, extras and sizes to uniquely \
identify the item from the menu.\
You respond in a short, very conversational friendly style. \
The menu includes \
pepperoni pizza  12.95, 10.00, 7.00 \
cheese pizza   10.95, 9.25, 6.50 \
eggplant pizza   11.95, 9.75, 6.75 \
fries 4.50, 3.50 \
greek salad 7.25 \
Toppings: \
extra cheese 2.00, \
mushrooms 1.50 \
sausage 3.00 \
canadian bacon 3.50 \
AI sauce 1.50 \
peppers 1.00 \
Drinks: \
coke 3.00, 2.00, 1.00 \
sprite 3.00, 2.00, 1.00 \
bottled water 5.00 \
"""} ]  # accumulate messages


inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Chat!")

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

dashboard                                                                               
```

这里有上下文,并包含菜单的系统消息,然后我们就可以执行这个命令了。
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/6.jpg)
<br>
<br>
我们就可以和披萨店的AI客服进行对话了！你可以和他确认任何你想要的pizza。

与此同时我们还可以要求模型创建一个JSON摘要发给订餐系统。

所以我们现在需要追加另一个系统消息,他是另外一条prompt,我们想要的是刚刚订单的JSON摘要,这个摘要需要包含刚才订单的所有内容。

在这种订单任务中,我们会使用一个比较低的temperature,让模型的回答尽可能的一致且可预测:

- **订单系统摘要**

```python
messages =  context.copy()
messages.append(
{'role':'system', 'content':'create a json summary of the previous food order. Itemize the price for each item\
 The fields should be 1) pizza, include size 2) list of toppings 3) list of drinks, include size   4) list of sides include size  5)total price '},    
)
 #The fields should be 1) pizza, price 2) list of toppings 3) list of drinks, include size include price  4) list of sides include size include price, 5)total price '},    

response = get_completion_from_messages(messages, temperature=0)
print(response)                                                       
```

<br>
<br>
同样的,我们可以翻译英文prompt做一个中文的披萨店客服,只需要我们把我们的prompt语句换成中文就可以了

- **中文披萨店客服**

```python
import panel as pn  # GUI
pn.extension()

panels = [] # collect display 

context = [{'role':'system', 'content':"""
你是订餐机器人，为披萨餐厅自动收集订单信息。
你要首先问候顾客。然后等待用户回复收集订单信息。收集完信息需确认顾客是否还需要添加其他内容。
最后需要询问是否自取或外送，如果是外送，你要询问地址。
最后告诉顾客订单总金额，并送上祝福。

请确保明确所有选项、附加项和尺寸，以便从菜单中识别出该项唯一的内容。
你的回应应该以简短、非常随意和友好的风格呈现。

菜单包括：

菜品：
意式辣香肠披萨（大、中、小） 12.95、10.00、7.00
芝士披萨（大、中、小） 10.95、9.25、6.50
茄子披萨（大、中、小） 11.95、9.75、6.75
薯条（大、小） 4.50、3.50
希腊沙拉 7.25

配料：
奶酪 2.00
蘑菇 1.50
香肠 3.00
加拿大熏肉 3.50
AI酱 1.50
辣椒 1.00

饮料：
可乐（大、中、小） 3.00、2.00、1.00
雪碧（大、中、小） 3.00、2.00、1.00
瓶装水 5.00
"""} ]  # accumulate messages


inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Chat!")

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)                                                    
```

效果如下:
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/8.jpg)
<br>
<br>


- **中文版食品订单的JSON摘要**


```python
messages =  context.copy()
messages.append(
{'role':'system', 'content':'创建上一个食品订单的 json 摘要。\
逐项列出每件商品的价格，字段应该是 1) 披萨，包括大小 2) 配料列表 3) 饮料列表，包括大小 4) 配菜列表包括大小 5) 总价'},    
)

response = get_completion_from_messages(messages, temperature=0)
print(response)                                                      
```


我们刚刚点了一个小的芝士披萨,这个订单已经被记录了下来:
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/9.jpg)
<br>
<br>
由此我们就构建了一个“订餐机器人”,我们可以通过它自动收集用户的信息,接受披萨店的订单。诸如此类的小程序我们可以通过人工智能的能力实现很多,快跟上我们的脚步一起拥抱人工智能吧!