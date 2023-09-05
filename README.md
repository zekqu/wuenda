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


 ## 如何使用

 Cloud Studio 在线编程平台支持使用CODING (opens new window)账号和 GitHub 账号，以及微信登录，可以在登录 (opens new window) 界面输入相应的账号登录前往 Web IDE，这里使用微信登录。
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/5/5-3.png)

 Cloud Studio 是基于浏览器的集成式开发环境(IDE)，为开发者提供了一个永不间断的云端工作站。用户在使用CloudStudio 时无需安装，随时随地打开浏览器就能在线编程。

 这里 Cloud Studio 提供了多种的模版,我们就选择jupter模版、。
同时我们运行然后我们就可以开始创建专属我们的披萨店AI客服了。

首先我们要设置一个 Openai Python包

```shell
import os
import openai
from dotenv import load_dotenv , find_dotenv 
_ = load_dotenv(find_dotenv())                                                          
```


同时我们要输入自己的OPENAI_API_KEY,可以去Openai官方获取,然后你只需要把你的API_KEY填进这里就行了。
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/4.png)

之后我们需要定义两个函数一个是getCompletion函数,我们在里面给定了prompt,将提示放入了类似于某种用户消息的东西中,这是因为模型是一个聊天模型,所以用户的信息是输入,而模型是输出的一种表达。

```shell
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

同时我们通过定义另一个辅助函数,它将从下面构建的用户界面中收集提示,
然后将其追加到一个名为上下文的列表中,并每次使用上下文调用模型。这样他就会不断的增长。

```shell
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

![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/5.jpg)
定义了两个函数之后,我们可以设置并运行这种UI以显示我们的AI客服

```

同时我们通过定义另一个辅助函数,它将从下面构建的用户界面中收集提示,
然后将其追加到一个名为上下文的列表中,并每次使用上下文调用模型。这样他就会不断的增长。

```shell
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

所以这里有上下文,并包含菜单的系统消息,然后我们就可以执行这个命令了。
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/6.jpg)
我们就可以和披萨店的AI客服进行对话了！你可以和他确认任何你想要的pizza。

接下来立即前往 Cloud Studio 体验一下创建自己的AI应用吧!

[![Cloud Studio Template](https://cs-res.codehub.cn/common/assets/icon-badge.svg)](https://cloudstudio.net/templates/rCtBrwUdoOo)