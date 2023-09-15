#WuTeachingAI


#  教你快速上手AI应用——吴恩达AI系列教程


人工智能风靡全球,它的应用已经渗透到我们生活的方方面面,从自动驾驶到智能家居,再到医疗辅助和量化交易等等。他们逐渐改变了我们的生活方式,然而,对于许多人来说,AI仍然是一个神秘且无法理解的领域。

为了帮助更多的人理解并掌握AI技术,更享受AI带给人们便捷的服务,吴恩达博士开设了一系列的AI教程。接下来我们会通过几个项目的教程让大家学会如何用AI解决生活中的一些小问题,在AI时代来临之际,教会大家如何利用好这一有力的武器。





## 介绍吴恩达博士


![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/7.jpg)

吴恩达（英语：Andrew Ng，1976年4月18日—）是斯坦福大学计算机科学系和电气工程系的客座教授，曾任斯坦福人工智能实验室主任。

2011年，吴恩达在谷歌创建了谷歌大脑项目

2014年5月16日，吴恩达加入百度，负责“百度大脑

2017年12月，吴恩达宣布成立人工智能公司Landing.ai，担任公司的首席执行官。

5月初，DeepLearning.ai 创始人吴恩达联合 OpenAI 推出入门大模型学习的经典课程《ChatGPT Prompt Engineering for Developers》，迅速成为了大模型学习的现象级课程，获得极高的热度。后续，吴恩达教授又联合 LangChain、Huggingface 等机构联合推出了多门深入学习课程，助力学习者全面、深入地学习如何使用大模型并基于大模型开发完整、强大的应用程序。


## **吴恩达AI系列第一课————教你如何利用AI创建一个披萨店客服**

在这篇博客中,我们将介绍吴恩达AI系列教程的第一部分,教你如何快速上手AI应用——我们将学习如何利用AI创造一个披萨店的客服人员,通过和它的对话我们可以购买需要的披萨。无论你是AI领域的初学者,还是有一定基础想要进一步提升的开发者。我们都能通过引导你让你在AI世界中发现自己的道路。



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
        pn.Row('Assistant:', pn.pane.Markdown(response, width=600, style={'background-color': '#333333'})))

    return pn.Column(*panels)                                                            
```

两个辅助函数定义显示如下:
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/5.jpg)
<br>
<br>

## 小试牛刀

现在我们尝试告诉模型你是一个说话像莎士比亚的助手。这是我们向助手描述它应该如何表现的方式。然后，第一个用户消息是，给我讲个笑话。接下来的消息是，为什么鸡会过马路？然后最后一个用户消息是，我不知道。
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/11.jpg)
<br>
<br>
让我们做另一个例子。助手的消息是，你是一个友好的聊天机器人，第一个用户消息是，嗨，我叫Isa。我们想要得到第一个用户消息。
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/14.jpg)
<br>
<br>

让我们再试一个例子。系统消息是，你是一个友好的聊天机器人，第一个用户消息是，是的，你能提醒我我的名字是什么吗？

如上所见，模型实际上并不知道我的名字。
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/12.jpg)
<br>
<br>
因此，每次与语言模型的交互都是一个独立的交互，这意味着我们必须提供所有相关的消息，以便模型在当前对话中进行引用。如果想让模型引用或 “记住” 对话的早期部分，则必须在模型的输入中提供早期的交流。我们将其称为上下文。让我们试试。
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/9/13.jpg)
<br>
<br>
现在我们已经给模型提供了上下文，也就是之前的对话中提到的我的名字，然后我们会问同样的问题，也就是我的名字是什么。因为模型有了需要的全部上下文，所以它能够做出回应，就像我们在输入的消息列表中看到的一样。
<br>
<br>
## AI披萨店客服

由此我们知道,我们可以通过**context**来描述prompt,在披萨店的规则也是一样:通过prompt让客服知道自己的工作是什么,以及披萨店商品的价格和基本规则。



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




## **吴恩达AI系列第二课————教你如何利用 Langchain 封装一本书**


在这篇博客中,我们将介绍吴恩达AI系列教程的第二部分,教你如何快速上手AI应用——我们将学习如何通过langchain构建向量数据库从而封装一本书,然后我们可以通过提问获取这本书相应的问题。

无论你是AI领域的初学者,还是有一定基础想要进一步提升的开发者。我们都能通过引导你让你在AI世界中发现自己的道路。



## **功能演示**


让我们先来看看我们封装书籍后我们现在需要有防晒效果的全部衬衫以及对这些衬衫做一个总结:
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/10/9.png)
<br>
<br>
 我们可以看到模型会把所有的防晒衬衫全部信息表出来,并且会有一句很精炼的总结。


##  如何应用



 **首先我们要设置环境的配置**

- 安装 langchain
```python
pip install langchain                                                        
```


- 安装 docarray
```python
pip install docarray                                                      
```

- 安装 tiktoken
```python
pip install tiktoken                                                    
```

- 同时我们要设置自己的API_KEY环境变量

```python
import openal
import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
openai.api_base= “https://service-4v8atua4-1259057771.sa.apiaw.tencentcs.com/v1”
             
```

只需要您将API_KET填写在里面即可
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/10/1.png)
<br>
<br>

**我们先明白如何通过 langchain 调用模型**


LangChain 是一个强大的框架，旨在帮助开发人员使用语言模型构建端到端的应用程序。它提供了一套工具、组件和接口，可简化创建由大型语言模型 (LLM) 和聊天模型提供支持的应用程序的过程。LangChain 可以轻松管理与语言模型的交互，将多个组件链接在一起，并集成额外的资源，例如 API 和数据库。

而 langchain 里面的模型主要分为三个类型:

**LLM（大型语言模型）**：这些模型将文本字符串作为输入并返回文本字符串作为输出。它们是许多语言模型应用程序的支柱。

 **聊天模型( Chat Model)**：聊天模型由语言模型支持，但具有更结构化的 API。他们将聊天消息列表作为输入并返回聊天消息。这使得管理对话历史记录和维护上下文变得容易。

 **文本嵌入模型(Text Embedding Models)**：这些模型将文本作为输入并返回表示文本嵌入的浮点列表。这些嵌入可用于文档检索、聚类和相似性比较等任务。


- **首先调用LLM模型**

```python
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) #读取环境变量                                                 
```
然后我们问 llm 模型如何评价人工智能,他就会通过 langchain 自动调用你的 OPENAI_API_KEY 告诉你 llm 模型生成的答案:

```python
from langchain.llms import OpenAI
# auto read OPENAI_API_KEY
llm = OpenAI(model_name="text-davinci-003",max_tokens=1024)
llm("怎么评价人工智能")                                               
```
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/10/2.png)
<br><br>
我们可以从图中看到,模型通过调用api接口回答了“怎么评价人工智能的回答”






## 导入 embedding 模型和向量存储组件

接下来我们会学习使用 embedding 模型和向量数据库做一个存储,利用 langchain 将我们的书籍进行封装。

- **我们首先把 langchain 的一些功能的包加载下来**:

```python
from langchain.chains import RetrievalQA #检索QA链，在文档上进行检索
from langchain.chat_models import ChatOpenAI #openai模型
from langchain.document_loaders import CSVLoader #文档加载器，采用csv格式存储
from langchain.vectorstores import DocArrayInMemorySearch #向量存储
from IPython.display import display, Markdown #在jupyter显示信息的工具                                                         
```
在本次小项目中,我们的数据使用 Dock Array 内存搜索向量存储中,作为一个内存向量存储，不需要连接外部数据库

- **读取我们的户外户外服装目录书籍**

我们首先可以在github仓库里获取该书籍[OutdoorClothingCatalog_1000.csv](https://github.com/Ryota-Kawamura/LangChain-for-LLM-Application-Development.git)

下载到本地后可以将该书上传到我们的 Cloud Studio 中,只需拖动即可上传:

- **加载书籍文件**

```python
#读取文件
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
#查看数据
import pandas as pd
data = pd.read_csv(file,header=None)
data                                                  
```



![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/10/3.png)
<br>
<br>
可以看到我们通过查看数据发现他提供了一个户外服装的CSV文件,文件中有很多种类衣服与他们的介绍,我们可以将这些与语言模型结合使用

- **创建向量存储**


我们通过导入索引,即向量存储索引创建器:

```python
from langchain.indexes import VectorstoreIndexCreator #导入向量存储索引创建器                                                     
```


```python
'''
将指定向量存储类,创建完成后，我们将从加载器中调用,通过文档记载器列表加载
'''

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])                                                 
```

之后问他一个问题,例如我们可以让他列一下带有防晒衣的衬衫,然后给我们总结一下


```python

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."
response = index.query(query)#使用索引查询创建一个响应，并传入这个查询
display(Markdown(response))#查看查询返回的内容                                           
```

我们就可以看到结果:
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/10/4.png)
<br>
<br>
我们发现得到了一个 Markdown 的表格,其中包含了所有带有防晒衣的衬衫的名字与描述,还通过llm的总结得到了一个不错的总结“our shirts provide UPF 50+ sun protection, blocking 98% of the sun's harmful rays. The Men's Tropical Plaid Short-Sleeve Shirt is made of 100% polyester and is wrinkle-resistant”

## 语言模型与文档的结合使用

我们上面完成了一个书籍的存储以及调用语言模型回答里面的问题,而在我们的实际生活中如果想让语言模型与许多文档结合,怎么才能让他回答其中所有的内容呢?我们可以通过embedding和向量存储可以实现
- **embedding**
文本片段创建数值表示文本语义，相似内容的文本片段将具有相似的向量，这使我们可以在向量空间中比较文本片段

- **向量数据库**
向量数据库是存储我们在上一步中创建的这些向量表示的一种方式，我们创建这个向量数据库的方式是用来自传入文档的文本块填充它。 当我们获得一个大的传入文档时，我们首先将其分成较小的块，因为我们可能无法将整个文档传递给语言模型，因此采用分块 embedding 的方式储存到向量数据库中。这就是创建索引的过程。

通过运行时使用索引来查找与传入查询最相关的文本片段，然后我们将其与向量数据库中的所有向量进行比较，并选择最相似的n个，返回语言模型得到最终答案

首先我们通过创建一个文档加载器,通过CSV格式加载



```python

#创建一个文档加载器，通过csv格式加载
loader = CSVLoader(file_path=file)
docs = loader.load()   

```
然后我们可以查看一下单独的文档,可以发现每个文档都对应了CSV中的一个块



![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/10/5.png)
<br>
<br>
之后我们可以对文档进行分块和 embedding ,当文档非常大的时候,我们需要对文档进行分块处理,因为如果在较大文件的情况下我们的索引和提取会占用较大的内存使得效率变得很低,但是在此次小实验中,我们的文档并不大所以不需要进行分块处理,仅仅去做一个 embedding 就可以了


```python

'''
因为这些文档已经非常小了，所以我们实际上不需要在这里进行任何分块,可以直接进行embedding
'''

from langchain.embeddings import OpenAIEmbeddings #要创建可以直接进行embedding，我们将使用OpenAI的可以直接进行embedding类
embeddings = OpenAIEmbeddings() #初始化  
embed = embeddings.embed_query("Hi my name is Harrison")#让我们使用embedding上的查询方法为特定文本创建embedding
print(len(embed))#查看这个embedding，我们可以看到有超过一千个不同的元素

```

我们的这个 embbding 可以查看到一千多个不同的元素,每个元素都是映射的数字值,组合起来就创建了这段文本的总体数值的表示  

- **接下来我们将 embedding 存储在向量存储中**

为刚才的文本创建embedding，准备将它们存储在向量存储中，使用向量存储上的 from documents 方法来实现。
该方法接受文档列表、嵌入对象，然后我们将创建一个总体向量存储

```python

db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

```

运行这个程序,我们就能得到存储了书籍的向量数据库了


![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/10/6.png)

这时我们可以通过一个类似查询的文本传会给向量数据库,我们可以让他返回一些文本:

```python

query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)#使用这个向量存储来查找与传入查询类似的文本，如果我们在向量存储中使用相似性搜索方法并传入一个查询，我们将得到一个文档列表
len(docs)

```


![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/10/7.png)
<br>
<br>
可以看到返回了四个文档,同时我们可以打开第一个文档:
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/10/8.png)
<br>
<br>
你可以看到,第一个文档的确是关于防晒的衬衫相关的内容


## 如何回答跟我们文档相关的问题

要回答和我们文档相关的问题我们需要通过检索器支持查询和返回文档的方法,并且通过导入语言模型的方式进行文本生成并返回自然语言响应

所以我们应该先做的第一步是**创建检索器通用接口以及导入语言模型**


```python

retriever = db.as_retriever() #创建检索器通用接口
llm = ChatOpenAI(temperature = 0.0,max_tokens=1024) #导入语言模型
qdocs = "".join([docs[i].page_content for i in range(len(docs))]) 
# 将合并文档中的所有页面内容到一个变量中


```


- **通过调用语言模型来问问题**

```python

response = llm.call_as_llm("Question: Please list all your shirts with sun protection in a table in markdown and summarize each one.") #列出所有具有防晒功能的衬衫并在Markdown表格中总结每个衬衫的语言模型

```

然后我们可以通过 markdown 形式查看语言模型通过调用语言模型的总结,以及在文本中存在的关于防晒功能衬衫的所有信息
<br>
<br>
![](https://help-assets-1257242599.cos.ap-shanghai.myqcloud.com/enterprise/2023/10/9.png)
<br>
<br>
这样我们就得到了我们想要的结果!


如果有多个文档，那么我们可以使用几种不同的方法

- **Map Reduce**

将所有块与问题一起传递给语言模型，获取回复，使用另一个语言模型调用将所有单独的回复总结成最终答案，它可以在任意数量的文档上运行。可以并行处理单个问题，同时也需要更多的调用。它将所有文档视为独立的

- **Refine**

用于循环许多文档，际上是迭代的，建立在先前文档的答案之上，非常适合前后因果信息并随时间逐步构建答案，依赖于先前调用的结果。它通常需要更长的时间，并且基本上需要与Map Reduce一样多的调用

- **Map Re-rank**

对每个文档进行单个语言模型调用，要求它返回一个分数，选择最高分，这依赖于语言模型知道分数应该是什么，需要告诉它，如果它与文档相关，则应该是高分，并在那里精细调整说明，可以批量处理它们相对较快，但是更加昂贵

- **Stuff**

将所有内容组合成一个文档

在这里我们就不举太多例子,欢迎各位进入 Cloud Studio 自己体验!!!

