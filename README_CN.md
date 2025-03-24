# ReasonGraph: Visualisation of Reasoning Methods and Extended Inference Paths

<p align="center">
  <b>语言</b>
</p>

<p align="center">
  <a href="README.md">English</a> •
  <a href="README_CN.md">中文</a>
</p>

<p align="center">
  <b>目录</b>
</p>

<p align="center">
  <a href="#news">🚀 新闻</a> •
  <a href="#todo">✏️ 计划</a> •
  <a href="#introduction">✨ 介绍</a>
</p>

<p align="center">
  <a href="#examples">👀 示例</a> •
  <a href="#quick use">🎨 快速使用</a> •
  <a href="#environment">🖥️ 环境配置</a>
</p>

<p align="center">
  <a href="#citation">📌 引用</a> •
  <a href="#license">🔖 许可</a>
</p>

<p align="center">
  <b>链接</b>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/ZongqianLi/ReasonGraph">在线实例</a> •
  <a href="https://arxiv.org/abs/2503.03979">论文</a> •
  <a href="https://discord.gg/tA9DV7Fjzz">Discord群组交流</a> •
  <a href="https://forms.office.com/r/ebBSAKqHwj">问卷反馈</a>
</p>

<div id="news">&nbsp;</div>



## 🚀 新闻

- **[2025.03.24]** ⭐ 支持推理模型长输出的可视化！！！
- **[2025.03.18]** 支持更多LLM供应商和模型。支持两种语言，英语和中文。
- **[2025.03.18]** 在[调查问卷](https://forms.office.com/r/ebBSAKqHwj)中打分和提供反馈（约需30秒）。
- **[2025.03.17]** 通过Huggingface页面尝试[在线实例](https://huggingface.co/spaces/ZongqianLi/ReasonGraph)！
- **[2025.03.14]** 加入我们的[Discord](https://discord.gg/tA9DV7Fjzz)群组交流！
- **[2025.03.07]** [论文](https://arxiv.org/abs/2503.03979)已上传至Arxiv。
- **[2025.02.22]** 创建了Github页面。

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="todo">&nbsp;</div>



## ✏️ 计划 & 所需帮助

- [ ] 上传示例视频。

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="introduction">&nbsp;</div>



## ✨ 介绍

**ReasonGraph**是一个开源网页平台，用于可视化和分析LLM的推理过程。

**功能:**

- **推理路径可视化:** 将文本格式的推理输出转化为流程图。
- **长篇推理:** 支持可视化推理模型的长输出。
- **元推理:** 内置功能，允许模型自行选择最合适的推理方法。

**特点:**

- **模型支持:** 集成了来自Anthropic、OpenAI、Google、Grok、Deepseek、Qwen和Together.AI等主要LLM提供商的50多种先进模型。
- **推理方法:** 实现了主流推理方法，包括顺序方法和基于树的方法。
- **模块化框架:** 标准化API，便于集成新的推理方法和模型。
- **适合初学者:** 直观的页面设计，具有可视化更新和简单配置。
- **多语言:** 支持多种语言，并易于扩展以支持更多语言。

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="examples">&nbsp;</div>



## 👀 示例

<details>
<summary><strong>示例视频：</strong></summary>

</details>

<div>&nbsp;</div>

<details open>
<summary><strong>页面截图:</strong></summary>

<p align="left">
  <img src="./figures/UI.png" width="80%">
</p>

</details>

<div>&nbsp;</div>

<details open>
<summary><strong>推理模型长输出的可视化:</strong></summary>

输入: Give me two suggestions for transitioning from a journalist to a book editor? Use sequential and tree-based thinking and refinement loops.  
模型: DeepSeek的deepseek-reasoner

<p align="left">
  <img src="./figures/long.png" width="100%">
</p>

</details>

<div>&nbsp;</div>

<details open>
<summary><strong>顺序推理方法的可视化：</strong></summary>

思维链 Chain of Thoughts (左上), 自我完善 (中上), 由简至繁 Least-to-most (右上), 自我一致性 Self-consistency (左下):

<p align="left">
  <img src="./figures/sequence_example.png" width="80%">
</p>

</details>

<div>&nbsp;</div>

<details open>
<summary><strong>基于树的推理方法的可视化：</strong></summary>

纯文本 Plain text (上), 束搜索 Beam Search (中), 思维树 Tree of Thoughts (下):

<p align="left">
  <img src="./figures/tree_example_2.png" width="80%">
</p>

</details>

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="quick use">&nbsp;</div>



## 🎨 快速使用

<details open>
<summary><strong>尝试在线实例：</strong></summary>

#### 1. 前往网页：https://huggingface.co/spaces/ZongqianLi/ReasonGraph

</details>

<div>&nbsp;</div>

<details>
<summary><strong>推理模型长输出的可视化:</strong></summary>

#### 1. 选择一个推理模型并输入其API Key。

- API提供商：Deepseek；模型：deepseek-reasoner
- API提供商：Qwen；模型：qwq-plus

#### 2. 输入Claude的API Key。

#### 3. 点击“长推理”。

</details>

<div>&nbsp;</div>

<details>
<summary><strong>本地使用：</strong></summary>

#### 1. 根据下方🖥️环境配置部分配置环境。

#### 2. 进入根目录：

<absolute_path>/ReasonGraph/

#### 3. 输入API key:

如果您不输入API key，界面仍然可以正常运行，但您将无法使用相应的模型进行推理。

<absolute_path>/ReasonGraph/api_keys.json

```
{
    "anthropic": "<to be filled>",
    "openai": "<to be filled>",
    "google": "<to be filled>",
    "together": "<to be filled>"
}
```

#### 4. 在终端中使用一行代码即可运行程序：

```
python app.py
```

#### 5. 打开浏览器并访问输出中显示的本地URL。
```
 * Running on all addresses (X.X.X.X)
 * Running on http://XXX.X.X.X:XXXX
 * Running on http://XX.XXX.XXX.XXX:XXXX
```

</details>

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="environment">&nbsp;</div>



## 🖥️ 环境配置

```
python==3.11.8
requests==2.31.0
openai==1.63.2
together==1.4.1
flask==3.1.0
google==3.0.0
google-genai==1.2.0
google-generativeai==0.8.4
```

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="citation">&nbsp;</div>



## 📌 引用

```
@misc{li2025reasongraphvisualisationreasoningpaths,
      title={ReasonGraph: Visualisation of Reasoning Paths}, 
      author={Zongqian Li and Ehsan Shareghi and Nigel Collier},
      year={2025},
      eprint={2503.03979},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.03979}, 
}
```

<div>&nbsp;</div>
<div>&nbsp;</div>
<div id="license">&nbsp;</div>



## 🔖 许可

```

```











