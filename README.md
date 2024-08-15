# BetaGuidance

**BetaGuidance** is a library designed for constrained generation with large language models (LLMs). It offers powerful functionality to control and refine the text generation process, making it easier to extract specific information or generate content within defined constraints.

## Features

### `select()`
The `select()` method enables the language model to choose the most likely outcome from a given set of arguments. This is particularly useful for guiding the LLM towards generating specific, desired results.

**Example:**

```python
prompt = '''There is a request from the user to create a task in the task management system. Please, extract the status of this task that will be placed in the Task Management System.

Example 1: "Обнови статус задачи по проведению пентеста на выполнено."
Answer: "выполнено"

Example 2: "Создай Ване задачу по рефакторингу кода. И поставь статус - в процессе"
Answer: "в процессе"

Example 3: "Покажи мне исполнителей задачи по построению графиков"
Answer: "null"

Example 4: "Создай задачу купить конфет, статус в процессе"
Answer: "'''

context = "Создай задачу купить конфет, статус в процессе"
k = 1

res = guidance_system.select(prompt, context, k)

# prompt - The prompt to be passed to the LLM
# context - The context from which the model should directly quote the generation result
# k - The number of response options (beam search)

# In this example, res will be: ['в процессе']
```

### `gen()`
The `gen()` method is used for text generation with the language model. Its advantage lies in the ability to halt the generation once a specified keyword is produced, ensuring the output stays within the desired bounds.

**Example:**

```python
gen_res = guidance_system.gen("I am a big fan of BMW",
                              stop_keywords="BMW",
                              max_length=100)

# gen_res is:
# ["'s and have owned one for 15 years. I have had the pleasure of driving the new M3 and M5 and they are a blast. I am looking for a used M3 for sale, but I am having a hard time finding one that I like. I am looking for one that is less than 5 years old, and has a manual transmission. I have been to several dealerships, and have looked at several"]
```

### `substring()`
The `substring()` method allows the language model to quote directly from a given text based on the prompt. This is useful when the generation needs to be constrained to a specific part of the input text.

**Example:**

```python
texts = ["What usually has 4 wheels?", 
         "I have been in a country in South Asia called Bang"]

choices_list = [["car", "horse"], 
                ["ladesh", "cheese"]]

output = guidance_system.select(texts, choices_list)

# Output is ['car', 'ladesh']
```
