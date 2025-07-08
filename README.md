# Bonus Assignment – CS Assignment Submission

**Student Name:** Vaishnavi Gopi  
**Student ID:** [700754518]  


---

## 📌 Overview

This repository contains solutions for the Bonus Assignment that consists of two parts:

1. **Question Answering using Transformers**  
2. **Digit-Class Controlled Image Generation with Conditional GAN (cGAN)**


---

## ✅ Question 1: Question Answering with Transformers

### 🔧 Setup

Install dependencies:
```bash
pip install transformers torch
```

### 1️⃣ Basic Pipeline Setup

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering")

context = "Charles Babbage is known as the father of the computer. He invented the Analytical Engine."
question = "Who is known as the father of the computer?"

result = qa_pipeline(question=question, context=context)
print(result)
```

✅ **Expected Output:**
```json
{
  'answer': 'Charles Babbage',
  'score': 0.85,
  'start': 0,
  'end': 16
}
```

---

### 2️⃣ Use a Custom Pretrained Model

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = "Charles Babbage is known as the father of the computer. He invented the Analytical Engine."
question = "Who is known as the father of the computer?"

result = qa_pipeline(question=question, context=context)
print(result)
```

✅ **Expected Output:** Answer should still be `"Charles Babbage"` with a confidence score above 0.70.

---

### 3️⃣ Test on Your Own Example

```python
context = "Python is a powerful programming language. It is widely used in AI and data science."

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

question1 = "What is Python?"
question2 = "Where is Python commonly used?"

print(qa_pipeline(question=question1, context=context))
print(qa_pipeline(question=question2, context=context))
```

---

## ✅ Question 2: Digit-Class Controlled Image Generation with Conditional GAN

### 🧠 Objective

Implement a Conditional GAN (cGAN) that generates MNIST digits based on given digit labels (0–9).

### 🔨 Steps

1. Modify the **Generator** to accept noise + digit label.
2. Modify the **Discriminator** to accept image + label.
3. Train on **MNIST dataset** using label embedding.
4. Generate images by feeding noise + fixed label (0–9).

### 📊 Sample Code Snippet (simplified)

```python
# Label embedding and concatenation for Generator input
label_emb = nn.Embedding(10, 10)
gen_input = torch.cat((noise, label_emb(labels)), dim=1)

# Label embedding and concatenation for Discriminator input
label_emb_img = label_emb(labels).view(labels.size(0), 10, 1, 1)
label_emb_img = label_emb_img.expand(-1, 10, 28, 28)
disc_input = torch.cat((real_or_fake_img, label_emb_img), dim=1)
```

---

### 📈 Expected Output

- A **row of 10 generated digits**, one for each label from 0 to 9.
- cGAN should learn to generate specific digits per label.
- Quality of digits and label accuracy should improve with training.

---

## ✍️ Short Answers

### Q1: How does a Conditional GAN differ from a vanilla GAN?

A **Conditional GAN** (cGAN) takes an additional label or condition as input, allowing control over generated outputs. In contrast, a vanilla GAN generates outputs based on noise only.

📌 **Real-World Application**:  
In **image-to-image translation**, such as turning sketches into colored drawings based on object type (e.g., apple vs. banana), conditioning helps generate the correct output.

---

### Q2: What does the discriminator learn in an image-to-image GAN?

The **Discriminator** learns to distinguish between real image-label pairs and fake image-label pairs.

📌 **Why is pairing important?**  
It ensures the model learns the correct relationship between **input condition** (e.g., label or sketch) and **output image**, maintaining semantic consistency.

---


