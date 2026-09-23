# Time Complexity Basics and Matrix Multiplication

## Prompt

Please write your entire response in Czech. 

Before solving the problem, first teach me the necessary concepts from the beginning. 
Explain clearly and intuitively: 

* What time complexity is and why we use it. 
* What Big-O notation means. 
* How to determine the Big-O time complexity of an algorithm. 
* What common complexities such as O(1), O(log n), O(n), O(n log n), O(n²) mean. 
* How loops and nested loops affect time complexity. 
* How Big-O describes how an algorithm scales as the input size grows. 

Use simple examples and explain the reasoning step by step. Assume I am a beginner and have little or no prior knowledge of time complexity. 

After teaching these concepts, apply them to the following problem: 

Assume you are given two matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$. 

What is the time complexity of computing their product AB using the standard matrix multiplication algorithm? Express your answer using Big-O notation and briefly explain why. 

Solve the problem step by step based on the concepts you just explained. 
Finally, clearly state the final answer in Big-O notation and give a concise explanation of why this is the correct complexity in 1–2 sentences. 
Do not skip the conceptual explanation before solving the problem. 

---

## Solution

**The time complexity is O(mnp).**

**Reasoning:** 
The resulting matrix after multiplication has m × p elements. Computing each of these elements requires calculating the dot product of a row from the first matrix and a column from the second matrix, which means iterating over n elements (performing n multiplications and n-1 additions). Since we must repeat this operation of length n for all m × p elements of the resulting matrix, the total number of arithmetic steps is proportional to the product of these dimensions, hence O(mnp).