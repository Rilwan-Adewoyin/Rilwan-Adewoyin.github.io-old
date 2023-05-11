---
layout: post
title:  Probalistic Answers to Open Ended Questions
date:   2023-03-15 16:40:16
description: Generating Probalistic Answers to From Language Models
tags: nlp
---

This blog post is based on work I have previously done while at Alan Turing Institute.

## Context

THE TASK: Given temporal causal graph and you want to determine the weights or existence of weights in a graph $G$, where the nodes and edges reflect a property in the real world such: 1) Nodes: Government Spending on specific budget items (bi) and socio-economic indicators (sei). Edges: reflect the degree to which a bi/sei affects a sei 2) Nodes: financial institutions and their credit profile Edges: some notion of dependency

Therefore we are aiming to use prompt engineering in order to ascertain the weights of edges based on textual profiles of nodes.

## Quick Background Info

- Why not simply prompt the model or a yes no answer directly and look at the perplexity of those two responses?
-- As discussed (in this recent work)[] Some language models have been trained on datasets (PILE) which contain a significantly uneven distribution of yes/no answers.
-- As discussed in these (recent works)[], LMs tend to perform better when the question is open-ended as opposed to a yes/no question. Intuitively, this follows the findings that allowing an LM to explain its reasoning e.g. "chain of thought" improves the output

- One important factor to consider is if you will model N^th order effects in your model?
-- When designing your prompt using terms such as "directly" or "indirectly" can lead to greatly different results
-- If you choose to only model 1st order effects, ensure to use "directly" in prompts


## Method : 

- Design your open-ended prompt which can elicit an answer to the question
- Create a secondary prompt which reduces the model's answer to a positive or negative sentiment. E.g. "The previous answer, represents an [Agreement or Disagreement]?"
- Evaluate the likelihood (perplexity) of either Agreement or Disagreement appearing in that final position. Ideally you want to choose two words which have the same number of tokens.
- Use some function of the relative likelihood of either term (e.g. Agreement and Disagreement) to determine the weight of the edge


#### Extending to 2nd order effects
In order to differentiate between if a 1st or 2nd order effect occur between two nodes:
-- Directly ask the LLM if the relationship between two nodes can be considred direct or indirect.
-- Ask open-ended question explaining the relationship between two nodes, then ask follow up question to reduce previous answer to "direct effect" or "indirect effect".
-- Use a similar method as above to get relative likelihoods for "direct effect" and "indirect effect"
-- Use some functions of the above likelihoods to determine weights for edges which reflect 1st order effects and 2nd order effects
