# Personalized Dialogue Agent Without Persona Descriptions
*By Charis Chan, Joyce Ching, & Inderpal Kaur*

*5th Year MIDS, w266 Spring 2021 Final Project*

## About
Our goal with this project was to create a personalized dialog agent that doesn’t require human-generated personality descriptions as inputs for personalization. Dialog agents of this kind could be applied to a much broader array of contexts, including film/TV scripts, interviews, podcasts, etc., because they would only require indicators for who is speaking at a given time to learn and emulate specific personalities. We based our model on [HuggingFace's winning entry](https://github.com/huggingface/transfer-learning-conv-ai) for ConvAI2 and added a third personality classification head (a “Persona” head) to the double-headed [TransferTransfo](https://arxiv.org/pdf/1901.08149.pdf) model. The Persona head tries to learn the personalities associated with each conversation in place of needing human-generated personality descriptions. For more on our results, check out our final paper [here](https://github.com/ikaur9/persona_chatbot/blob/main/266_Final_Paper_CC_JC_IK.pdf).

## Instructions

**MODEL TYPE:**

- `baseline`
- `baseline_named`
- `triplehead`

**DATA SIZE:**

- `10`
- `100`

To train a model, run the following commands:

`> pip install -r requirements.txt`

`> cd model_[MODEL TYPE]`

`> python train_[MODEL TYPE]_[DATA SIZE].py [--args]`

To test a pre-trained model, run the following command in the same directory:

`> python train_[MODEL TYPE]_[DATA SIZE].py --model_checkpoint [CHECKPOINT FOLDER] --test --n_epochs 0 [--args]`

To interact with a pre-trained model, run the following command in the same directory:

`> python interact_[MODEL TYPE]_[DATA SIZE].py --model_checkpoint [CHECKPOINT FOLDER] [--args]`
